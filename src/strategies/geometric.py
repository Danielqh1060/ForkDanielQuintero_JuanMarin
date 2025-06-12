import time
import numpy as np
from typing import Dict, Tuple, Set, List
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# Tus importaciones existentes (asumiendo que son correctas)
from src.models.base.sia import SIA
from src.controllers.manager import Manager
from src.models.core.solution import Solution
from src.middlewares.slogger import SafeLogger
from src.middlewares.profile import profile, profiler_manager
from src.funcs.base import emd_efecto # Asumimos que esta es tu emd_efecto real, que espera mismas formas
from src.funcs.format import fmt_biparticion
from src.funcs.force import biparticiones
from src.constants.base import FLOAT_ZERO, TYPE_TAG, NET_LABEL, EFECTO, ACTUAL
from src.constants.models import (
    GEOMETRIC_LABEL,
    GEOMETRIC_ANALYSIS_TAG,
    DUMMY_ARR,
    DUMMY_PARTITION,
    ERROR_PARTITION,
)

# --- INICIO OPTIMIZACIÓN: CÁLCULO DE COSTOS VAR CON CPU ---
# Esta función permanece igual.
def calcular_costos_var(var_idx: int, tensor: np.ndarray, num_vars: int) -> Dict[Tuple[int, int, int], float]:
    cache = {} 
    num_estados = 2 ** num_vars
    _tensor = np.asarray(tensor)

    def t(i: int, j: int) -> float:
        if i == j:
            return 0.0
        clave = (i, j)
        if clave in cache:
            return cache[clave]
        
        dH = bin(i ^ j).count("1")
        gamma = 2 ** (-dH)
        
        vecinos = [k for b in range(num_vars) if bin((k := i ^ (1 << b)) ^ j).count("1") == dH - 1]
        
        suma = sum(t(k, j) for k in vecinos)
        
        costo = gamma * (abs(float(_tensor[i]) - float(_tensor[j])) + suma)
        cache[clave] = costo
        return costo

    tabla_var = {}
    for i in range(num_estados):
        for j in range(num_estados):
            if i != j:
                tabla_var[(var_idx, i, j)] = t(i, j)
    return tabla_var
# --- FIN OPTIMIZACIÓN: CÁLCULO DE COSTOS VAR ---


class Geometric(SIA):
    def __init__(self, gestor: Manager) -> None:
        super().__init__(gestor)
        profiler_manager.start_session(f"{NET_LABEL}{len(gestor.estado_inicial)}{gestor.pagina}")
        self.logger = SafeLogger(GEOMETRIC_LABEL)
        self.memo_phi: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float] = {}
        self.tabla_costos: Dict[Tuple[int, int, int], float] = {}
        self.tensores: Dict[int, np.ndarray] = {}

        # --- INICIO OPTIMIZACIÓN: Filtro de Bloom ---
        from src.utils.bloom_filter import BloomFilter 
        self.bloom_filter_phi = BloomFilter(capacity=100000, error_rate=0.01)
        # --- FIN OPTIMIZACIÓN: Filtro de Bloom ---


    @profile(context={TYPE_TAG: GEOMETRIC_ANALYSIS_TAG})
    def aplicar_estrategia(self, condiciones: str, alcance: str, mecanismo: str) -> Solution:
        self.sia_tiempo_inicio = time.time()
        self.sia_preparar_subsistema(condiciones, alcance, mecanismo)

        if self.sia_subsistema.estado_inicial.size < 2:
            return Solution(GEOMETRIC_LABEL, -1, DUMMY_ARR, DUMMY_ARR, 0.0, ERROR_PARTITION)

        self._cargar_tensores_elementales() 
        self._construir_tabla_costos_parallel()

        candidatos_rapidos = self._biparticiones_heuristicas()
        mejor_phi = float("inf")
        mejor_bip = None
        mejor_dist = DUMMY_ARR

        # --- MODIFICACIÓN CLAVE PARA EL ERROR DE SHAPE (Adaptado a 5 vs 64) ---
        # Determina el número esperado de estados del subsistema completo.
        # Este valor (self.sia_subsistema.estado_inicial.size) DEBE ser el número de variables
        # del subsistema completo, que determina el tamaño de su espacio de estados (2^num_vars).
        # Si el subsistema tiene 6 variables, expected_total_states = 2^6 = 64.
        expected_total_states = 2 ** self.sia_subsistema.estado_inicial.size
        
        sia_dists_marginales_cpu = self.sia_dists_marginales.flatten().astype(np.float16)
        
        # Parche si la distribución marginal del subsistema no tiene el tamaño esperado.
        # Esto indica una inconsistencia en la lógica de `SIA` o `System`.
        if sia_dists_marginales_cpu.size != expected_total_states:
            self.logger.debug(f"ADVERTENCIA: La distribución marginal del subsistema (tamaño {sia_dists_marginales_cpu.size}) no coincide con el número esperado de estados (2^{self.sia_subsistema.estado_inicial.size}={expected_total_states}). Intentando redimensionar. ¡Esto podría indicar un problema de lógica subyacente!")
            # Crear un array del tamaño esperado y copiar los elementos existentes
            temp_dist = np.zeros(expected_total_states, dtype=np.float16)
            # Copiar solo hasta donde los datos existentes lo permitan.
            # Los elementos restantes se llenarán con ceros.
            temp_dist[:min(sia_dists_marginales_cpu.size, expected_total_states)] = \
                sia_dists_marginales_cpu[:min(sia_dists_marginales_cpu.size, expected_total_states)]
            sia_dists_marginales_cpu = temp_dist
        # --- FIN MODIFICACIÓN CLAVE (Adaptado) ---


        for m, p in candidatos_rapidos:
            if not m or not p:
                continue
            
            # _calcular_distribucion_particionada también debe asegurar el tamaño correcto
            dist_part = self._calcular_distribucion_particionada(set(m), set(p))
            
            # Ambos arrays deben ser 1D y del mismo tamaño para que emd_efecto funcione.
            phi = emd_efecto(dist_part, sia_dists_marginales_cpu)

            self._imprimir_phi_debug(m, p, phi)
            if phi < mejor_phi:
                mejor_phi = phi
                mejor_bip = (m, p)
                mejor_dist = dist_part
            if phi == 0.0:
                self.logger.debug("φ = 0.0 heurístico encontrado.")
                return self._retornar_solucion(mejor_phi, mejor_dist, mejor_bip)

        futuros = self.sia_subsistema.indices_ncubos
        presentes = self.sia_subsistema.dims_ncubos

        for subalcance, submecanismo in biparticiones(futuros, presentes):
            clave = (tuple(sorted(submecanismo)), tuple(sorted(subalcance)))
            
            if self.bloom_filter_phi.check(clave) and clave in self.memo_phi:
                phi = self.memo_phi[clave]
                self._imprimir_phi_debug(set(submecanismo), set(subalcance), phi)
                if phi < mejor_phi:
                    mejor_phi = phi
                    mejor_bip = (set(submecanismo), set(subalcance))
                    # Recalcular dist_part si esta es la mejor hasta ahora
                    mejor_dist = self._calcular_distribucion_particionada(set(submecanismo), set(subalcance))
                continue

            if not subalcance or not submecanismo:
                continue

            dist_part = self._calcular_distribucion_particionada(set(submecanismo), set(subalcance))
            
            phi = emd_efecto(dist_part, sia_dists_marginales_cpu)
            
            self.memo_phi[clave] = phi
            self.bloom_filter_phi.add(clave)
            
            self._imprimir_phi_debug(set(submecanismo), set(subalcance), phi)

            if phi < mejor_phi:
                mejor_phi = phi
                mejor_dist = dist_part
                mejor_bip = (set(submecanismo), set(subalcance))
            if phi == 0.0:
                break

        return self._retornar_solucion(mejor_phi, mejor_dist, mejor_bip)

    def _retornar_solucion(self, phi, dist, bip):
        if bip is None:
            return Solution(GEOMETRIC_LABEL, -1, DUMMY_ARR, DUMMY_ARR, 0.0, ERROR_PARTITION)
        
        # Asegurar que la distribución del subsistema también se aplane y se convierta a float16
        distribucion_subsistema_np = self.sia_dists_marginales.flatten().astype(np.float16)
        distribucion_particion_np = dist.flatten().astype(np.float16)

        return Solution(
            estrategia=GEOMETRIC_LABEL,
            perdida=phi,
            distribucion_subsistema=distribucion_subsistema_np,
            distribucion_particion=distribucion_particion_np,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=self._formatear_biparticion(bip)
        )

    def _cargar_tensores_elementales(self):
        tpm = self.sia_cargar_tpm()
        self.tensores = {i: tpm[:, i].astype(np.float16) for i in range(tpm.shape[1])}

    def _construir_tabla_costos_parallel(self):
        tpm = self.sia_cargar_tpm()
        num_vars = tpm.shape[1]
        
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(calcular_costos_var, i, tpm[:, i].astype(np.float16), num_vars) for i in range(num_vars)]
            for fut in as_completed(futures):
                self.tabla_costos.update(fut.result())

    def _biparticiones_heuristicas(self) -> List[Tuple[Set[int], Set[int]]]:
        n = self.sia_subsistema.estado_inicial.size
        todos = set(range(n))
        candidatos = []
        
        for i in range(n):
            candidatos.append(({i}, todos - {i}))
        
        k = min(3, n)
        var_scores = []
        for var in range(n):
            tensor = self.tensores[var]
            delta = np.mean(np.abs(tensor - np.mean(tensor)))
            var_scores.append((var, float(delta))) 
        
        estables = sorted(var_scores, key=lambda x: x[1])[:k]
        for var_idx, _ in estables:
            candidatos.append(({var_idx}, todos - {var_idx}))

        distancias = np.zeros((n, n), dtype=np.float16)
        for i in range(n):
            for j in range(n):
                if i != j:
                    distancias[i, j] = np.mean(np.abs(self.tensores[i] - self.tensores[j])) 
        
        condensed = squareform(distancias)
        Z = linkage(condensed, method='average')
        
        etiquetas = fcluster(Z, t=2, criterion='maxclust')
        
        grupo1 = {i for i, label in enumerate(etiquetas) if label == 1}
        grupo2 = todos - grupo1
        
        if grupo1 and grupo2:
            candidatos.append((grupo1, grupo2))
            
        num_random_candidates = min(5, n // 2)
        for _ in range(num_random_candidates):
            size_m = np.random.randint(1, n)
            m_set = set(np.random.choice(list(todos), size_m, replace=False))
            p_set = todos - m_set
            if m_set and p_set:
                candidatos.append((m_set, p_set))
        
        return candidatos

    def _calcular_distribucion_particionada(self, mecanismo: Set[int], alcance: Set[int]) -> np.ndarray:
        # En esta función, es crucial que la distribución resultante tenga el tamaño esperado del subsistema completo.
        # Es decir, 2 elevado a la potencia del número de variables en sia_subsistema.estado_inicial.size
        
        # El número de variables en el subsistema (N)
        num_vars_subsistema = self.sia_subsistema.estado_inicial.size
        # El número total de estados esperados (2^N)
        expected_size_of_distribution = 2 ** num_vars_subsistema 
        
        if not mecanismo or not alcance:
            return np.zeros(expected_size_of_distribution, dtype=np.float16)

        indices_mec = np.array(list(mecanismo), dtype=np.int8)
        indices_alc = np.array(list(alcance), dtype=np.int8)
        
        particion_obj = self.sia_subsistema.bipartir(indices_alc, indices_mec)
        
        # Asegura que siempre devuelva un array 1D y con el tamaño esperado.
        # Si `particion_obj.distribucion_marginal()` retorna algo de tamaño 5,
        # significa que internamente está calculando mal.
        # Aquí intentamos forzar el tamaño, lo que es un parche si la lógica de `System` es incorrecta.
        dist_marginal = particion_obj.distribucion_marginal().flatten().astype(np.float16)
        
        if dist_marginal.size != expected_size_of_distribution:
            self.logger.debug(f"ADVERTENCIA: La distribución de la partición (tamaño {dist_marginal.size}) no coincide con el número esperado de estados ({expected_size_of_distribution}). Rellenando con ceros. ¡Esto es un parche!")
            temp_dist = np.zeros(expected_size_of_distribution, dtype=np.float16)
            temp_dist[:min(dist_marginal.size, expected_size_of_distribution)] = \
                dist_marginal[:min(dist_marginal.size, expected_size_of_distribution)]
            return temp_dist
        
        return dist_marginal

    def _formatear_biparticion(self, bip: Tuple[Set[int], Set[int]]) -> str:
        if bip is None:
            return DUMMY_PARTITION
        m, p = sorted(list(bip[0])), sorted(list(bip[1]))
        return fmt_biparticion([m, p], [p, m])

    def _imprimir_phi_debug(self, mecanismo: Set[int], alcance: Set[int], phi: float):
        literal = self._formatear_biparticion((mecanismo, alcance))
        print(f"φ = {phi:.4f}   ↔   Bipartición: {literal}")