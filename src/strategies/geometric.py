import time
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor
from src.middlewares.slogger import SafeLogger
from src.controllers.manager import Manager
from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.constants.models import (
    GEOMETRIC_ANALYSIS_TAG,
    GEOMETRIC_LABEL,
    DUMMY_ARR,
    DUMMY_EMD,
    DUMMY_PARTITION,
    ERROR_PARTITION,
)
from src.constants.base import (
    TYPE_TAG,
    NET_LABEL,
)
from src.middlewares.profile import profiler_manager, profile
from src.funcs.format import fmt_biparte_q

class Geometric(SIA):
    """
    Estrategia Geométrica optimizada:
    - Precalcula tabla de costos en paralelo usando threads.
    - Greedy extendido con pocos seeds.
    - Evalúa biparticiones muestreando pares de estados para máxima velocidad.
    """

    def __init__(self, gestor: Manager):
        super().__init__(gestor)
        profiler_manager.start_session(
            f"{NET_LABEL}{len(gestor.estado_inicial)}{gestor.pagina}"
        )
        self.logger = SafeLogger(GEOMETRIC_LABEL)
        self.memo_t = {}
        self.X = None
        self.tabla_costos = {}

    @profile(context={TYPE_TAG: GEOMETRIC_ANALYSIS_TAG})
    def aplicar_estrategia(self, condicion, alcance, mecanismo):
        self.sia_preparar_subsistema(condicion, alcance, mecanismo)
        n = len(self.sia_subsistema.indices_ncubos)
        num_estados = 2 ** n

        if n < 2:
            return Solution(
                estrategia=GEOMETRIC_LABEL,
                perdida=DUMMY_EMD,
                distribucion_subsistema=DUMMY_ARR,
                distribucion_particion=DUMMY_ARR,
                particion=ERROR_PARTITION,
                tiempo_total=0.0,
            )

        self.X = self._extraer_tensores_elementales(n, num_estados)
        self._construir_tabla_costos_parallel(n, num_estados)

        mejor_phi, mejor_bip = self._heuristica_greedy_extendida_random_seeds(n, seeds=3, num_samples=500)
        mejor_phi, mejor_bip = self._hill_climbing(mejor_bip, mejor_phi, n, num_samples=500)

        if mejor_bip:
            grupo1, grupo2 = mejor_bip
            prim = [(0, idx) for idx in grupo1]
            dual = [(0, idx) for idx in grupo2]
            mejor_particion_str = fmt_biparte_q(prim, dual)
            indices_grupo1 = np.array(grupo1, dtype=np.int8)
            indices_grupo2 = np.array(grupo2, dtype=np.int8)
            particion = self.sia_subsistema.bipartir(indices_grupo1, indices_grupo2)
            distribucion_particion = particion.distribucion_marginal()
        else:
            mejor_particion_str = DUMMY_PARTITION
            distribucion_particion = DUMMY_ARR

        return Solution(
            estrategia=GEOMETRIC_LABEL,
            perdida=mejor_phi,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=distribucion_particion,
            particion=mejor_particion_str,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
        )

    def _extraer_tensores_elementales(self, n, num_estados):
        """
        Calcula los tensores elementales X[v][i] para cada variable v y estado i.
        """
        X = np.zeros((n, num_estados))
        for v, ncubo in enumerate(self.sia_subsistema.ncubos):
            for i in range(num_estados):
                bits = [(i >> k) & 1 for k in reversed(range(n))]
                X[v, i] = ncubo.data[tuple(bits)]
        return X

    def _construir_tabla_costos_parallel(self, n, num_estados):
        """
        Precalcula la tabla de costos T[v][i][j] en paralelo usando ThreadPoolExecutor.
        """
        def fill_tabla(v):
            tabla = np.zeros((num_estados, num_estados))
            for i in range(num_estados):
                for j in range(num_estados):
                    tabla[i, j] = self._calcular_t(i, j, v, n)
            return v, tabla

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(fill_tabla, range(n)))
        for v, tabla in results:
            self.tabla_costos[v] = tabla

    def _calcular_t(self, i, j, v, n):
        clave = (v, i, j)
        if clave in self.memo_t:
            return self.memo_t[clave]
        dH = bin(i ^ j).count("1")
        gamma = 2 ** (-dH)
        xi, xj = self.X[v, i], self.X[v, j]
        costo_directo = abs(xi - xj)
        if dH <= 1:
            t = gamma * costo_directo
        else:
            vecinos = self._vecinos_optimos(i, j, n)
            suma_vecinos = sum(self._calcular_t(k, j, v, n) for k in vecinos)
            t = gamma * (costo_directo + suma_vecinos)
        self.memo_t[clave] = t
        return t

    def _vecinos_optimos(self, i, j, n):
        vecinos = []
        dH_ij = bin(i ^ j).count("1")
        for b in range(n):
            k = i ^ (1 << b)
            if bin(k ^ j).count("1") < dH_ij:
                vecinos.append(k)
        return vecinos

    def _heuristica_greedy_extendida_random_seeds(self, n, seeds=3, num_samples=500):
        """
        Greedy extendido: solo 'seeds' inicializaciones y evaluaciones usando muestreo aleatorio.
        """
        if n < 2:
            return float("inf"), ([], [])

        mejor_phi = float('inf')
        mejor_bip = None
        indices = list(range(n))
        seed_pairs = [(i, j) for i in indices for j in indices if i != j]
        random.shuffle(seed_pairs)
        seed_pairs = seed_pairs[:seeds]  # Solo unas pocas inicializaciones

        for i, j in seed_pairs:
            grupo1 = [i]
            grupo2 = [j]
            restantes = [k for k in indices if k not in (i, j)]
            for k in restantes:
                phi1 = self._evaluar_biparticion(grupo1 + [k], grupo2, num_samples)
                phi2 = self._evaluar_biparticion(grupo1, grupo2 + [k], num_samples)
                if phi1 < phi2:
                    grupo1.append(k)
                else:
                    grupo2.append(k)
            phi = self._evaluar_biparticion(grupo1, grupo2, num_samples)
            if phi < mejor_phi:
                mejor_phi = phi
                mejor_bip = (grupo1[:], grupo2[:])
        return mejor_phi, mejor_bip

    def _hill_climbing(self, biparticion, phi_actual, n, max_iter=10, num_samples=500):
        """
        Refinamiento post-greedy: intenta mover variables entre grupos para mejorar φ.
        """
        if not biparticion:
            return phi_actual, biparticion
        grupo1, grupo2 = [list(g) for g in biparticion]
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            # Prueba mover de grupo1 a grupo2
            for v in grupo1[:]:
                nuevo_g1 = [x for x in grupo1 if x != v]
                nuevo_g2 = grupo2 + [v]
                phi = self._evaluar_biparticion(nuevo_g1, nuevo_g2, num_samples)
                if phi < phi_actual:
                    grupo1, grupo2 = nuevo_g1, nuevo_g2
                    phi_actual = phi
                    improved = True
            # Prueba mover de grupo2 a grupo1
            for v in grupo2[:]:
                nuevo_g2 = [x for x in grupo2 if x != v]
                nuevo_g1 = grupo1 + [v]
                phi = self._evaluar_biparticion(nuevo_g1, nuevo_g2, num_samples)
                if phi < phi_actual:
                    grupo1, grupo2 = nuevo_g1, nuevo_g2
                    phi_actual = phi
                    improved = True
        return phi_actual, (grupo1, grupo2)

    def _evaluar_biparticion(self, grupo1, grupo2, num_samples=500):
        """
        Evalúa la bipartición usando muestreo aleatorio de pares (i, j) para máxima velocidad.
        """
        if not grupo1 or not grupo2:
            return float("inf")
        n = len(self.sia_subsistema.indices_ncubos)
        num_estados = 2 ** n
        max_pairs = num_estados ** 2
        # Evita muestrear más de todos los pares posibles
        num_samples = min(num_samples, max_pairs)
        if num_samples < max_pairs:
            sample_indices = random.sample(range(max_pairs), num_samples)
        else:
            sample_indices = range(max_pairs)
        total = 0
        for v1 in grupo1:
            for v2 in grupo2:
                v1_costs, v2_costs = [], []
                for idx in sample_indices:
                    i, j = divmod(idx, num_estados)
                    v1_costs.append(self.tabla_costos[v1][i, j])
                    v2_costs.append(self.tabla_costos[v2][i, j])
                total += np.mean(v1_costs) + np.mean(v2_costs)
        return total / (len(grupo1) * len(grupo2))
