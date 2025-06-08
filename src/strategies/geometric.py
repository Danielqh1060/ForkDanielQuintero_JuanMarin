import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# Módulos internos del framework/proyecto:
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
from src.funcs.format import fmt_biparte_q  # Para formato amigable de bipartición

# ==== PARTE 1: Worker para el cálculo de tabla de costos de transición ====

def _tabla_costos_worker(args):
    """
    Worker externo que permite paralelizar el cálculo de la tabla de costos t(i, j)
    para una sola variable v usando el modelo geométrico (distancia de Hamming, etc).
    Se usa con ProcessPoolExecutor para usar múltiples núcleos del CPU.
    """
    v, X_v, n, num_estados = args
    tabla = np.zeros((num_estados, num_estados))
    memo_t = {}  # Diccionario de memoización para evitar cálculos repetidos

    def calcular_t(i, j):
        """
        Calcula recursivamente el costo t(i, j) para el estado i → j,
        usando distancia de Hamming y sumando el costo de caminos óptimos
        (como sugiere la topología del hipercubo).
        """
        clave = (i, j)
        if clave in memo_t:
            return memo_t[clave]
        dH = bin(i ^ j).count("1")  # Distancia de Hamming entre i y j
        gamma = 2 ** (-dH)          # Penalización exponencial por distancia
        xi, xj = X_v[i], X_v[j]
        costo_directo = abs(xi - xj)
        if dH <= 1:
            t = gamma * costo_directo
        else:
            # Busca vecinos de i que se acercan a j y suma recursivamente
            vecinos = [i ^ (1 << b) for b in range(n) if bin((i ^ (1 << b)) ^ j).count("1") < dH]
            suma_vecinos = sum(calcular_t(k, j) for k in vecinos)
            t = gamma * (costo_directo + suma_vecinos)
        memo_t[clave] = t
        return t

    # Calcula la matriz completa t(i, j) para todos los estados de la variable v
    for i in range(num_estados):
        for j in range(num_estados):
            tabla[i, j] = calcular_t(i, j)
    return (v, tabla)

# ==== PARTE 2: Estrategia geométrica principal ====

class Geometric(SIA):
    """
    Estrategia Geométrica para bipartición óptima de sistemas.
    Usa propiedades topológicas del hipercubo y distancia de Hamming para evaluar el costo de transición.
    Permite ejecución paralela para máxima eficiencia en CPUs multinúcleo.
    """

    def __init__(self, gestor: Manager):
        super().__init__(gestor)
        # Inicializa el profiling (monitoreo de ejecución)
        profiler_manager.start_session(
            f"{NET_LABEL}{len(gestor.estado_inicial)}{gestor.pagina}"
        )
        self.logger = SafeLogger(GEOMETRIC_LABEL)
        self.tabla_costos = {}   # Almacena la tabla de costos para cada variable
        self.X = None            # Tensor elemental para cada variable

    @profile(context={TYPE_TAG: GEOMETRIC_ANALYSIS_TAG})
    def aplicar_estrategia(self, condicion, alcance, mecanismo):
        """
        Método principal llamado por el sistema para aplicar la estrategia.
        Prepara el subsistema, construye la tabla de costos y busca la mejor bipartición.
        """
        self.sia_preparar_subsistema(condicion, alcance, mecanismo)

        n = len(self.sia_subsistema.indices_ncubos)
        num_estados = 2 ** n
        if n < 2:
            # No se puede particionar si hay menos de 2 variables
            return Solution(
                estrategia=GEOMETRIC_LABEL,
                perdida=DUMMY_EMD,
                distribucion_subsistema=DUMMY_ARR,
                distribucion_particion=DUMMY_ARR,
                particion=ERROR_PARTITION,
                tiempo_total=0.0,
            )

        # Obtiene el tensor elemental X para el sistema reducido
        self.X = self._extraer_tensores_elementales(n, num_estados)
        # Construye la tabla de costos de forma paralela (multicore)
        self._construir_tabla_costos_parallel(n, num_estados)
        # Busca la mejor bipartición usando una heurística simple
        mejor_phi, mejor_bip = self._heuristica_biparticion(n)

        if mejor_bip:
            grupo1, grupo2 = mejor_bip
            # Construye representación tipo Q-Nodes para impresión
            prim = [(0, idx) for idx in grupo1]
            dual = [(0, idx) for idx in grupo2]
            mejor_particion_str = fmt_biparte_q(prim, dual)
            # Calcula las distribuciones marginales para la partición encontrada
            indices_grupo1 = np.array(grupo1, dtype=np.int8)
            indices_grupo2 = np.array(grupo2, dtype=np.int8)
            particion = self.sia_subsistema.bipartir(indices_grupo1, indices_grupo2)
            distribucion_particion = particion.distribucion_marginal()
        else:
            # Si no hay bipartición válida, devuelve valores dummy
            mejor_particion_str = DUMMY_PARTITION
            distribucion_particion = DUMMY_ARR

        # Retorna el objeto Solution, compatible con el framework del proyecto
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
        Construye X[v, i]: valor del n-cubo v para el estado binario i.
        Ignora cubos cuyas dimensiones no existen en el sistema reducido.
        """
        X = np.zeros((n, num_estados))
        for v, ncubo in enumerate(self.sia_subsistema.ncubos):
            dims_cubo = ncubo.dims
            # Si alguna dimensión está fuera de rango, ignora ese cubo
            if np.any(dims_cubo < 0) or np.any(dims_cubo >= n):
                print(f"Skipping cubo {v} (dims fuera de rango): {dims_cubo} para n={n}")
                continue
            for i in range(num_estados):
                # bits_global representa el estado binario global para n variables
                bits_global = np.array([(i >> k) & 1 for k in reversed(range(n))])
                # bits_cubo selecciona solo las dimensiones activas del cubo actual
                bits_cubo = tuple(bits_global[dims_cubo])
                X[v, i] = ncubo.data[bits_cubo]
        return X

    def _construir_tabla_costos_parallel(self, n, num_estados):
        """
        Construye la tabla de costos t(i, j) para cada variable en paralelo.
        Utiliza todos los núcleos del procesador disponibles para máxima eficiencia.
        """
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            args_list = [(v, self.X[v].copy(), n, num_estados) for v in range(n)]
            futures = [executor.submit(_tabla_costos_worker, args) for args in args_list]
            for fut in as_completed(futures):
                v, tabla = fut.result()
                self.tabla_costos[v] = tabla

    def _heuristica_biparticion(self, n):
        """
        Heurística simple: para cada variable, la separa del resto y calcula el costo.
        Retorna la partición que minimiza el costo phi.
        Puedes reemplazar por metaheurística si quieres más calidad.
        """
        mejor_phi = float("inf")
        mejor_bip = None
        for v in range(n):
            grupo1 = [v]
            grupo2 = [i for i in range(n) if i != v]
            phi = self._evaluar_biparticion(grupo1, grupo2)
            if phi < mejor_phi:
                mejor_phi = phi
                mejor_bip = (grupo1, grupo2)
        return mejor_phi, mejor_bip

    def _evaluar_biparticion(self, grupo1, grupo2):
        """
        Calcula el costo de una bipartición: suma los promedios de los costos
        entre variables de ambos grupos.
        """
        if not grupo1 or not grupo2:
            return float("inf")
        total = 0
        for v1 in grupo1:
            for v2 in grupo2:
                total += np.mean(self.tabla_costos[v1]) + np.mean(self.tabla_costos[v2])
        return total / (len(grupo1) * len(grupo2))
