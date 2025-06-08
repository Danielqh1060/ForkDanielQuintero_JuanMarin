import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
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
from src.funcs.format import fmt_biparte_q  # <--- Importa el formateador

def _tabla_costos_worker(args):
    """Worker externo para cálculo paralelizado de la tabla de costos de una variable."""
    v, X_v, n, num_estados = args
    tabla = np.zeros((num_estados, num_estados))
    memo_t = {}
    def calcular_t(i, j):
        clave = (i, j)
        if clave in memo_t:
            return memo_t[clave]
        dH = bin(i ^ j).count("1")
        gamma = 2 ** (-dH)
        xi, xj = X_v[i], X_v[j]
        costo_directo = abs(xi - xj)
        if dH <= 1:
            t = gamma * costo_directo
        else:
            vecinos = [i ^ (1 << b) for b in range(n) if bin((i ^ (1 << b)) ^ j).count("1") < dH]
            suma_vecinos = sum(calcular_t(k, j) for k in vecinos)
            t = gamma * (costo_directo + suma_vecinos)
        memo_t[clave] = t
        return t
    for i in range(num_estados):
        for j in range(num_estados):
            tabla[i, j] = calcular_t(i, j)
    return (v, tabla)

class Geometric(SIA):
    """
    Estrategia Geométrica para bipartición óptima mediante propiedades topológicas.
    Calcula la tabla de costos de transición entre estados usando la función recursiva geométrica.
    ¡Ahora paralelizada en multicore!
    """

    def __init__(self, gestor: Manager):
        super().__init__(gestor)
        profiler_manager.start_session(
            f"{NET_LABEL}{len(gestor.estado_inicial)}{gestor.pagina}"
        )
        self.logger = SafeLogger(GEOMETRIC_LABEL)
        self.tabla_costos = {}
        self.X = None

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
        self._construir_tabla_costos_parallel(n, num_estados)  # <-- paralelizada
        mejor_phi, mejor_bip = self._heuristica_biparticion(n)

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
        X = np.zeros((n, num_estados))
        for v, ncubo in enumerate(self.sia_subsistema.ncubos):
            for i in range(num_estados):
                bits = [(i >> k) & 1 for k in reversed(range(n))]
                X[v, i] = ncubo.data[tuple(bits)]
        return X

    def _construir_tabla_costos_parallel(self, n, num_estados):
        """Crea la tabla de costos usando todos los núcleos."""
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            args_list = [(v, self.X[v].copy(), n, num_estados) for v in range(n)]
            futures = [executor.submit(_tabla_costos_worker, args) for args in args_list]
            for fut in as_completed(futures):
                v, tabla = fut.result()
                self.tabla_costos[v] = tabla

    def _heuristica_biparticion(self, n):
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
        if not grupo1 or not grupo2:
            return float("inf")
        total = 0
        for v1 in grupo1:
            for v2 in grupo2:
                total += np.mean(self.tabla_costos[v1]) + np.mean(self.tabla_costos[v2])
        return total / (len(grupo1) * len(grupo2))
