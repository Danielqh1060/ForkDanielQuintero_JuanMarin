import time
import numpy as np
import multiprocessing as mp
import os
from itertools import combinations

# --- Imports del framework ---
from src.middlewares.slogger import SafeLogger
from src.controllers.manager import Manager
from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.models.enums.distance import MetricDistance
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
from src.funcs.base import emd_efecto

# ========================
# Worker EMD (eficiente)
# ========================
def _evaluar_biparticion_worker(args):
    grupo1, grupo2, subsistema, dists_base = args
    indices_grupo1 = np.array(grupo1, dtype=np.int8)
    indices_grupo2 = np.array(grupo2, dtype=np.int8)
    try:
        particion = subsistema.bipartir(indices_grupo1, indices_grupo2)
        distribucion = particion.distribucion_marginal()
        phi = emd_efecto(distribucion, dists_base)
    except Exception:
        phi = float("inf")
        distribucion = DUMMY_ARR
    return grupo1, grupo2, phi, distribucion

# ============================
# Recursivo: Tabla T de costos
# ============================
def _calcular_tabla_costos_recursiva(ncubo, n, num_estados):
    """
    Calcula recursivamente la tabla t(i, j) para un solo cubo (teórico, n pequeño)
    """
    tabla = np.zeros((num_estados, num_estados))
    memo_t = {}
    def calcular_t(i, j):
        clave = (i, j)
        if clave in memo_t:
            return memo_t[clave]
        dH = bin(i ^ j).count("1")
        gamma = 2 ** (-dH)
        # bits_global → bits para cada cubo
        bits_i = np.array([(i >> k) & 1 for k in reversed(range(n))])
        bits_j = np.array([(j >> k) & 1 for k in reversed(range(n))])
        # Selecciona solo las dimensiones activas
        dims_cubo = ncubo.dims
        if np.any(dims_cubo < 0) or np.any(dims_cubo >= n):
            return 0
        xi = ncubo.data[tuple(bits_i[dims_cubo])]
        xj = ncubo.data[tuple(bits_j[dims_cubo])]
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
    return tabla

# ========================
# Clase principal
# ========================
class Geometric(SIA):
    """
    Estrategia Geométrica con:
    - Evaluación eficiente por EMD paralela para sistemas grandes.
    - Opción de imprimir tabla T de costos recursiva (casos pequeños).
    """

    def __init__(self, gestor: Manager):
        super().__init__(gestor)
        profiler_manager.start_session(
            f"{NET_LABEL}{len(gestor.estado_inicial)}{gestor.pagina}"
        )
        self.logger = SafeLogger(GEOMETRIC_LABEL)

    @profile(context={TYPE_TAG: GEOMETRIC_ANALYSIS_TAG})
    def aplicar_estrategia(
        self, condicion, alcance, mecanismo,
        max_grupo=None, mostrar_tabla_costos=False
    ):
        self.sia_preparar_subsistema(condicion, alcance, mecanismo)
        n = len(self.sia_subsistema.indices_ncubos)
        if n < 2:
            return Solution(
                estrategia=GEOMETRIC_LABEL,
                perdida=DUMMY_EMD,
                distribucion_subsistema=DUMMY_ARR,
                distribucion_particion=DUMMY_ARR,
                particion=ERROR_PARTITION,
                tiempo_total=0.0,
            )

        # --- Opción 1: Imprimir la tabla T de costos (solo n pequeño por eficiencia) ---
        if mostrar_tabla_costos and n <= 6:
            num_estados = 2 ** n
            print("\n===== TABLA T de COSTOS (t(i, j)) PARA CADA VARIABLE =====")
            for v, ncubo in enumerate(self.sia_subsistema.ncubos):
                tabla = _calcular_tabla_costos_recursiva(ncubo, n, num_estados)
                print(f"\nVariable {v} (dims={ncubo.dims}):")
                print(np.round(tabla, 4))
            print("==========================================================")

        # --- Opción 2: Estrategia eficiente para sistemas grandes ---
        if max_grupo is None:
            max_grupo = n // 2

        biparticiones = []
        for k in range(1, max_grupo + 1):
            for grupo1 in combinations(range(n), k):
                grupo2 = [i for i in range(n) if i not in grupo1]
                biparticiones.append((list(grupo1), grupo2))
        dists_base = self.sia_dists_marginales.copy()
        args_list = [
            (grupo1, grupo2, self.sia_subsistema, dists_base)
            for grupo1, grupo2 in biparticiones
        ]
        mejor_phi = float("inf")
        mejor_bip = None
        mejor_dist = DUMMY_ARR
        t0 = time.time()
        with mp.get_context("spawn").Pool(processes=os.cpu_count()) as pool:
            for grupo1, grupo2, phi, distribucion in pool.imap_unordered(
                _evaluar_biparticion_worker, args_list, chunksize=2
            ):
                if phi < mejor_phi:
                    mejor_phi = phi
                    mejor_bip = (grupo1, grupo2)
                    mejor_dist = distribucion
        # --- Formato final de la partición (mayúscula/minúscula, QNodes) ---
        if mejor_bip:
            grupo1, grupo2 = mejor_bip
            prim = [(1, idx) for idx in grupo1]  # Futuros: mayúscula
            dual = [(0, idx) for idx in grupo2]  # Presentes: minúscula
            mejor_particion_str = fmt_biparte_q(prim, dual)
        else:
            mejor_particion_str = DUMMY_PARTITION
        # --- Salida final ---
        return Solution(
            estrategia=GEOMETRIC_LABEL,
            perdida=mejor_phi,
            distribucion_subsistema=dists_base,
            distribucion_particion=mejor_dist,
            particion=mejor_particion_str,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
        )
