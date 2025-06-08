import time
import numpy as np
import multiprocessing as mp
import os
from itertools import combinations
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

# ----------- Worker con semilla fija -----------
def _evaluar_biparticion_worker(args):
    grupo1, grupo2, X, n, muestras, semilla = args
    num_estados = 2 ** n
    total = 0
    rng = np.random.default_rng(seed=semilla)  # Semilla fija para determinismo
    idx_pairs = rng.integers(0, num_estados, size=(muestras, 2))
    for v1 in grupo1:
        for v2 in grupo2:
            suma = 0
            for i, j in idx_pairs:
                if i != j:
                    dH = bin(i ^ j).count("1")
                    gamma = 2 ** (-dH)
                    xi1, xj1 = X[v1, i], X[v1, j]
                    xi2, xj2 = X[v2, i], X[v2, j]
                    t1 = gamma * abs(xi1 - xj1)
                    t2 = gamma * abs(xi2 - xj2)
                    suma += t1 + t2
            suma /= muestras
            total += suma
    return (grupo1, grupo2, total / (len(grupo1) * len(grupo2)) if grupo1 and grupo2 else float("inf"))

# ----------------- Estrategia principal Geometric ------------------
class Geometric(SIA):
    def __init__(self, gestor: Manager):
        super().__init__(gestor)
        profiler_manager.start_session(
            f"{NET_LABEL}{len(gestor.estado_inicial)}{gestor.pagina}"
        )
        self.logger = SafeLogger(GEOMETRIC_LABEL)
        self.X = None

    @profile(context={TYPE_TAG: GEOMETRIC_ANALYSIS_TAG})
    def aplicar_estrategia(self, condicion, alcance, mecanismo, muestras=4096, max_grupo=3, semilla_global=42):
        """
        - `muestras`: número de pares (i, j) para estimar t(i,j).
        - `max_grupo`: tamaño máximo del grupo 1 de la bipartición.
        - `semilla_global`: fija aleatoriedad para reproducibilidad.
        """
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

        # Genera todas las combinaciones de bipartición válidas
        biparticiones = []
        for k in range(1, min(max_grupo, n // 2) + 1):
            for grupo1 in combinations(range(n), k):
                grupo2 = [i for i in range(n) if i not in grupo1]
                biparticiones.append((list(grupo1), grupo2))

        # Llama a los workers en paralelo, cada uno con semilla única reproducible
        mejor_phi = float("inf")
        mejor_bip = None
        args_list = [
            (grupo1, grupo2, self.X, n, muestras, semilla_global + idx)
            for idx, (grupo1, grupo2) in enumerate(biparticiones)
        ]
        with mp.get_context("spawn").Pool(processes=os.cpu_count()) as pool:
            for grupo1, grupo2, phi in pool.imap_unordered(_evaluar_biparticion_worker, args_list, chunksize=1):
                if phi < mejor_phi:
                    mejor_phi = phi
                    mejor_bip = (grupo1, grupo2)

        # Arma el resultado
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
        X = np.zeros((n, num_estados), dtype=np.float32)
        for v, ncubo in enumerate(self.sia_subsistema.ncubos):
            dims_cubo = ncubo.dims
            if np.any(dims_cubo < 0) or np.any(dims_cubo >= n):
                continue
            for i in range(num_estados):
                bits_global = np.array([(i >> k) & 1 for k in reversed(range(n))])
                bits_cubo = tuple(bits_global[dims_cubo])
                X[v, i] = ncubo.data[bits_cubo]
        return X
