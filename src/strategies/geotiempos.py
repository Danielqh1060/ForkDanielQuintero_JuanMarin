import time
import numpy as np
from typing import Dict, Tuple, Set, List
from itertools import combinations
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from src.models.base.sia import SIA
from src.controllers.manager import Manager
from src.models.core.solution import Solution

from src.middlewares.slogger import SafeLogger
from src.middlewares.profile import profile, profiler_manager

from src.funcs.base import emd_efecto
from src.funcs.format import fmt_biparticion

from src.constants.base import FLOAT_ZERO, TYPE_TAG, NET_LABEL
from src.constants.models import (
    GEOMETRIC_LABEL,
    GEOMETRIC_ANALYSIS_TAG,
    DUMMY_ARR,
    DUMMY_PARTITION,
    ERROR_PARTITION,
)


class GeometricTiempo(SIA):
    def __init__(self, gestor: Manager) -> None:
        super().__init__(gestor)
        profiler_manager.start_session(
            f"{NET_LABEL}{len(gestor.estado_inicial)}{gestor.pagina}"
        )
        self.logger = SafeLogger(GEOMETRIC_LABEL)
        self.cache_transiciones: Dict[Tuple[int, int, int], float] = {}
        self.tensores: Dict[int, np.ndarray] = {}

    @profile(context={TYPE_TAG: GEOMETRIC_ANALYSIS_TAG})
    def aplicar_estrategia(self, condiciones: str, alcance: str, mecanismo: str) -> Solution:
        self.sia_tiempo_inicio = time.time()
        self.sia_preparar_subsistema(condiciones, alcance, mecanismo)

        if self.sia_subsistema.estado_inicial.size < 2:
            return Solution(GEOMETRIC_LABEL, -1, DUMMY_ARR, DUMMY_ARR, 0.0, ERROR_PARTITION)

        self._cargar_tensores_elementales()

        candidatos = self._biparticiones_heuristicas()
        mejor_phi = float("inf")
        mejor_dist = DUMMY_ARR
        mejor_bip = None

        for m, p in candidatos:
            if not m or not p:
                continue

            dist_part = self._calcular_distribucion_particionada(m, p)
            phi = emd_efecto(dist_part, self.sia_dists_marginales)

            if phi < mejor_phi:
                mejor_phi = phi
                mejor_dist = dist_part
                mejor_bip = (m, p)

            if phi == 0.0:
                break

        if mejor_bip is None:
            return Solution(GEOMETRIC_LABEL, -1, DUMMY_ARR, DUMMY_ARR, 0.0, ERROR_PARTITION)

        return Solution(
            estrategia=GEOMETRIC_LABEL,
            perdida=mejor_phi,
            distribucion_subsistema=self.sia_dists_marginales.flatten(),
            distribucion_particion=mejor_dist.flatten(),
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=self._formatear_biparticion(mejor_bip),
        )

    def _cargar_tensores_elementales(self):
        tpm = self.sia_cargar_tpm()
        for i in range(tpm.shape[1]):
            self.tensores[i] = tpm[:, i]

    def _t(self, i: int, j: int, var_idx: int) -> float:
        clave = (var_idx, i, j)
        if clave in self.cache_transiciones:
            return self.cache_transiciones[clave]

        tensor = self.tensores[var_idx]
        dH = bin(i ^ j).count("1")
        if dH == 0:
            return FLOAT_ZERO
        gamma = 2 ** -dH
        vecinos = self._vecinos_optimos(i, j)
        suma_vecinos = sum(self._t(k, j, var_idx) for k in vecinos)
        valor = gamma * (abs(tensor[i] - tensor[j]) + suma_vecinos)
        self.cache_transiciones[clave] = valor
        return valor

    def _vecinos_optimos(self, i: int, j: int) -> List[int]:
        n = self.sia_subsistema.estado_inicial.size
        return [i ^ (1 << b) for b in range(n) if bin((i ^ (1 << b)) ^ j).count("1") == bin(i ^ j).count("1") - 1]

    def _biparticiones_heuristicas(self) -> List[Tuple[Set[int], Set[int]]]:
        n = self.sia_subsistema.estado_inicial.size
        todos = set(range(n))
        candidatos: List[Tuple[Set[int], Set[int]]] = []

        # Cortes unitarios
        for i in range(n):
            candidatos.append(({i}, todos - {i}))

        # Top-k variables más estables
        k = min(3, n)
        var_scores = []
        for var in range(n):
            tensor = self.tensores[var]
            delta = np.mean(np.abs(tensor - np.mean(tensor)))
            var_scores.append((var, delta))
        estables = sorted(var_scores, key=lambda x: x[1])[:k]
        for var_idx, _ in estables:
            candidatos.append(({var_idx}, todos - {var_idx}))

        # Clustering jerárquico
        distancias = np.zeros((n, n))
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

        self.logger.debug(f"{len(candidatos)} biparticiones heurísticas generadas.")
        return candidatos

    def _calcular_distribucion_particionada(
        self, mecanismo: Set[int], alcance: Set[int]
    ) -> np.ndarray:
        if not mecanismo or not alcance:
            return np.zeros_like(self.sia_dists_marginales)
        indices_mec = np.array(list(mecanismo), dtype=np.int8)
        indices_alc = np.array(list(alcance), dtype=np.int8)
        particion = self.sia_subsistema.bipartir(indices_alc, indices_mec)
        return particion.distribucion_marginal()

    def _formatear_biparticion(self, bip: Tuple[Set[int], Set[int]]) -> str:
        if bip is None:
            return DUMMY_PARTITION
        m, p = sorted(bip[0]), sorted(bip[1])
        return fmt_biparticion([m, p], [p, m])

