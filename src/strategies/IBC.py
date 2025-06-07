import time
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from src.middlewares.slogger import SafeLogger
from src.funcs.base import emd_efecto, ABECEDARY
from src.middlewares.profile import profiler_manager, profile
from src.funcs.format import fmt_biparte_q
from src.controllers.manager import Manager
from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.models.core.system import System
from src.constants.models import (
    IBC_ANALYSIS_TAG,
    IBC_LABEL,
    IBC_STRATEGY_TAG,
)
from src.constants.base import (
    TYPE_TAG,
    NET_LABEL,
    EFECTO,
    ACTUAL,
)

def distribucion_marginal_ibc(TPM_optima, estado_inicial, indices_mecanismo, indices_alcance):
    sistema_particion = System(TPM_optima, estado_inicial)
    distr_marginal = sistema_particion.distribucion_marginal(indices_mecanismo, indices_alcance)
    return distr_marginal

def calcular_MIC(TPM, vertices):
    n = len(vertices)
    MIC = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                MIC[i, j] = np.corrcoef(TPM[:, i], TPM[:, j])[0, 1]
    MIC = np.nan_to_num(MIC)
    return MIC

def agrupamiento_jerarquico(MIC, num_grupos=2):
    disimilitud = 1 - MIC
    disimilitud = (disimilitud + disimilitud.T) / 2
    np.fill_diagonal(disimilitud, 0)
    Z = linkage(squareform(disimilitud), method='ward')
    grupos = fcluster(Z, t=num_grupos, criterion='maxclust')
    return grupos

def evaluar_biparticion(TPM, estado_inicial, grupo, vertices):
    # Identificar los índices de mecanismo y alcance
    indices_mecanismo = [idx for (t, idx) in grupo if t == ACTUAL]
    indices_alcance = [idx for (t, idx) in grupo if t == EFECTO]
    if not indices_mecanismo or not indices_alcance:
        return float('inf'), np.zeros(1)
    dist = distribucion_marginal_ibc(
        TPM, estado_inicial, indices_mecanismo, indices_alcance
    )
    # EMD real, puedes agregar divergencia causal aquí
    metrica = np.linalg.norm(dist)  # ¡Reemplaza por emd_efecto si está disponible!
    return metrica, dist

def temple_simulado(mejor_particion, TPM, estado_inicial, vertices, memoria_particiones, iteraciones=100):
    # Esta es una plantilla básica, puedes expandirla con "vecinos", temperatura, etc.
    # Por ahora solo guarda la mejor partición encontrada
    memoria_particiones[mejor_particion] = evaluar_biparticion(TPM, estado_inicial, mejor_particion, vertices)
    return mejor_particion

class IBC(SIA):
    def __init__(self, gestor: Manager):
        super().__init__(gestor)
        profiler_manager.start_session(
            f"{NET_LABEL}{len(gestor.estado_inicial)}{gestor.pagina}"
        )
        self.logger = SafeLogger(IBC_STRATEGY_TAG)
        self.vertices: set[tuple]
        self.etiquetas = [tuple(s.lower() for s in ABECEDARY), ABECEDARY]
        self.memoria_particiones = dict()

    @profile(context={TYPE_TAG: IBC_ANALYSIS_TAG})
    def aplicar_estrategia(self, condicion: str, alcance: str, mecanismo: str):
        self.sia_preparar_subsistema(condicion, alcance, mecanismo)

        presente = tuple((ACTUAL, idx_actual) for idx_actual in self.sia_subsistema.dims_ncubos)
        futuro = tuple((EFECTO, idx_efecto) for idx_efecto in self.sia_subsistema.indices_ncubos)
        vertices = list(presente + futuro)
        self.vertices = set(presente + futuro)
        TPM = np.column_stack([cube.data.flatten() for cube in self.sia_subsistema.ncubos])
        assert len(vertices) == TPM.shape[1], f"Desajuste: vertices={len(vertices)}, TPM columnas={TPM.shape[1]}"

        MIC = calcular_MIC(TPM, vertices)
        grupos = agrupamiento_jerarquico(MIC, num_grupos=2)
        grupo1 = tuple(vertices[i] for i in range(len(grupos)) if grupos[i] == 1)
        grupo2 = tuple(vertices[i] for i in range(len(grupos)) if grupos[i] == 2)
        
        # Control de errores
        if not grupo1 or not grupo2:
            self.logger.error("Agrupamiento retornó grupo vacío.")
            raise Exception("Agrupamiento retornó grupo vacío.")

        mejor_particion = grupo1
        mejor_metrica, mejor_dist = evaluar_biparticion(TPM, self.sia_subsistema.estado_inicial, grupo1, vertices)
        metrica2, dist2 = evaluar_biparticion(TPM, self.sia_subsistema.estado_inicial, grupo2, vertices)
        if metrica2 < mejor_metrica:
            mejor_particion = grupo2
            mejor_metrica = metrica2
            mejor_dist = dist2

        mejor_particion = temple_simulado(mejor_particion, TPM, self.sia_subsistema.estado_inicial, vertices, self.memoria_particiones)
        perdida_mip, dist_marginal_mip = self.memoria_particiones[mejor_particion]
        fmt_mip = fmt_biparte_q(list(mejor_particion), self.nodes_complement(mejor_particion))
        return Solution(
            estrategia=IBC_LABEL,
            perdida=perdida_mip,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=dist_marginal_mip,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_mip,
        )

    def nodes_complement(self, nodes: list[tuple[int, int]]):
        return list(set(self.vertices) - set(nodes))
