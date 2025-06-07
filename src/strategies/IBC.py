import time
import numpy as np
import math
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from random import random
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

def distribucion_marginal_ibc(TPM_optima, estado_inicial):
    """
    Calcula la distribución marginal de la partición óptima generada por IBC.

    Args:
        TPM_optima (np.ndarray): TPM reconstruida por IBC.
        estado_inicial (np.ndarray): Vector binario del estado inicial del sistema.
        indices_mecanismo (np.ndarray): Índices de mecanismo (presente, t).
        indices_alcance (np.ndarray): Índices de alcance (futuro, t+1).

    Returns:
        np.ndarray: Vector 1D de distribución marginal de la partición.
    """
    sistema_particion = System(TPM_optima, estado_inicial)
    distr_marginal = sistema_particion.distribucion_marginal()
    return distr_marginal

# Paso 1: Calcular MIC
def calcular_MIC(TPM, vertices):
    n = len(vertices)
    MIC = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                MIC[i, j] = np.corrcoef(TPM[:, i], TPM[:, j])[0, 1]
    MIC = np.nan_to_num(MIC)
    return MIC

# Paso 2: Agrupamiento jerárquico adaptativo
def agrupamiento_jerarquico(MIC, num_grupos=2):
    disimilitud = 1 - MIC
    disimilitud = (disimilitud + disimilitud.T) / 2
    np.fill_diagonal(disimilitud, 0)
    Z = linkage(squareform(disimilitud), method='ward')
    grupos = fcluster(Z, t=num_grupos, criterion='maxclust')
    return grupos

# Paso 3: Evaluación con métrica híbrida
def evaluar_biparticion(TPM_optima, estado_inicial, grupo, vertices, indices_presente, indices_futuro):
    """
    Calcula la métrica y la distribución marginal para un grupo (bipartición).
    `grupo` debe ser una lista/tupla de índices de nodos en la partición.
    """
    # Determina qué nodos de 'vertices' corresponden a presente y futuro
    indices_mecanismo = [idx for (t, idx) in grupo if t == ACTUAL]
    indices_alcance   = [idx for (t, idx) in grupo if t == EFECTO]

    if not indices_mecanismo or not indices_alcance:
        # No es válida la bipartición
        return float('inf'), np.zeros(1)

    # Calcula la distribución marginal real usando la función del usuario
    dist = distribucion_marginal_ibc(
        TPM_optima, estado_inicial,
        indices_mecanismo=indices_mecanismo,
        indices_alcance=indices_alcance,
    )
    # Calcula la métrica (puedes poner aquí EMD real si lo deseas)
    metrica = np.linalg.norm(dist)
    return metrica, dist

# Paso 4: Optimización con temple simulado
def optimizar_biparticion(TPM, vertices, memoria_particiones, estado_inicial, iteraciones=100):
    MIC = calcular_MIC(TPM, vertices)
    grupos = agrupamiento_jerarquico(MIC, num_grupos=2)
    grupo1 = tuple(vertices[i] for i in range(len(grupos)) if grupos[i] == 1)
    grupo2 = tuple(vertices[i] for i in range(len(grupos)) if grupos[i] == 2)
    
    # Proteger: evita guardar particiones vacías
    if not grupo1 or not grupo2:
        print("¡Advertencia! Agrupamiento retornó grupo vacío. Revise la lógica de clustering.")
        return None

    mejor_particion = grupo1
    mejor_metrica, mejor_dist = evaluar_biparticion(
        TPM, estado_inicial, grupo1, vertices,
        indices_presente=None,
        indices_futuro=None,
    )

    metrica2, dist2 = evaluar_biparticion(
        TPM, estado_inicial, grupo2, vertices,
        indices_presente=None,
        indices_futuro=None,
    )
    if metrica2 < mejor_metrica:
        mejor_particion = grupo2
        mejor_metrica = metrica2
        mejor_dist = dist2
        
    print("Longitudes de grupo1 y grupo2:", len(grupo1), len(grupo2))
    print("Grupo1:", grupo1)
    print("Grupo2:", grupo2)
    print("TPM shape:", TPM.shape)

    memoria_particiones[mejor_particion] = (mejor_metrica, mejor_dist)

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
        self.indices_alcance: np.ndarray
        self.indices_mecanismo: np.ndarray
        self.tiempos: tuple[np.ndarray, np.ndarray]
        self.memoria_particiones = dict()

    @profile(context={TYPE_TAG: IBC_ANALYSIS_TAG})
    def aplicar_estrategia(self, condicion: str, alcance: str, mecanismo: str):
        self.sia_preparar_subsistema(condicion, alcance, mecanismo)

        presente = tuple((ACTUAL, idx_actual) for idx_actual in self.sia_subsistema.dims_ncubos)
        futuro = tuple((EFECTO, idx_efecto) for idx_efecto in self.sia_subsistema.indices_ncubos)
        vertices = list(presente + futuro)
        self.vertices = set(presente + futuro)
        TPM = np.column_stack([cube.data.flatten() for cube in self.sia_subsistema.ncubos])

        # Ejecuta el optimizador, que debe retornar la mejor partición
        mejor_particion = optimizar_biparticion(
        TPM,
        vertices,
        self.memoria_particiones,
        self.sia_subsistema.estado_inicial,
        iteraciones=100
        )

        # Saca la métrica y dist marginal de memoria_particiones
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