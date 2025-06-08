import time
import numpy as np
import multiprocessing as mp
import os
from itertools import combinations

# Importaciones del sistema
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


# Función que se ejecuta en paralelo: evalúa una bipartición usando EMD
def _evaluar_biparticion_worker(args):
    grupo1, grupo2, subsistema, dists_base = args  # Desempaqueta los argumentos
    indices_grupo1 = np.array(grupo1, dtype=np.int8)  # Convierte a arreglo numpy
    indices_grupo2 = np.array(grupo2, dtype=np.int8)

    try:
        # Realiza la bipartición del subsistema
        particion = subsistema.bipartir(indices_grupo1, indices_grupo2)
        # Calcula la distribución marginal de la partición
        distribucion = particion.distribucion_marginal()
        # Calcula la pérdida EMD respecto a la distribución del sistema
        phi = emd_efecto(distribucion, dists_base)
    except Exception:
        # En caso de error, retorna valores dummy
        phi = float("inf")
        distribucion = DUMMY_ARR

    return grupo1, grupo2, phi, distribucion  # Retorna la partición evaluada


# Clase principal de la estrategia Geométrica
class Geometric(SIA):
    def __init__(self, gestor: Manager):
        super().__init__(gestor)  # Llama constructor base
        # Inicia sesión de perfilamiento
        profiler_manager.start_session(
            f"{NET_LABEL}{len(gestor.estado_inicial)}{gestor.pagina}"
        )
        self.logger = SafeLogger(GEOMETRIC_LABEL)  # Logger para trazas

    # Método principal: ejecuta la estrategia
    @profile(context={TYPE_TAG: GEOMETRIC_ANALYSIS_TAG})
    def aplicar_estrategia(self, condicion, alcance, mecanismo, max_grupo=None):
        # Prepara subsistema con condición, alcance y mecanismo
        self.sia_preparar_subsistema(condicion, alcance, mecanismo)
        n = len(self.sia_subsistema.indices_ncubos)  # Número de nodos

        if n < 2:
            # No es posible particionar menos de 2 nodos
            return Solution(
                estrategia=GEOMETRIC_LABEL,
                perdida=DUMMY_EMD,
                distribucion_subsistema=DUMMY_ARR,
                distribucion_particion=DUMMY_ARR,
                particion=ERROR_PARTITION,
                tiempo_total=0.0,
            )

        # Si no se especifica max_grupo, usa n//2 para cobertura total
        if max_grupo is None:
            max_grupo = n // 2

        # Genera todas las combinaciones de biparticiones posibles
        biparticiones = []
        for k in range(1, max_grupo + 1):
            for grupo1 in combinations(range(n), k):
                grupo2 = [i for i in range(n) if i not in grupo1]
                biparticiones.append((list(grupo1), grupo2))

        # Copia la distribución marginal base del sistema
        dists_base = self.sia_dists_marginales.copy()

        # Prepara los argumentos para ejecutar cada evaluación en paralelo
        args_list = [
            (grupo1, grupo2, self.sia_subsistema, dists_base)
            for grupo1, grupo2 in biparticiones
        ]

        mejor_phi = float("inf")  # Mejor pérdida encontrada
        mejor_bip = None  # Mejores grupos encontrados
        mejor_dist = DUMMY_ARR  # Distribución de la mejor partición

        # Ejecuta en paralelo todas las evaluaciones
        with mp.get_context("spawn").Pool(processes=os.cpu_count()) as pool:
            for grupo1, grupo2, phi, distribucion in pool.imap_unordered(
                _evaluar_biparticion_worker, args_list, chunksize=2
            ):
                if phi < mejor_phi:
                    mejor_phi = phi
                    mejor_bip = (grupo1, grupo2)
                    mejor_dist = distribucion

        # Si se encontró una buena partición, la formatea
        if mejor_bip:
            grupo1, grupo2 = mejor_bip
            prim = [(0, idx) for idx in grupo1]
            dual = [(0, idx) for idx in grupo2]
            mejor_particion_str = fmt_biparte_q(prim, dual)
        else:
            mejor_particion_str = DUMMY_PARTITION

        # Retorna la solución como objeto estándar
        return Solution(
            estrategia=GEOMETRIC_LABEL,
            perdida=mejor_phi,
            distribucion_subsistema=dists_base,
            distribucion_particion=mejor_dist,
            particion=mejor_particion_str,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
        )
