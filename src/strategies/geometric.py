import time
import numpy as np
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

class Geometric(SIA):
    """
    Estrategia Geométrica para bipartición óptima mediante propiedades topológicas.
    Calcula la tabla de costos de transición entre estados usando la función recursiva geométrica.
    Evalúa biparticiones candidatas usando heurísticas.
    Compatible con los modelos y tags definidos en models.py.
    """

    def __init__(self, gestor: Manager):
        super().__init__(gestor)
        profiler_manager.start_session(
            f"{NET_LABEL}{len(gestor.estado_inicial)}{gestor.pagina}"
        )
        self.logger = SafeLogger(GEOMETRIC_LABEL)
        self.tabla_costos = {}    # T[v][i][j]: costos de transición para cada variable
        self.memo_t = {}          # Memoization para t(i, j, v)
        self.X = None             # X[v][i]: tensor elemental de variable v en estado i

    @profile(context={TYPE_TAG: GEOMETRIC_ANALYSIS_TAG})
    def aplicar_estrategia(self, condicion, alcance, mecanismo):
        self.sia_preparar_subsistema(condicion, alcance, mecanismo)
        n = len(self.sia_subsistema.indices_ncubos)
        num_estados = 2 ** n
        if n < 2:
            # Igual que en otras estrategias, retorna error si no se puede particionar
            return Solution(
                estrategia=GEOMETRIC_LABEL,
                perdida=DUMMY_EMD,
                distribucion_subsistema=DUMMY_ARR,
                distribucion_particion=DUMMY_ARR,
                particion=ERROR_PARTITION,
                tiempo_total=0.0,
            )

        self.X = self._extraer_tensores_elementales(n, num_estados)
        self._construir_tabla_costos(n, num_estados)
        mejor_phi, mejor_bip = self._heuristica_biparticion(n)
        # Formato de partición amigable (ejemplo: "A|BC" o "0|1,2")
        mejor_particion_str = self._formatear_particion(mejor_bip, n)

        # Supón que mejor_bip = (grupo1, grupo2)
        if mejor_bip:
            # indices para bipartir (opcional: aquí podrías usar dims)
            indices_grupo1 = np.array(mejor_bip[0], dtype=np.int8)
            indices_grupo2 = np.array(mejor_bip[1], dtype=np.int8)
            # bipartir el sistema según esos grupos (adapta si tienes lógica especial)
            particion = self.sia_subsistema.bipartir(indices_grupo1, indices_grupo2)
            distribucion_particion = particion.distribucion_marginal()
        else:
            distribucion_particion = DUMMY_ARR

        return Solution(
            estrategia=GEOMETRIC_LABEL,
            perdida=mejor_phi,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=distribucion_particion,
            particion=mejor_particion_str if mejor_bip else DUMMY_PARTITION,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
        )





    def _extraer_tensores_elementales(self, n, num_estados):
        """
        X[v][i]: valor del tensor elemental de la variable v para el estado i.
        Usa los n-cubos del sistema. Accede a .data con los bits del estado.
        """
        X = np.zeros((n, num_estados))
        for v, ncubo in enumerate(self.sia_subsistema.ncubos):
            for i in range(num_estados):
                bits = [(i >> k) & 1 for k in reversed(range(n))]
                X[v, i] = ncubo.data[tuple(bits)]
        return X

    def _construir_tabla_costos(self, n, num_estados):
        """
        Llena la tabla de costos T[v][i][j] usando la función recursiva t(i, j, v).
        """
        for v in range(n):
            self.tabla_costos[v] = np.zeros((num_estados, num_estados))
            for i in range(num_estados):
                for j in range(num_estados):
                    self.tabla_costos[v][i, j] = self._calcular_t(i, j, v, n)

    def _calcular_t(self, i, j, v, n):
        """
        Cálculo recursivo de t(i, j) según la función topológica del documento.
        """
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
        """
        Devuelve los vecinos de i que se acercan a j en distancia de Hamming (caminos óptimos).
        """
        vecinos = []
        dH_ij = bin(i ^ j).count("1")
        for b in range(n):
            k = i ^ (1 << b)
            if bin(k ^ j).count("1") < dH_ij:
                vecinos.append(k)
        return vecinos

    def _heuristica_biparticion(self, n):
        """
        Heurística simple: prueba todas las biparticiones dejando una variable sola y el resto en el otro grupo.
        Se puede reemplazar por una heurística más avanzada (greedy, swap, clustering, etc.)
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
        Evalúa la bipartición sumando los costos promedio entre variables de ambos grupos.
        Puede modificarse para mejorar la métrica según la función geométrica/tensorial.
        """
        if not grupo1 or not grupo2:
            return float("inf")
        total = 0
        for v1 in grupo1:
            for v2 in grupo2:
                total += np.mean(self.tabla_costos[v1]) + np.mean(self.tabla_costos[v2])
        return total / (len(grupo1) * len(grupo2))

    def _formatear_particion(self, biparticion, n):
        """
        Devuelve una representación amigable de la bipartición (por índices, pero puedes mejorarlo con literales)
        """
        if biparticion is None:
            return DUMMY_PARTITION
        g1, g2 = biparticion
        # Por defecto muestra los índices, pero puedes mapearlos a literales si quieres
        return f"{g1}|{g2}"
