import numpy as np
import heapq
from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.constants.models import GEOMETRIC_ANALYSIS_TAG, GEOMETRIC_LABEL
from src.constants.base import TYPE_TAG, NET_LABEL
from src.middlewares.profile import profiler_manager, profile
from src.middlewares.slogger import SafeLogger


class Geometric(SIA):
    """
    Estrategia Geometric para análisis heurístico del sistema.
    Utiliza distancias de Hamming y una heurística voraz para formar particiones
    que minimicen el costo de transición entre estados.
    """

    def __init__(self, gestor):
        super().__init__(gestor)
        # Inicia una sesión de profiling para analizar tiempos de ejecución
        profiler_manager.start_session(f"{NET_LABEL}{len(gestor.estado_inicial)}{gestor.pagina}")

        self.logger = SafeLogger(GEOMETRIC_LABEL)
        self.memoria_t = {}         # Memoria para guardar costos t(i, j) ya calculados
        self.tabla_costos = {}      # Diccionario: T[v][i][j] → costo de transición para variable v entre estados i y j
        self.vertices = None        # Lista de índices de variables

    @profile(context={TYPE_TAG: GEOMETRIC_ANALYSIS_TAG})
    def aplicar_estrategia(self, condicion, alcance, mecanismo):
        """
        Método principal invocado por el sistema. Prepara el subsistema,
        construye la tabla de costos y ejecuta la heurística voraz.
        """
        self.sia_preparar_subsistema(condicion, alcance, mecanismo)

        estado = self.sia_subsistema.estado_inicial
        n = len(estado)
        self.vertices = list(range(n))  # Variables representadas como índices

        self._construir_tabla_costos()  # Calcula T[v][i][j] para cada variable

        mejor_phi, mejor_particion = self.algorithm_voraz(self.vertices)  # Estrategia voraz

        return Solution(
            estrategia=GEOMETRIC_LABEL,
            perdida=mejor_phi,
            distribucion_subsistema=None,         # Se puede agregar distribución si se desea
            distribucion_particion=None,
            particion=f"Partición heurística: {mejor_particion}",
        )

    def _construir_tabla_costos(self):
        """
        Construye la tabla de costos T[v][i][j] para cada variable v.
        El costo t(i,j) se calcula si i ≠ j.
        """
        n = len(self.sia_subsistema.estado_inicial)
        num_estados = 2 ** n

        for v in range(n):
            self.tabla_costos[v] = np.zeros((num_estados, num_estados))
            for i in range(num_estados):
                for j in range(num_estados):
                    if i != j:
                        self.tabla_costos[v][i][j] = self._calcular_t(i, j, v)

    def _calcular_t(self, i, j, v):
        """
        Calcula el costo t(i, j) para la variable v.
        Incluye costo directo + vecinos adyacentes con un factor γ = 2^(-dH(i, j)).
        Se guarda en memoria para no recalcular.
        """
        clave = (v, i, j)
        if clave in self.memoria_t:
            return self.memoria_t[clave]

        dH = bin(i ^ j).count("1")  # Distancia de Hamming entre i y j
        gamma = 2 ** (-dH)

        X = self.sia_subsistema.tpm[:, v]  # Columna de la TPM asociada a la variable v

        costo_directo = abs(X[i] - X[j])
        vecinos = self._vecinos_adyacentes(i, j)
        suma_vecinos = sum(abs(X[k] - X[j]) for k in vecinos)

        t = gamma * (costo_directo + suma_vecinos)
        self.memoria_t[clave] = t
        return t

    def _vecinos_adyacentes(self, i, j):
        """
        Calcula los vecinos adyacentes de i y j: estados que difieren en un solo bit.
        """
        n = self.sia_subsistema.tpm.shape[0].bit_length() - 1
        vecinos = set()
        for b in range(n):
            vecinos.add(i ^ (1 << b))
            vecinos.add(j ^ (1 << b))
        vecinos.discard(i)
        vecinos.discard(j)
        return list(vecinos)

    def _costo_conjunto(self, variables):
        """
        Calcula el costo promedio de transición para un conjunto de variables.
        Se promedia el costo medio de la tabla T[v] para cada variable en el conjunto.
        """
        total = 0
        for v in variables:
            T = self.tabla_costos[v]
            total += np.mean(T)
        return total / len(variables)

    def algorithm_voraz(self, variables):
        """
        Estrategia heurística voraz:
        - Comienza con cada variable como semilla.
        - Iterativamente añade la que menos aumente el costo.
        - Retorna la mejor partición encontrada.
        """
        n = len(variables)
        mejor_phi = float("inf")
        mejor_conjunto = []

        for inicio in variables:
            conjunto = [inicio]
            restantes = [v for v in variables if v != inicio]
            costo_total = 0

            while restantes:
                mejor_v = None
                mejor_costo = float("inf")
                for v in restantes:
                    costo = self._costo_conjunto(conjunto + [v])
                    if costo < mejor_costo:
                        mejor_costo = costo
                        mejor_v = v

                conjunto.append(mejor_v)
                restantes.remove(mejor_v)
                costo_total = mejor_costo

            if costo_total < mejor_phi:
                mejor_phi = costo_total
                mejor_conjunto = conjunto

        return mejor_phi, mejor_conjunto
