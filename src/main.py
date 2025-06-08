from src.controllers.manager import Manager

from src.strategies.force import BruteForce
from src.strategies.phi import Phi
from src.strategies.q_nodes import QNodes
from src.strategies.geometric import Geometric


def iniciar():
    """Punto de entrada principal"""

    estado_inicial = "1000000000"  # 15 bits → 15 nodos
    condiciones =    "1111111111"
    alcance =        "1111111111"  # t+1
    mecanismo =      "1111111111"  # t

    gestor_sistema = Manager(estado_inicial)

    analizador_geom = Geometric(gestor_sistema)  # 👈 usar clase real

    sia_resultado = analizador_geom.aplicar_estrategia(
        condiciones,
        alcance,
        mecanismo,
    )

    print(sia_resultado)
