from src.controllers.manager import Manager

from src.strategies.force import BruteForce
from src.strategies.phi import Phi
from src.strategies.q_nodes import QNodes
from src.strategies.geometric import Geometric


def iniciar():
    """Punto de entrada principal"""

    estado_inicial = "100000000000000" 
    condiciones =    "111111111111111"
    alcance =        "110110110110110"  # t+1
    mecanismo =      "101010101010101"  # t

    gestor_sistema = Manager(estado_inicial)

    analizador_geom = Geometric(gestor_sistema)  # 👈 usar clase real

    sia_resultado = analizador_geom.aplicar_estrategia(
        condiciones,
        alcance,
        mecanismo,
    )

    print(sia_resultado)
