from src.controllers.manager import Manager

from src.strategies.force import BruteForce
from src.strategies.phi import Phi
from src.strategies.IBC import IBC
from src.strategies.q_nodes import QNodes



def iniciar():
    """Punto de entrada principal"""
                    # ABCD #
    estado_inicial = "100000000000000"
    condiciones =    "111111111111111"
    alcance =        "111111111111111" #t+1
    mecanismo =      "111111111111111" #t

    gestor_sistema = Manager(estado_inicial)

    analizador_bf = IBC(gestor_sistema)

    sia_cero = analizador_bf.aplicar_estrategia(
        condiciones,
        alcance,
        mecanismo,
    )

    print(sia_cero)
