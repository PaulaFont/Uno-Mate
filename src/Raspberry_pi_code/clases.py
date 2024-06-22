
class Carta:
    def __init__(self, color, valor, jugador):
        self.color = color
        self.valor = valor
        self.color_wild = ""
        self.jugador = jugador

    def to_dict(self):
        return {
            'color': self.color,
            'valor': self.valor,
            'color_wild': self.color_wild,
            'jugador': self.jugador
        }


class Partida:
    def __init__(self, noms_jugadors):
        self.jugadors = [Jugador(nom, i) for i, nom in enumerate(noms_jugadors)]
        self.torn_actual = 0
        self.torns_totals = 1
        self.direccio = 0  # Zero es normal
        self.skip_torn = False
        self.cartes_a_robar = 0
        self.skips_acumulats = 1
        self.cadena_robar = False
        self.pila_joc = []
        self.historial = []

    def get_jugadors(self):
        return self.jugadors


class Jugador:
    def __init__(self, nom, torn):
        self.nom = nom
        self.torn_jugador = torn
        self.num_cartes = 7
        self.ganador = False

