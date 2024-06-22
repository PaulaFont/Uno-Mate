import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("unoMatePrivateKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()


# Obtenir un document de la firestore
def get_document(collection_name, document_id):
    doc_ref = db.collection(collection_name).document(document_id)
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict()
    else:
        print("No such document!")
        return None


def get_nom(collection_name, document_id):
    doc_ref = db.collection(collection_name).document(document_id)
    doc = doc_ref.get()
    if doc.exists:
        data = doc.to_dict()
        return data.get('nombre')
    else:
        print("No such document!")
        return None


# Enviar dades a la Firebase
def save_game(document_id, data):
    db.collection("partidas").document(document_id).set(data)


def actualitzar_jugadors(document_id, data):
    db.collection("usuarios").document(document_id).set(data)


"""
# Rebre les dades de la partida a partir de la seva ID
partida_id = 'partidaID'
collection_name = 'partidas'
document = firebase.get_document(collection_name, partida_id)
print(f"Document data: {document}")

# Enviar nova partida jugada
nova_id_partida = "EjemploMisisuko"
game_data = {
        'fecha': '22/05/2024',
        'hora': '10:00',
        'nombre': 'partida2',
        'userId': 'IDJugadorActual',
        'torns': 10,
        'nombre de jugadors': len(partida.get_jugadors()),
        'resultado': {'Ganador': jugadors[3], 'Puntuacion': 100}
}
firebase.save_game(nova_id_partida,game_data)
"""