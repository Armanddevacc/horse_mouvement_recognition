from ModelePrediction import ModelePrediction
import sqlite3
import numpy as np
import json
mon_modele = ModelePrediction('/Users/Utilisateur1/Documents/data/model_cheveux.keras')



def load_data():
    conn = sqlite3.connect("test_data.db")
    cursor = conn.cursor()
    query = """SELECT data,label FROM test_data"""
    cursor.execute(query)
    data_label_pairs=  cursor.fetchall()

    # Fermer la connexion à la base de données
    cursor.close()
    conn.close()

    # Convertir les données JSON en ndarray
    X_test = np.array([np.array(json.loads(data)) for data, label in data_label_pairs])
    y_test = np.array([np.array(json.loads(label)) for data, label in data_label_pairs])
    return X_test,y_test

def get_matrice():
    X_test,y_test = load_data()

    # Liste des labels de classe
    labels=["eating" ,"fighting" ,"galloping-natural","galloping-rider" ,"grazing" ,"head-shake" ,"jumping" ,"rolling" ,"rubbing" ,"scared" ,"scratch-biting" ,"shaking","standing","trotting-natural","trotting-rider","walking-natural","walking-rider"]
    #labels = [ "galloping-rider",  # 0 "head-shake",       # 1 "scared",           # 2"scratch-biting",   # 3"standing",         # 4"trotting-rider",   # 5 "walking-natural",  # 6"walking-rider"     # 7]
    mon_modele.afficher_matrice_confusion(X_test, y_test, labels)




X_test,y_test = load_data()

print(X_test[4])
mon_modele.predire(X_test[4])
print(y_test[4])