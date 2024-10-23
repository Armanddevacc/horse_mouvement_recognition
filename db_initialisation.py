#-------------------save to data base-----------------
import sqlite3
import json
import numpy as np

def init_table():
    conn = sqlite3.connect("data/test_data.db")
    cursor = conn.cursor()

    # Activer les clés étrangères
    cursor.execute('''PRAGMA foreign_keys = ON;''')
    cursor.execute(f"DROP TABLE IF EXISTS test_data;")

    # Créer la table test_data
    query = """ CREATE TABLE test_data (
        matrice_id TEXT,
        data TEXT,
        label TEXT,
        PRIMARY KEY (matrice_id)
    )"""
    cursor.execute(query)

    # Valider les changements
    conn.commit()
    cursor.close()
    conn.close()


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
