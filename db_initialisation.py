#-------------------save to data base-----------------
import sqlite3


def init_table():
    conn = sqlite3.connect("test_data.db")
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