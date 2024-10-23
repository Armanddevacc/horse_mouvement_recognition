import numpy as np
import data_formating
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import json
import sqlite3


L = ["csv/subject_1_Viva_part_","csv/subject_2_Happy_part_","csv/subject_3_Zafir_part_","csv/subject_4_Blondy_part_","csv/subject_5_Flower_part_","csv/subject_6_Pan_part_","csv/subject_7_Driekus_part_","csv/subject_8_Galoway_part_","csv/subject_9_Barino_part_","csv/subject_10_Zonnerante_part_","csv/subject_11_Patron_part_","csv/subject_12_Duke_part_","csv/subject_13_Porthos_part_","csv/subject_14_Bacardi_part_","csv/subject_15_Niro_part_","csv/subject_16_Sense_part_","csv/subject_17_Clever_part_","csv/subject_18_Noortje_part_"]
L2=[(1,9),(2,12),(3,7),(4,4),(5,9),(6,8),(7,12),(8,12), (9,8),(10,10),(11,12),(12,9),(13,8),(14,12),(15,5),(16,5),(17,2),(18,3)]
L_test,L_test2 = ["csv/subject_1_Viva_part_"],[(1,9)]


###----------------------------------formatage données--------------------------------------

#toutes matrice 8*? 
group_list, label_list= data_formating.main(L,L2)
print("done:   -------------creation matrices succeful--------------")


#sépare donnée test et entrainement
train_groups,train_groups_label,test_groups,test_groups_label= data_formating.sequence_data(group_list, label_list)
print("done:   -------------formatage succeful--------------")



##________________________________________Padding--------------------------------------------
def pading_matrice(X, pad_value, max_value=-1):
    """
    Ajoute des sous-listes vides aux groupes qui ont moins de sous-listes que la longueur maximale.
    """
    for group in X:
        while len(group) < max_value:
            group.append(pad_value)
    return X
# application du Padding des groupes d'entraînement et de test

X_padded_train = pading_matrice(train_groups, [0,0,0,0,0,0,0,0] ,200)  
X_padded_test = pading_matrice(test_groups, [0,0,0,0,0,0,0,0] ,200)


print("done:   -------------padding succeful--------------")






###-------------------------------encode-------------------------------
from sklearn.preprocessing import LabelEncoder

# Initialisation de l'encodeur
label_encoder = LabelEncoder()

# Encodage des labels d'entraînement
train_labels = label_encoder.fit_transform(train_groups_label)

# Encodage des labels de test
test_labels = label_encoder.transform(test_groups_label)


# Convertir les labels en flottants
y_train = np.array(train_labels, dtype=np.float32)
y_val = np.array(test_labels, dtype=np.float32)

# Affichage de la correspondance entre chaque label original et son encodage

print("\nCorrespondance globale des classes pour les données globales  :")
for original_label, encoded_label in enumerate(label_encoder.classes_):
    print(f"{encoded_label} --> {original_label}")

# Convertir les matrices en flottants
X_padded_train = np.array(X_padded_train, dtype=np.float32)
X_padded_test = np.array(X_padded_test, dtype=np.float32)
print("done:   -------------encoding succeful--------------")



###-------------------------------normalisation-------------------------------
from sklearn.preprocessing import StandardScaler
import numpy as np

# Reshape les données pour qu'elles soient 2D (StandardScaler travaille sur 2D)
X_padded_reshaped = X_padded_train.reshape(-1, X_padded_train.shape[-1])

# Initialisation du StandardScaler
scaler = StandardScaler()

# Entraînement et transformation des données
features_scaled = scaler.fit_transform(X_padded_reshaped)

# Reshape pour revenir à la forme 3D initiale
X_train = features_scaled.reshape(X_padded_train.shape)


X_padded_reshaped = X_padded_test.reshape(-1, X_padded_test.shape[-1])
features_scaled = scaler.transform(X_padded_reshaped)
X_val = features_scaled.reshape(X_padded_test.shape)


# Affichage des données normalisées
print("-------Données normalisées (StandardScaler) appliquées à des matrices 3D---------")




###-------------------------------réseau de neurone-------------------------------

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Définir les dimensions d'entrée : 8 (features) et 200 (nombre de points après padding)
input_shape = (200, 8)

# Construction du modèle
model = Sequential()

# Ajouter une couche convolutionnelle avec 32 filtres, kernel de taille 3, et activation ReLU
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))

# Ajouter une couche de max-pooling pour réduire la dimensionnalité
model.add(MaxPooling1D(pool_size=2))

# Ajouter une autre couche convolutionnelle
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))

# Ajouter une autre couche de max-pooling
model.add(MaxPooling1D(pool_size=2))

# Aplatir la sortie de la couche convolutionnelle pour passer aux couches denses
model.add(Flatten())

# Ajouter une couche dense entièrement connectée avec 128 neurones et activation ReLU
model.add(Dense(128, activation='relu'))

# Ajouter un Dropout pour éviter le surapprentissage
model.add(Dropout(0.5))

# Ajouter la couche de sortie pour la classification (nombre de classes = 3 par exemple)
model.add(Dense(17, activation='softmax'))

# Compiler le modèle avec l'optimiseur Adam et la fonction de perte cross-entropy
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Afficher la structure du modèle
model.summary()




from tensorflow.keras.utils import to_categorical

# Convertir les labels en one-hot encoding
y_train = to_categorical(train_labels, num_classes=17)  
y_val = to_categorical(test_labels, num_classes=17)

# Maintenant, 'y_train' et 'y_val' seront sous forme one-hot


# Entraîner le modèle (X_train: tes données d'entraînement, y_train: tes labels)
# Ici y_train doit être un encodage one-hot de tes labels
print(y_train,' ',y_val)
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))




# Sauvegarder tout le modèle
model.save('model_cheveux.keras')




import db_initialisation
db_initialisation.init_table()
conn = sqlite3.connect("test_data.db")
cursor = conn.cursor()
for i, (data, label) in enumerate(zip(X_val, y_val)):
    data_list = data.tolist()  # Convertir ndarray en liste

    # Convertir les données en format JSON
    data_json = json.dumps(data_list)
    label = json.dumps(label.tolist())

    query = """ INSERT INTO test_data (matrice_id,data,label) VALUES (?,?,?)"""

    cursor.execute(query, (i,data_json,label))
conn.commit()
cursor.close()
conn.close()


print("finished")

