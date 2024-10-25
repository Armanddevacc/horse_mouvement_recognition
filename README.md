# Projet de Réseau de Neurones Convolutif (CNN) pour la Classification Multi-Classes

## Description du projet

Ce projet implémente un **réseau de neurones convolutif (CNN)** pour résoudre un problème de classification multi-classes en utilisant les bibliothèques **TensorFlow** et **Keras**. Le modèle est conçu pour traiter des données d'entrée structurées sous la forme de séquences avec 8 caractéristiques (Ax,Ay,Az,Gx,Gy,Gz,A3,G3) et 200 points par séquence (après padding, enregistrement entre 1 à 2 minutes car la fréquence du capteur est 100Hz). L'objectif est de classer ces données dans l'une des 17 classes disponibles. 
Les 17 classes:
```python
labels=["eating" ,"fighting" ,"galloping-natural","galloping-rider" ,"grazing" ,"head-shake" ,"jumping" ,"rolling" ,"rubbing" ,"scared" ,"scratch-biting" ,"shaking","standing","trotting-natural","trotting-rider","walking-natural","walking-rider"]
```
La base de donnée utilisé est disponible ici: https://data.4tu.nl/articles/Horsing_Around_--_A_Dataset_Comprising_Horse_Movement/12687551 Ce groupe a aussi publié un article sur leur recherche https://www.researchgate.net/publication/335978678_Horsing_Around-A_Dataset_Comprising_Horse_Movement

## Architecture du réseau

Le réseau de neurones convolutif utilisé ici se compose des couches suivantes :

- **Couche d'entrée** : Reçoit les séquences de longueur 200 avec 8 caractéristiques chacune.
- **Conv1D (32 filtres)** : Applique 32 filtres de convolution avec une taille de noyau de 3.
- **MaxPooling1D** : Réduit la dimension spatiale après la première convolution.
- **Conv1D (64 filtres)** : Applique 64 filtres de convolution pour extraire des caractéristiques plus complexes.
- **MaxPooling1D** : Nouvelle réduction spatiale après la deuxième convolution.
- **Flatten** : Aplatit la sortie des couches convolutives pour permettre leur passage à une couche entièrement connectée.
- **Dense (128 neurones)** : Combine les caractéristiques avec 128 neurones.
- **Dropout (50%)** : Utilisé pour éviter le surapprentissage.
- **Couche de sortie (17 neurones)** : Utilise une fonction d'activation **softmax** pour prédire la classe parmi les 17 classes disponibles.

### Schéma de l'architecture

```plaintext
Input Layer (8 features)
   |
Conv1D (32 filters) → MaxPooling1D
   |
Conv1D (64 filters) → MaxPooling1D
   |
Flatten → Dense (128 neurons) → Dropout (50%)
   |
Output Layer (17 classes)
```

### Prérequis

Assurez-vous que les bibliothèques suivantes sont installées dans votre environnement Python :

pip install tensorflow pandas matplotlib seaborn

## Structure du projet

```bash
Copier le code
├── csv/                       # Dossier contenant les fichiers CSV
│   ├── subject_2_Happy_part_1.csv
│   ├── subject_2_Happy_part_2.csv
│   └── ...                    # Jusqu'à subject_2_Happy_part_12.csv
├── data_formating.py           # Fichier contenant la fonction de formatage des données
├── neurone2.py                 # Fichier principal qui contient le modèle et son entraînement
└── README.md                   # Documentation du projet
```

## Utilisation du modèle

1. Formatage des données
Le fichier data_formating.py contient la fonction get_list qui permet de charger et concaténer les fichiers CSV de données d'entraînement.

```python
import pandas as pd
import os

def get_list(file_path, rang):
    dataframes = []
    for i in range(1, rang + 1):
        current_file_path = os.path.join(f'{file_path}_{i}.csv')
        df = pd.read_csv(current_file_path, keep_default_na=False)
        dataframes.append(df)

    data = pd.concat(dataframes, ignore_index=True)
    data = data.drop(['Mx', 'My','Mz',"M3D","segment","subject"], axis=1)
    return data
```

Cette fonction lit et concatène les fichiers CSV de la série subject_2_Happy_part_X.csv, puis nettoie certaines colonnes inutiles.

2. Modèle de réseau de neurones
Le modèle de réseau de neurones convolutif (CNN) est défini dans le fichier neurone2.py :

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
```

## Définir les dimensions d'entrée : 8 (features) et 200 (nombre de points après padding)
```python
input_shape = (200, 8)
```
## Construction du modèle
```python
model = Sequential()

model.add(Conv1D(32, 3, activation='relu', input_shape=input_shape))
model.add(MaxPooling1D(2))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(17, activation='softmax'))
```
## Compilation du modèle
```python
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
```
## Entraîner le modèle
```python
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
```
3. Entraînement du modèle
Avant d'entraîner le modèle, assurez-vous d'avoir préparé les données :

```python
from tensorflow.keras.utils import to_categorical
```
## Convertir les labels en one-hot encoding
```python
y_train = to_categorical(train_labels, num_classes=17)
y_val = to_categorical(test_labels, num_classes=17)
```
## Entraîner le modèle
```python
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
```
Le modèle est ensuite compilé et entraîné sur vos données d'entraînement avec les labels one-hot.

4. Générer un graphique de l'architecture
Le fichier neurone2.py génère également un graphique représentant l'architecture du réseau :

```python
from tensorflow.keras.utils import plot_model
```


## Résultats attendus

À chaque époque d'entraînement, vous verrez apparaître les résultats de la précision et de la perte pour l'ensemble d'entraînement et l'ensemble de validation, par exemple :

```plaintext
Copier le code
Epoch 1/10
137/137 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.8014 - loss: 0.7025 - val_accuracy: 0.9360 - val_loss: 0.2073
Epoch 2/10
137/137 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.9293 - loss: 0.2252 - val_accuracy: 0.9488 - val_loss: 0.1382
...
```
### Analyse des performances

Une fois l'entraînement terminé, vous pouvez évaluer les performances de votre modèle en affichant la matrice de confusion et d'autres métriques comme la précision, le rappel et le F1-score pour chaque classe.

```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
```
## Générer les prédictions
```python
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)
```
## Afficher la matrice de confusion
```python
conf_matrix = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
Contributions
```

## API
Ce repositorie a pour objectif à servir de serveur pour que des utilisateurs mobiles puis faire des requetes à cette API. Ainsi les utilisateurs peuvent depuis leur mobile utiliser le réseau de neurone en utilisant la route:
```bash 
POST localhost:5000/predict HTTP/1.1
```
Cette route est amené à être sécurisé en utilisant des tockens JWT.
```bash
Authorization: Bearer <token> 
```

Si vous souhaitez contribuer à ce projet, n'hésitez pas à ouvrir une Pull Request sur ce dépôt GitHub ou à contacter l'auteur pour discuter des améliorations potentielles.


