from tensorflow.keras.models import load_model # type: ignore
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import db_initialisation
from sklearn.metrics import classification_report

class ModelePrediction:
    def __init__(self, model_path):
        # Charger le modèle lors de l'initialisation de la classe
        self.model = load_model(model_path)

    def predire(self, X):
        # Utiliser le modèle pour prédire des résultats
        y_pred_proba =self.model.predict(X)
        return np.argmax(y_pred_proba, axis=1)

    def afficher_matrice_confusion(self):
        """
        affiche la matrice de confusion quand on l'appelle
        """
        X, y_true = db_initialisation.load_data()

        # Liste des labels de classe
        labels=["eating" ,"fighting" ,"galloping-natural","galloping-rider" ,"grazing" ,"head-shake" ,"jumping" ,"rolling" ,"rubbing" ,"scared" ,"scratch-biting" ,"shaking","standing","trotting-natural","trotting-rider","walking-natural","walking-rider"]
        #labels = [ "galloping-rider",  # 0 "head-shake",       # 1 "scared",           # 2"scratch-biting",   # 3"standing",         # 4"trotting-rider",   # 5 "walking-natural",  # 6"walking-rider"     # 7]

        # Prédire les résultats
        y_pred = self.predire(X)
        y_true_classes = np.argmax(y_true, axis=1)

        # Calculer la matrice de confusion
        cm = confusion_matrix(y_true_classes, y_pred)

        # Afficher la matrice de confusion avec seaborn
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.ylabel('Vraies classes')
        plt.xlabel('Classes prédites')
        plt.title('Matrice de confusion')
        plt.show()

    def affichage_classification_report(self):
        X_test,y_test = db_initialisation.load_data()

        ## one hot en int
        y_test_labels = np.argmax(y_test, axis=1)

        y_pred_mlp = self.predire(X_test)

        print('Classification Report (Multi-Layer Perceptron Classifier):\n', classification_report(y_test_labels, y_pred_mlp))


                
