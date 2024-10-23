from ModelePrediction import ModelePrediction
import db_initialisation
import numpy as np


mon_modele = ModelePrediction('data/model_cheveux.keras')


mon_modele.afficher_matrice_confusion()
mon_modele.affichage_classification_report()




