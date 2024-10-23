from ModelePrediction import ModelePrediction
import db_initialisation



mon_modele = ModelePrediction('/Users/Utilisateur1/Documents/data/model_cheveux.keras')




def get_matrice():
    X_test,y_test = db_initialisation.load_data()

    # Liste des labels de classe
    labels=["eating" ,"fighting" ,"galloping-natural","galloping-rider" ,"grazing" ,"head-shake" ,"jumping" ,"rolling" ,"rubbing" ,"scared" ,"scratch-biting" ,"shaking","standing","trotting-natural","trotting-rider","walking-natural","walking-rider"]
    #labels = [ "galloping-rider",  # 0 "head-shake",       # 1 "scared",           # 2"scratch-biting",   # 3"standing",         # 4"trotting-rider",   # 5 "walking-natural",  # 6"walking-rider"     # 7]
    mon_modele.afficher_matrice_confusion(X_test, y_test, labels)




X_test,y_test = db_initialisation.load_data()

print(X_test[4])
mon_modele.predire(X_test[4])
print(y_test[4])