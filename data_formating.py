import pandas as pd
import numpy as np
import os


def main(L,L2):
    matrices, labels = [],[]
    for i in range(len(L)):
        link = L[i]
        dataframes = []
        for i in range(1, L2[i][1]+1):  # De 1 à 12 inclus
            print(f'{link}{i}.csv')
            fp = os.path.join(f'{link}{i}.csv')
            
            # Lire le fichier CSV et l'ajouter à la liste
            df = pd.read_csv(fp, keep_default_na=False)
            dataframes.append(df)
        group_list, label_list= get_matrices(dataframes)
        matrices += group_list
        labels += label_list
    return matrices,labels



def get_matrices(dataframes):

    # Concaténer tous les DataFrames ensemble
    data = pd.concat(dataframes, ignore_index=True)

    data = data.drop(['Mx', 'My','Mz',"M3D","segment","subject"], axis=1)

    # Initialisation des listes pour stocker les groupes
    group_list = []
    label_list = []
    current_group = []
    current_label = None

    # Parcourir les données pour grouper les valeurs consécutives ayant le même label
    for i,row in data.iterrows():
        label = row['label']
        
        if pd.isna(label) or label == "unknown" or label == "null" :
            continue
        elif current_label is None:
            current_label = label
            current_group=[(row.tolist())[:-1]]
            

        
        elif label == current_label:
            #row['label'] = np.nan
            #print((row.tolist())," and ",(row.tolist())[:-1])
            current_group.append((row.tolist())[:-1])
            if len(current_group) == 100:
                group_list.append(current_group)
                label_list.append(label)
                current_group=[row.tolist()[:-1]]
        else:
            group_list.append(current_group)
            label_list.append(label)
            current_group=[(row.tolist())[:-1]]
            current_label = label



    return group_list, label_list




def sequence_data(group_list, label_list):
    from sklearn.model_selection import train_test_split

    train_groups, test_groups, train_groups_label, test_groups_label = train_test_split(
        group_list, label_list, test_size=0.3, random_state=42
    )
    return train_groups,train_groups_label,test_groups,test_groups_label