import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import ExtraTreesRegressor
import sys
from sklearn.externals import joblib




def fitted_Q (data_path,action_list, discount = 0.95,n_iterations = 30):
    #Get the data
    data = pd.read_csv(data_path)
    data = data.values

    # [bar_center_x, bar_velocity, fruit_center_x, fruit_center_y, action]
    X = data[:,:5]

    #rewards
    y = data[:,5]
    model = ExtraTreesRegressor(n_estimators = 10)

    #First training -> just trying to predict immediate reward
    model.fit(X,y)

    for i in range(n_iterations):
        print("yo")


        predictions = []
        batch = []
        batch_size = 256 # essayer d'autres valeurs ?

        for j, line in enumerate(data[:,6:]):
            #Pour tous les états "suivants" = data[:,6:], on teste toutes les actions possibles
            #et on garde la meilleure en terme de récompense prédite par le modèle précédent
            #Ca suppose aussi que c'est la fonction qui appelle ceci qui doit discretiser les actions

            #TODO Cette boucle est hyper lente, je ne sais pas si il y a moyen de l'améliorer,

            # append les 5 lignes au batch (5 = len(action_list))
            batch.extend([np.append(line,a) for a in action_list])


            # attention le dernier batch peut être de taille quelconque
            if j % batch_size == batch_size-1 or j == len(data)-1:

                batch_predictions = model.predict(batch)

                # ! placeholder !
                # à la place: pour chaque tranche de 5 valeurs, prendre le max
                # si j'ai bien compris
                #batch_predictions = batch_predictions[:j%batch_size+1]

                predictions.extend(batch_predictions)

                # reset pour le prochain batch
                batch = []

        best_prediction = []
        nb_actions = len(action_list)

        
        for i in range(len(y)-1):
            
            best_prediction.append(max(predictions[i * nb_actions : (i+1) * nb_actions]))
        best_prediction.append(max(predictions[-nb_actions:]))
            
        #Nouveau y = récompense + gamma*prédiction de la récompense max d'après
        y = data[:,5] + discount* np.array(best_prediction)
        model.fit(X,y)
    joblib.dump(model,"big_tree.sav")
    
fitted_Q("tree_data_big.csv", [-80,-60,-40,-20,-10,-5,0,5,10,20,40,60,80])
#tree = joblib.load("test_model.sav")
#print(tree.predict([[160.0,0.0,20,300,-50]]))
