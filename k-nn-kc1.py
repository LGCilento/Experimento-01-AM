#### IMPORTANTE, este script apenas gera os valores de TP e FP para os algurítimos descritos. A visualização destes resultados é feita em um notebook no jupyter


import pandas as pd
import numpy as np
#from google.colab import files
from sklearn import preprocessing
from scipy.io import arff
from sklearn.utils import shuffle
from statistics import mode
from statistics import mean 
import code
import matplotlib.pyplot as plt

def euclideanDistance(x, y):
    dist = (np.linalg.norm(x-y))
    return dist

def adaptativeEuclideanDistance(x, y,z):
    if z == 0:
        z = 0.00000001
    diff = euclideanDistance(x, y)
    if diff > 0:
        diff = diff - 0.0000000001
    dist = diff/z
    return dist

def knn_normal(k, train, test, test_labels):
    train_aux = train.drop(columns=['problems'])
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    total = test.shape[0]
    for l in test.index:
        i = test.iloc[l]
        i_neibours = []
        for j in train_aux.index:
            neibour = (train.iloc[j]["problems"],
                       euclideanDistance(i, train_aux.iloc[j]))
            i_neibours.append(neibour)
        i_neibours = sorted(i_neibours, key=lambda x: x[1])
        get_nearst_neibours = i_neibours[:k]
        get_nearst_neibours_values = [a for a, b in get_nearst_neibours]
        classification = int(mode(get_nearst_neibours_values))
        real_value = test_labels[l]
        if (classification > 0 and real_value > 0):
            true_positive += 1
        elif classification > 0 and real_value == 0:
            false_positive += 1
        elif classification == 0 and real_value == 0:
            true_negative += 1
        elif classification == 0 and real_value > 0:
            false_negative += 1
    return true_positive, false_positive, true_negative, false_negative

def knn_weight(k, train, test, test_labels):
    train_aux = train.drop(columns=['problems'])
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    total = test.shape[0]
    for l in test.index:
        i = test.iloc[l]
        i_neibours = []

        for j in train_aux.index:
            neibour = (train.iloc[j]["problems"],euclideanDistance(i, train_aux.iloc[j]))
            i_neibours.append(neibour)

        i_neibours = sorted(i_neibours, key=lambda x: x[1])
        get_nearst_neibours = i_neibours[:k]
        get_nearst_neibours_values = [a for a, b in get_nearst_neibours]
        get_nearst_neibours = i_neibours[:k]
        classification = int(mode(get_nearst_neibours_values))
        real_value = test_labels[l]
        #classification = int(mode(get_nearst_neibours_values))
        real_value = test_labels[l]

        if (classification > 0 and real_value > 0):
            true_positive += 1
        elif classification > 0 and real_value == 0:
            false_positive += 1
        elif classification == 0 and real_value == 0:
            true_negative += 1
        elif classification == 0 and real_value > 0:
            false_negative += 1
    return true_positive, false_positive, true_negative, false_negative

def knn_adaptative(k, train, test, test_labels,max_dist_list):
    train_aux = train.drop(columns=['problems'])
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    total = test.shape[0]
    for l in test.index:
        i = test.iloc[l]
        max_dist = [item for item in max_dist_list if item[0] == l][0][1]
        i_neibours = []
        for j in train_aux.index:
            neibour = (train.iloc[j]["problems"],adaptativeEuclideanDistance(i, train_aux.iloc[j],test_labels[l]))
            i_neibours.append(neibour)
        i_neibours = sorted(i_neibours, key=lambda x: x[1])
        get_nearst_neibours = i_neibours[:k]
        get_nearst_neibours_values = [a for a, b in get_nearst_neibours]
        classification = int(mode(get_nearst_neibours_values))
        real_value = test_labels[l]
        if (classification > 0 and real_value > 0):
            true_positive += 1
        elif classification > 0 and real_value == 0:
            false_positive += 1
        elif classification == 0 and real_value == 0:
            true_negative += 1
        elif classification == 0 and real_value > 0:
            false_negative += 1
    return true_positive, false_positive, true_negative, false_negative

def k_fold_manual(df,option,k):
    df = shuffle(df)

    df_divided_list = np.array_split(df, 10)

    true_positive_rate_list = []
    false_positive_rate_list = []

    for p in range(0,10):
        df_aux = df_divided_list.copy()
        test = df_aux[p]
        df_aux.pop(p)
        train = pd.concat(df_aux)

        test = test.reset_index(drop=True)
        train = train.reset_index(drop=True)

        test_labels = test['problems'].values
        test = test.drop(columns=['problems'])

        train_positive = train.loc[train['problems'] > 0]
        train_positive = train_positive.drop(columns=['problems'])

        train_negative = train.loc[train['problems'] == 0]
        train_negative = train_negative.drop(columns=['problems'])

        indexes_from_negatives = train_negative.index
        indexes_from_positives = train_positive.index

        if option == 2:
            min_dist_list_negatives = []
            index = 0
            
            #code.interact(local=dict(globals(), **locals()))
            for i in train_negative.values:
                min_dist = 10000000
                for j in train_positive.values:
                    if euclideanDistance(i, j) < min_dist:
                        min_dist = euclideanDistance(i, j)
                result = (indexes_from_negatives[index], min_dist)
                min_dist_list_negatives.append(result)
                index += 1


            min_dist_list_positives = []
            index = 0
            indexes = train_negative.index
            for i in train_positive.values:
                min_dist = 10000000
                for j in train_negative.values:
                    if euclideanDistance(i, j) < min_dist:
                        min_dist = euclideanDistance(i, j)
                result = (indexes_from_positives[index], min_dist)
                min_dist_list_positives.append(result)
                index += 1

            min_dist_list = sorted(min_dist_list_negatives +min_dist_list_positives, key=lambda x: x[0])

        if (option == 0):
            true_positive, false_positive, true_negative, false_negative = knn_normal( k, train, test, test_labels)
        elif option == 1:
            true_positive, false_positive, true_negative, false_negative = knn_weight( k, train, test, test_labels)
        elif option == 2:
            true_positive, false_positive, true_negative, false_negative = knn_adaptative( k, train, test, test_labels,min_dist_list)

        if true_positive + false_negative == 0:
            tp_rate = 0
        else:
            tp_rate = true_positive/(true_positive + false_negative)
        if false_positive + true_negative == 0:
            fp_rate = 0
        else:
            fp_rate = false_positive/(false_positive + true_negative)

        true_positive_rate_list.append(tp_rate)
        false_positive_rate_list.append(fp_rate)
    tp_rate_mean = mean(true_positive_rate_list)
    fp_rate_mean = mean(false_positive_rate_list)
    print("K = {}: tp_rate = {}, fp_rate = {}".format(k,tp_rate_mean,fp_rate_mean))
    return tp_rate_mean,fp_rate_mean
      
database_name_1 = 'kc1.arff'
database_name_2 = 'kc2.arff'

for i in [database_name_1,database_name_2]:
    data = arff.loadarff(i)

    df = pd.DataFrame(data[0])

    if i == 'kc2.arff':
        df['problems'] = df['problems'].apply(lambda x: x.decode("utf-8"))
        df['problems'] = df['problems'].map({"no": 0, "yes": 1})
        df['problems']
    elif i == 'kc1.arff':
        df.rename(columns = {'defects': 'problems'}, inplace = True)
        df['problems'] = df['problems'].apply(lambda x: x.decode("utf-8"))
        df['problems'] = df['problems'].map({"false": 0, "true": 1})
        df['problems']


    tp_rate_list_normal = []
    fp_rate_list_normal = []

    tp_rate_list_weith = []
    fp_rate_list_weith = []

    tp_rate_list_adaptative = []
    fp_rate_list_adaptative = []
    
    print("{}: INICIANDO TESTES PARA k-NN normal".format(i))
    for k in [1,2,3,5,7,9,11,13,15]:
        tp_rate,fp_rate =  k_fold_manual(df,0,k)
        tp_rate_list_normal.append(tp_rate)
        fp_rate_list_normal.append(fp_rate)
    print("TESTES CONCLUÍDOS PARA k-NN normal: Segue abaixo a lista das taxas de tp e fp")
    print("tp -> {}".format(str(tp_rate_list_normal)))
    print("fp -> {}".format(str(fp_rate_list_normal)))
    print("")
    
    print("{}: INICIANDO TESTES PARA k-NN com peso".format(i))
    for k in [15]:
        tp_rate,fp_rate =  k_fold_manual(df,1,k)
        tp_rate_list_weith.append(tp_rate)
        fp_rate_list_weith.append(fp_rate)

    print("TESTES CONCLUÍDOS PARA k-NN normal: Segue abaixo a lista das taxas de tp e fp")
    print("tp -> {}".format(str(tp_rate_list_weith)))
    print("fp -> {}".format(str(fp_rate_list_weith)))
    print("")
    
    print("{}: INICIANDO TESTES PARA k-NN adaptativo".format(i))
    for k in [15]:
        tp_rate,fp_rate =  k_fold_manual(df,2,k)
        tp_rate_list_adaptative.append(tp_rate)
        fp_rate_list_adaptative.append(fp_rate)
    print("TESTES CONCLUÍDOS PARA k-NN normal: Segue abaixo a lista das taxas de tp e fp")
    print("tp -> {}".format(str(tp_rate_list_adaptative)))
    print("fp -> {}".format(str(fp_rate_list_adaptative)))
    print("")


