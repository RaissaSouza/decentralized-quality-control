from pickle import FALSE
from datagenerator_test import DataGenerator
from sklearn.metrics import confusion_matrix, accuracy_score,  roc_curve, auc, f1_score
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
import SimpleITK as sitk
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from numpy.random import seed
seed(1)
tf.random.set_seed = 1
random.seed(1)
import argparse
import os
import sys
os.environ['TF_DETERMINISTIC_OPS'] = '1'


#parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-fn', type=str, help='filename to infer')
parser.add_argument('-en', type=str, help='encoderfolder path to infer')
parser.add_argument('-pd', type=str, help='classifierfolder path to infer')
parser.add_argument('-o', type=str, help='output name')
args = parser.parse_args()
import csv

def model_eval(y_test, y_pred_raw):
    y_pred = (y_pred_raw>=0.5)
    y_pred = y_pred.astype(int)
    y_test = y_test.to_frame()
    y_test = y_test.rename(columns={'Group_bin': 'ground_truth'})
    y_test['preds'] = y_pred
    y_test['preds_raw'] = y_pred_raw
    return y_test



def compute_metrics(df,fn):
    y_test = df['ground_truth'].values
    y_pred = df['preds'].values
    y_score = df['preds_raw'].values
    print(y_score)
    cm = confusion_matrix(y_test, y_pred)
    cm = cm / np.sum(cm, axis=1, keepdims=True)
    print("confusion_matrix =",cm)
    cm_df = pd.DataFrame(cm,
            index = ['HC','PD'], 
            columns = ['HC','PD'])

    
    ac=accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print("accuracy =",ac)
    tn, fp, fn, tp = cm.ravel()
    sens = tp/(tp+fn)
    spec = tn/(tn+fp)
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    print("metrics =",[ac, sens, spec,roc_auc])
    return [ac, sens, spec, roc_auc,f1]
    

params = {'batch_size': 5,
        'imagex': 160,
        'imagey': 192,
        'imagez': 160
        }


fn_test = args.fn
test = pd.read_csv(fn_test)
IDs_list=test['Subject'].to_numpy()
test_IDs=IDs_list
test_generator=DataGenerator(test_IDs, 1, (
    params['imagex'], params['imagey'], params['imagez']), False, fn_test, 'Group_bin')


from os import listdir
from os.path import isfile, join
mypath_encoder = args.en
mypath_classifier = args.pd


# folder = args.m
name=args.o

results = pd.DataFrame(columns=['cycle','accuracy','sensitivity','specificity','AUC_ROC','F1_SCORE','male_accuracy','male_sensitivity','male_specificity','male AUC_ROC','female_accuracy','female_sensitivity','female_specificity','female AUC_ROC'])
cycle_count=1


# for _, (encoder_file, classifier_file) in enumerate(zip(folder_encoder, folder_classifier)):
#     print(encoder_file)
#     print(classifier_file)
for i in range(10,30):
    encoder = tf.keras.models.load_model(mypath_encoder + "encoder_distributed_BS5_"+str(i)+".h5")
    classifier = tf.keras.models.load_model(mypath_classifier+ "classifier_PD_distributed_BS5_"+str(i)+".h5")
    encoder.trainable = False
    classifier.trainable = False

    #Make predictions and save the results
    y_test = test['Group_bin']
    y_pred_enc=encoder.predict(test_generator)
    y_pred = classifier.predict(y_pred_enc)
    preds = model_eval(y_test, y_pred)

    df = pd.merge(preds, test, left_index=True, right_index=True)
    df.to_csv(name+'_predictions.csv')

    df_male = df.loc[df['Sex']=='M']
    df_female = df.loc[df['Sex']=='F']

    print("For cycle number "+str(cycle_count))
    print("######Overall Metrics######")
    metrics = compute_metrics(df,"cm_agg_"+name)
    print("########Male Metrics#######")
    metrics_male = compute_metrics(df_male,"cm_male_"+name)
    print("######Female Metrics#######")
    metrics_female = compute_metrics(df_female,"cm_female_"+name)



    new_result = {'cycle':cycle_count, 'accuracy':metrics[0], 'sensitivity':metrics[1],'specificity':metrics[2],'AUC_ROC':metrics[3], 'F1_SCORE': metrics[4],
                   'male_accuracy':metrics_male[0],'male_sensitivity':metrics_male[1],'male_specificity':metrics_male[2],'male AUC_ROC':metrics_male[3],
                   'female_accuracy':metrics_female[0],'female_sensitivity':metrics_female[1],'female_specificity':metrics_female[2],'female AUC_ROC':metrics_female[3]}

    results.loc[len(results)] = new_result

    cycle_count += 1


results.to_csv(name+'_metrics.csv')


