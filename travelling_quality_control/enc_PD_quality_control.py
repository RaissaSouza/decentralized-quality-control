from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(1)
import random
random.seed(1)
import tensorflow as tf
import csv
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from numpy import argmax
import pandas as pd
import tensorflow.keras.backend as K
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Activation, Concatenate
from tensorflow.keras.layers import Conv3D, MaxPool3D, Flatten, Dense, ReLU, AveragePooling3D, LeakyReLU, Add
from tensorflow.keras.layers import Dropout, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from datagenerator_pd import DataGenerator
from datagenerator_test import DataGenerator as DataGeneratorTest
from tensorflow.keras.utils import to_categorical
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
#os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import argparse
import sys
import math
import time

LEARNING_RATE = 0.0001

#parse input arguments
#parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-fn_train', type=str, help='training set')
parser.add_argument('-fn_test', type=str, help='test set')
parser.add_argument('-cycles', type=int, help='number of cycles')
parser.add_argument('-en', type=str, help='pretrained encoder')
parser.add_argument('-pd', type=str, help='pretrained PD classifier')
parser.add_argument('-revisit', type=int, help='revisit cycle')
parser.add_argument('-error', type=float, help='acceptable error')
parser.add_argument('-fn_save', type=str, help='name to save')
args = parser.parse_args()



params = {'batch_size': 5,
        'imagex': 160,
        'imagey': 192,
        'imagez': 160
        }


CYCLES = args.cycles
EPOCHS = 1
BATCH_SIZE = 5
ERROR_THRESHOLD = args.error
REVISIT_ROUND = args.revisit
# Define column names
columns = ['Cycle', 'Subject', 'Study', 'ACC', 'FP', 'FN', 'PFP', 'PFN','Ignored']

# Create an empty DataFrame with specified columns
test_df = pd.DataFrame(columns=columns)
revisit={}
banned=[]

# pretrained models
encoder = tf.keras.models.load_model(args.en)
classifier_PD = tf.keras.models.load_model(args.pd)

encoder.trainable = True
classifier_PD.trainable = True

train_loss_pd = tf.keras.losses.BinaryCrossentropy(from_logits=False)
val_loss_pd = tf.keras.losses.BinaryCrossentropy(from_logits=False)
train_acc_pd = tf.keras.metrics.BinaryAccuracy()
val_acc_pd = tf.keras.metrics.BinaryAccuracy()


# loss
def categorical_cross_entropy_label_predictor(y_true, y_pred, batch_size):
    ccelp = tf.keras.losses.BinaryCrossentropy()
    return ccelp(y_true, y_pred)/batch_size


def scheduler(lr):
    return lr * tf.math.exp(-0.1)


fn_train_PD = args.fn_train
train_PD = pd.read_csv(fn_train_PD)
studies =  train_PD['Study'].unique()
np.random.seed(42)  
np.random.shuffle(studies)


def evaluation(y_test, y_pred, test_df,error):
    ignore=0
    y_test=y_test.to_numpy()
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total_pd = np.sum(y_test == 1)
    total_hc = np.sum(y_test == 0)
    
    pfp = fp / total_pd if total_pd != 0 else 0
    pfn = fn / total_hc if total_hc != 0 else 0
    
    if not test_df.empty:
        #print(test_df)
        # Define your condition
        condition = test_df['Ignored'] == 0
        #print(condition)
        # Get the last index where the condition is True
        last_index = test_df[condition].index[-1]
        #print(last_index)
        pfp_before = test_df.loc[last_index, 'PFP']
        pfn_before = test_df.loc[last_index, 'PFN']
    else:
        pfp_before, pfn_before = 0, 0

    #if(pfp > (pfp_before+ERROR_THRESHOLD) or pfn > (pfn_before+ERROR_THRESHOLD)):
    if(pfp > (pfp_before+error) or pfn > (pfn_before+error)):
        ignore = 1
    ac=accuracy_score(y_test, y_pred)
    return ac, fp, fn, pfp, pfn,ignore



def server_evaluation(test_PD,test_generator, test_df,error):
    y_test = test_PD['Group_bin']
    y_pred_encoder=encoder.predict(test_generator)
    y_pred_raw = classifier_PD.predict(y_pred_encoder)
    y_pred = (y_pred_raw>=0.5)
    y_pred = y_pred.astype(int)
    return evaluation(y_test, y_pred, test_df,error)

# train step for PD classifier
@tf.function
def train_step(X, y_PD,batch_size):
    with tf.GradientTape() as tape:
        logits_enc = encoder(X, training=True)
        logits_PD = classifier_PD(logits_enc, training=True)
        y_PD = tf.reshape(y_PD, [batch_size, 1])
        train_loss_PD = train_loss_pd(y_PD, logits_PD)
        train_acc_pd.update_state(y_PD, logits_PD)
       

    # compute gradient 
    grads = tape.gradient(train_loss_PD, [encoder.trainable_weights, classifier_PD.trainable_weights])
    #tf.print(grads)

    # update weights
    encoder.optimizer.apply_gradients(zip(grads[0], encoder.trainable_weights))
    classifier_PD.optimizer.apply_gradients(zip(grads[1], classifier_PD.trainable_weights))

    return train_loss_PD, logits_PD

def filter(banned, revisit, subject, c):
    if len(banned) > 0:
        subject = [x for x in subject if x not in banned]
        print("banned check")
    if len(revisit) > 0:
        print("revisit check")
        for i in range(c,c+REVISIT_ROUND,1):
            if i in revisit:
                print(i)
                subject = [x for x in subject if x not in revisit[i]]
    return subject

def regular_cycle(studies,test_df, banned, revisit, c):
    end_list=[]
    for s in studies:
        print("Time start study training - "+str(time.time()))
        batch_size = BATCH_SIZE
        print("STUDY --> "+str(s))

        train_aux =  train_PD[train_PD['Study']==s]
        IDs_list = train_aux['Subject'].to_numpy()

        print(IDs_list)
        train_IDs = filter(banned, revisit, IDs_list, c)
        print(train_IDs)

        if(len(train_IDs)>0):
            if(len(train_IDs)<batch_size): 
                batch_size=len(train_IDs)
                
            for epoch in range(EPOCHS):
                training_generator_PD = DataGenerator(train_IDs, batch_size, (params['imagex'], params['imagey'], params['imagez']), True, fn_train_PD, 'Group_bin')

                for batch in range (training_generator_PD.__len__()):
                    step_batch = tf.convert_to_tensor(batch, dtype=tf.int64)
                    X, y_PD, batch_ids = training_generator_PD.__getitem__(step_batch)
                    print("HERE:"+str(batch_ids))
                    # Save the initial weights
                    initial_weights_enc = encoder.get_weights()
                    initial_weights_pd = classifier_PD.get_weights()

                    # Save the initial optimizer state
                    initial_optimizer_state_enc = encoder.optimizer.get_weights()
                    initial_optimizer_state_pd = classifier_PD.optimizer.get_weights()
            
                    train_loss_PD, logits_PD= train_step(X, y_PD, batch_size)
                    print('\nBatch '+str(batch+1)+'/'+str(training_generator_PD.__len__()))
                    print("LOSS PD -->", train_loss_PD)
                    for _ in range(tf.size(logits_PD)):
                        print("LOGITS PD -->", logits_PD[_])
                        print("ACTUAL PD -->", y_PD[_])
                    print("SERVER evaluation after forward pass in site: "+str(s)+" batch: "+str(batch))
                    error=batch_size*ERROR_THRESHOLD
                    acc, fpta, fnta, pfpa,pfna,ignore = server_evaluation(test_PD,test_generator, test_df,error)
                    print("Time after server evaluation - "+str(time.time()))
                    # Define the row data as a dictionary
                    new_row_test = {'Cycle':c, 'Subject':'batch', 'Study':s, 'ACC':[acc],'FP':[fpta], 'FN':[fnta], 'PFP':[pfpa],'PFN':[pfna], 'Ignored':ignore}
                    # Create the new DataFrame
                    new_df = pd.DataFrame(new_row_test)
                    # Append the new row to the DataFrame
                    test_df = pd.concat([test_df, new_df],ignore_index=True)
                    print("Time after concat to dataframe - "+str(time.time()))

                    if(ignore==1):
                        #reset model
                        print("Reset model because it will ignore batch")
                        encoder.set_weights(initial_weights_enc)
                        classifier_PD.set_weights(initial_weights_pd)
                        encoder.optimizer.set_weights(initial_optimizer_state_enc)
                        classifier_PD.optimizer.set_weights(initial_optimizer_state_pd)

                        #train bs1
                        print("Batch ID to revisit: "+ str(batch_ids))
                        for sub_id in batch_ids:
                            subject = np.array([sub_id])
                            training_generator_PD_2 = DataGenerator(subject, 1, (params['imagex'], params['imagey'], params['imagez']), False, fn_train_PD, 'Group_bin')
                            for batch in range (training_generator_PD_2.__len__()):
                                step_batch = tf.convert_to_tensor(batch, dtype=tf.int64)
                                X, y_PD, batch_ids = training_generator_PD_2.__getitem__(step_batch)
                                # Save the initial weights
                                initial_weights_enc = encoder.get_weights()
                                initial_weights_pd = classifier_PD.get_weights()

                                # Save the initial optimizer state
                                initial_optimizer_state_enc = encoder.optimizer.get_weights()
                                initial_optimizer_state_pd = classifier_PD.optimizer.get_weights()
                
                                train_loss_PD, logits_PD= train_step(X, y_PD, 1)
                                print('\nBatch '+str(batch+1)+'/'+str(training_generator_PD_2.__len__()))
                                print("LOSS PD -->", train_loss_PD)
                                for _ in range(tf.size(logits_PD)):
                                    print("LOGITS PD -->", logits_PD[_])
                                    print("ACTUAL PD -->", y_PD[_])
                                print("SERVER evaluation after forward pass in site: "+str(s)+" batch: "+str(batch))
                                error=1*ERROR_THRESHOLD
                                acc, fpta, fnta, pfpa,pfna,ignore = server_evaluation(test_PD,test_generator, test_df,error)
                                print("Time after server evaluation - "+str(time.time()))
                                # Define the row data as a dictionary
                                new_row_test = {'Cycle':c, 'Subject':sub_id, 'Study':s, 'ACC':[acc],'FP':[fpta], 'FN':[fnta], 'PFP':[pfpa],'PFN':[pfna], 'Ignored':ignore}
                                # Create the new DataFrame
                                new_df = pd.DataFrame(new_row_test)
                                # Append the new row to the DataFrame
                                test_df = pd.concat([test_df, new_df],ignore_index=True)
                                print("Time after concat to dataframe - "+str(time.time()))
                                if(ignore==1):
                                    print("Add revisit list")
                                    print("Reset model because it will ignore subject")
                                    encoder.set_weights(initial_weights_enc)
                                    classifier_PD.set_weights(initial_weights_pd)
                                    encoder.optimizer.set_weights(initial_optimizer_state_enc)
                                    classifier_PD.optimizer.set_weights(initial_optimizer_state_pd)
                                    end_list.append(sub_id)
    return test_df, end_list

def revisit_cycle(sub_rev, test_PD,test_generator, test_df, banned):
    for s in sub_rev:
        # Save the initial weights
        initial_weights_enc = encoder.get_weights()
        initial_weights_pd = classifier_PD.get_weights()

        # Save the initial optimizer state
        initial_optimizer_state_enc = encoder.optimizer.get_weights()
        initial_optimizer_state_pd = classifier_PD.optimizer.get_weights()

        print("SUBJECT --> "+str(s))
        print("Time before loading subject - "+str(time.time()))

        train_aux = train_PD[train_PD['Subject']==s]
        site = train_aux['Study'].values
        train_IDs = train_aux['Subject'].to_numpy()
        training_generator_PD = DataGenerator(train_IDs, 1, (params['imagex'], params['imagey'], params['imagez']), False, fn_train_PD, 'Group_bin')
        print("Time after loading subject - "+str(time.time()))    
        
        for epoch in range(EPOCHS):
            for batch in range (training_generator_PD.__len__()):
                step_batch = tf.convert_to_tensor(batch, dtype=tf.int64)
                X, y_PD, batch_ids = training_generator_PD.__getitem__(step_batch)
        
                train_loss_PD, logits_PD= train_step(X, y_PD, 1)
                print('\nBatch '+str(batch+1)+'/'+str(training_generator_PD.__len__()))
                print("LOSS PD -->", train_loss_PD)
                for _ in range(tf.size(logits_PD)):
                    print("LOGITS PD -->", logits_PD[_])
                    print("ACTUAL PD -->", y_PD[_])
        print("SERVER evaluation after forward pass in site: "+str(site)+" subject: "+str(s))
        error=1*ERROR_THRESHOLD
        acc, fpta, fnta, pfpa,pfna,ignore = server_evaluation(test_PD,test_generator, test_df,error)
        print("Time after server evaluation - "+str(time.time()))
        # Define the row data as a dictionary
        new_row_test = {'Cycle':c, 'Subject':s, 'Study':site, 'ACC':[acc],'FP':[fpta], 'FN':[fnta], 'PFP':[pfpa],'PFN':[pfna], 'Ignored':ignore}
        # Create the new DataFrame
        new_df = pd.DataFrame(new_row_test)
        # Append the new row to the DataFrame
        test_df = pd.concat([test_df, new_df],ignore_index=True)
        print("Time after concat to dataframe - "+str(time.time()))
        if(ignore==1):
            print("ADD TO BANNED LIST")
            banned.append(s)
            encoder.set_weights(initial_weights_enc)
            classifier_PD.set_weights(initial_weights_pd)
            encoder.optimizer.set_weights(initial_optimizer_state_enc)
            classifier_PD.optimizer.set_weights(initial_optimizer_state_pd)
    return test_df, banned
    

print("Time before loading test set - "+str(time.time()))

fn_test_PD = args.fn_test
test_PD = pd.read_csv(fn_test_PD)
test_IDs=test_PD['Subject'].to_numpy()
test_generator=DataGeneratorTest(test_IDs, 1, (params['imagex'], params['imagey'], params['imagez']), False, fn_test_PD, 'Group_bin')

print("Time after loading test set - "+str(time.time()))

####################################################################################################################

#get baseline evaluation
print("Time before loading compute baseline metrics - "+str(time.time()))
acc, fpt, fnt,pfpt,pfnt,ignore = server_evaluation(test_PD,test_generator, test_df,0.05)
new_row_test = {'Cycle':0, 'Subject':'n/a', 'Study':'n/a', 'ACC':[acc], 'FP':[fpt], 'FN':[fnt], 'PFP':[pfpt],'PFN':[pfnt], 'Ignored':[0]}
# Append the new row to the DataFrame
test_df = pd.concat([test_df, pd.DataFrame(new_row_test)],ignore_index=True)
print("Time after compute baseline metrics - "+str(time.time()))


for c in range(10,CYCLES):
    np.random.seed(42+c)  
    np.random.shuffle(studies)
    print("CYCLE --> "+str(c)+'\n')
    print("Time start cycle - "+str(time.time()))
    ########################
    test_df, rev_list =regular_cycle(studies,test_df, banned, revisit, c)
    revisit[c+REVISIT_ROUND]= rev_list

    #revisit cycle
    if len(revisit) > 0:
        if c in revisit:
            sub_rev = revisit[c]
            test_df, banned = revisit_cycle(sub_rev, test_PD, test_generator, test_df, banned)
    print(banned)

    # Display metrics at the end of each epoch.
    train_acc = train_acc_pd.result()
    print("Training acc over cycle: %.4f" % (float(train_acc),))
   


    ########################

    # learning rate scheduling
    LEARNING_RATE = scheduler(LEARNING_RATE)
    encoder.optimizer.learning_rate.assign(LEARNING_RATE) 
    classifier_PD.optimizer.learning_rate.assign(LEARNING_RATE) 
    print(encoder.optimizer.learning_rate.numpy())
    #optimizer_encoder = Adam(learning_rate=LEARNING_RATE)
    #optimizer_PD = Adam(learning_rate=1*LEARNING_RATE)

    # Reset training metrics and loss at the end of each epoch
    train_acc_pd.reset_states()

    #model save
    if(c<=9 and c>=0): 
        encoder.save('encoder_'+args.fn_save+"_0"+str(c)+".h5") 
        classifier_PD.save('classifier_PD_'+args.fn_save+"_0"+str(c)+".h5")
    else:
        encoder.save('encoder_'+args.fn_save+"_0"+str(c)+".h5") 
        classifier_PD.save('classifier_PD_'+args.fn_save+"_0"+str(c)+".h5")
    test_df.to_csv(args.fn_save+".csv")
    print(revisit)
    print(banned)

####################################################################################################################


