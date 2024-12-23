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
from sklearn.metrics import confusion_matrix
from datagenerator_pd import DataGenerator
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
parser = argparse.ArgumentParser()
parser.add_argument('-fn_train', type=str, help='training set')
parser.add_argument('-cycles', type=int, help='number of cycles')
parser.add_argument('-epochs', type=int, help='number of local epochs per cycle')
parser.add_argument('-batch_size', type=int, help='batch size')
args = parser.parse_args()


params = {'batch_size': args.batch_size,
        'imagex': 160,
        'imagey': 192,
        'imagez': 160
        }


CYCLES = args.cycles
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size


def sfcn(inputLayer):
    #block 1
    x=Conv3D(filters=32, kernel_size=(3, 3, 3),padding='same',name="conv1")(inputLayer[0])
    x=BatchNormalization(name="norm1")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool1")(x)
    x=ReLU()(x)
    #block 2
    x=Conv3D(filters=64, kernel_size=(3, 3, 3),padding='same',name="conv2")(x)
    x=BatchNormalization(name="norm2")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool2")(x)
    x=ReLU()(x)
    #block 3
    x=Conv3D(filters=128, kernel_size=(3, 3, 3),padding='same',name="conv3")(x)
    x=BatchNormalization(name="norm3")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool3")(x)
    x=ReLU()(x)
    #block 4
    x=Conv3D(filters=256, kernel_size=(3, 3, 3),padding='same',name="conv4")(x)
    x=BatchNormalization(name="norm4")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool4")(x)
    x=ReLU()(x)
    #block 5
    x=Conv3D(filters=256, kernel_size=(3, 3, 3),padding='same',name="conv5")(x)
    x=BatchNormalization(name="norm5")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool5")(x)
    x=ReLU()(x)
    #block 6
    x=Conv3D(filters=64, kernel_size=(1, 1, 1),padding='same',name="conv6")(x)
    x=BatchNormalization(name="norm6")(x)
    x=ReLU()(x)
    #block 7
    x=AveragePooling3D()(x)
    x=Dropout(.2)(x)
    x=Flatten(name="flat1")(x)

    return x


optimizer_encoder = Adam(learning_rate=0.0001)
optimizer_PD = Adam(learning_rate=0.0001)
train_loss_pd = tf.keras.losses.BinaryCrossentropy(from_logits=False)
train_acc_metric = tf.keras.metrics.BinaryAccuracy()


# encoder
inputA = Input(shape=(params['imagex'], params['imagey'], params['imagez'], 1), name="InputA")
feature_dense_enc = sfcn([inputA])

encoder = Model(inputs=[inputA], outputs=[feature_dense_enc])
encoder.compile(optimizer=optimizer_encoder)


def add_PD_layer(input_shape):
    inputs_PD = Input(shape=(input_shape))
    output_PD = Dense(1, activation='sigmoid')(inputs_PD)
    return Model(inputs_PD, output_PD)


classifier_PD = add_PD_layer(encoder.output.shape[1])
classifier_PD.compile(optimizer=optimizer_PD)


# loss
def categorical_cross_entropy_label_predictor(y_true, y_pred, batch_size):
    ccelp = tf.keras.losses.BinaryCrossentropy()
    return ccelp(y_true, y_pred)/batch_size


def scheduler(epoch, lr):
    return lr * tf.math.exp(-0.1)


fn_train_PD = args.fn_train
train_PD = pd.read_csv(fn_train_PD)
studies =  train_PD['Study'].unique()
np.random.seed(40)  
np.random.shuffle(studies)


# train step for PD classifier
@tf.function
def train_step(X, y_PD,batch_size):
    with tf.GradientTape() as tape:
        logits_enc = encoder(X, training=True)
        logits_PD = classifier_PD(logits_enc, training=True)
        y_PD = tf.reshape(y_PD, [batch_size, 1])
        train_loss_PD = train_loss_pd(y_PD, logits_PD)
        train_acc_metric.update_state(y_PD, logits_PD)
       

    # compute gradient 
    grads = tape.gradient(train_loss_PD, [encoder.trainable_weights, classifier_PD.trainable_weights])
    #tf.print(grads)

    # update weights
    encoder.optimizer.apply_gradients(zip(grads[0], encoder.trainable_weights))
    classifier_PD.optimizer.apply_gradients(zip(grads[1], classifier_PD.trainable_weights))

    return train_loss_PD, logits_PD

####################################################################################################################

for c in range(CYCLES):
    np.random.seed(40+c)  
    np.random.shuffle(studies)
    print("CYCLE --> "+str(c)+'\n')

    ########################
    for s in studies:
        batch_size = BATCH_SIZE
        print("STUDY --> "+str(s))

        train_aux =  train_PD[train_PD['Study']==s]
        IDs_list = train_aux['Subject'].to_numpy()
        train_IDs = IDs_list
        if(len(train_IDs)<batch_size): 
            batch_size=len(train_IDs)
            
        
        for epoch in range(EPOCHS):
            training_generator_PD = DataGenerator(train_IDs, batch_size, (params['imagex'], params['imagey'], params['imagez']), True, fn_train_PD, 'Group_bin')
            t1 = time.time()


            for batch in range (training_generator_PD.__len__()):
                step_batch = tf.convert_to_tensor(batch, dtype=tf.int64)
                X, y_PD, batch_ids = training_generator_PD.__getitem__(step_batch)
        
                train_loss_PD, logits_PD= train_step(X, y_PD, batch_size)
                print('\nBatch '+str(batch+1)+'/'+str(training_generator_PD.__len__()))
                print("LOSS PD -->", train_loss_PD)
                for _ in range(tf.size(logits_PD)):
                    print("LOGITS PD -->", logits_PD[_])
                    print("ACTUAL PD -->", y_PD[_])

    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print("Training acc over cycle: %.4f" % (float(train_acc),))

    t2 = time.time()
    template = 'TRAINING disease - ETA: {} - cycle: {}\n'
    print(template.format(round((t2-t1)/60, 4), c+1))


    ########################

    # learning rate scheduling
    LEARNING_RATE = scheduler(epoch, LEARNING_RATE)
    encoder.optimizer.learning_rate.assign(LEARNING_RATE) 
    classifier_PD.optimizer.learning_rate.assign(LEARNING_RATE) 
    #optimizer_encoder = Adam(learning_rate=LEARNING_RATE)
    #optimizer_PD = Adam(learning_rate=1*LEARNING_RATE)

    # Reset training metrics and loss at the end of each epoch
    train_acc_metric.reset_states()

    #model save
    if(c<=9 and c>=0): 
        encoder.save('encoder_distributed_BS'+str(BATCH_SIZE)+"_0"+str(c)+".h5") 
        classifier_PD.save('classifier_PD_distributed_BS'+str(BATCH_SIZE)+"_0"+str(c)+".h5")
    else:
        encoder.save('encoder_distributed_BS'+str(BATCH_SIZE)+"_"+str(c)+".h5")
        classifier_PD.save('classifier_PD_distributed_BS'+str(BATCH_SIZE)+"_"+str(c)+".h5")

####################################################################################################################


