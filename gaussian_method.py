import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.utils import class_weight
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold
from scipy.stats import spearmanr

from sklearn import preprocessing

#Gaussian function from the oscillation
def gaussian(x,a,p,w):
    return a*np.exp(-((x-p)**2)/(2*w**2))


def get_timeSeries_trial(Data,k):
    y=np.zeros(k)
    x=np.linspace(0,3,k)

    pkMag= np.array(Data['pkMag'])
    tStart= np.array(Data['tStart'])
    tEnd= np.array(Data['tEnd'])
    tDur= np.array(Data['tDur'])
    for i in range(len(pkMag)):
        y=y+gaussian(x,pkMag[i], tStart[i]+(tEnd[i]-tStart[i])/2, tDur[i])
    return y

def get_timeSeries(Data, electrodes):
    nTrials=len(Data['trial'].unique())
    if Data['tEnd'].max()<1.5:
#Some subjects have 1.5 seconds trials
        timeSeries= np.zeros((nTrials,1500,electrodes.shape[0]))
        k=1500
    else:
        timeSeries= np.zeros((nTrials,3000,electrodes.shape[0]))
        k=3000
    for i in range(electrodes.shape[0]):
        auxData=Data[Data['Electrode name']==electrodes[i]]
        power_sum=auxData.groupby('trial',as_index=False)['Recalled'].mean()
        sums=auxData.groupby('trial',as_index=False)['pkMag'].sum()
        power_sum['pkMag'] = power_sum['trial'].map(sums.set_index('trial')['pkMag'])
        power_sum.fillna(0)

        corr, _ = spearmanr(power_sum['pkMag'], power_sum['Recalled'])
        sign=1
        if corr <0:
            sign=-1 
        for j in range(nTrials):
            timeSeries[j,:,i]= sign*get_timeSeries_trial(auxData[auxData['trial']==j],k)
    return timeSeries
#LSTM model
def make_model(input_shape,units):
    input_layer = keras.layers.Input(input_shape)

    if (len(units)==2):
      x = keras.layers.LSTM( units[0], return_sequences=True)(input_layer)
      x = keras.layers.Dropout(0.2)(x)
      x = keras.layers.LSTM( units[1], return_sequences=False)(x)
      x = keras.layers.Dropout(0.2)(x)
    else:
      x = keras.layers.LSTM( units[0], return_sequences=False)(input_layer)
      x = keras.layers.Dropout(0.2)(x)          
    
    output_layer = keras.layers.Dense(1, activation="sigmoid")(x)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)

def gaussian_method(highgamma,beta,subject,units):
    highGamma= pd.read_csv(highgamma)
    beta= pd.read_csv(beta)
    electrodes= highGamma['Electrode name'].unique()
    y=np.array(highGamma.groupby('trial')['Recalled'].mean())
    timeSeriesB=get_timeSeries(beta,electrodes)
    timeSeriesHG= get_timeSeries(highGamma, electrodes)
    X= np.concatenate([timeSeriesHG,timeSeriesB],axis=2)
    acc_per_fold = []
    auc_per_fold = []
    loss_per_fold = []

    kfold = StratifiedKFold(n_splits=4, shuffle=True)

    fold_no = 1
    for train, test in kfold.split(X, y):

        model = make_model((X.shape[1],X.shape[2]),units)
        epochs = 100
        batch_size = 128
        optimizer = keras.optimizers.Adam(learning_rate=0.01)
        callbacks = [keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=15, min_lr=0.0001    ),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, verbose=1),]
        model.compile(  optimizer=optimizer,
            loss= "binary_crossentropy",
                metrics=[
                keras.metrics.BinaryAccuracy(),
                keras.metrics.AUC(),
                keras.metrics.Precision(),
                keras.metrics.Recall(),
            ],)

        X_train= X[train]
        X_test= X[test]
        y_train= y[train]
        y_test=y[test]

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, shuffle=True)
        scaler = preprocessing.StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)


        class_weights = class_weight.compute_class_weight(
                                                  class_weight = "balanced",
                                                  classes = np.unique(y_train),
                                                  y = y_train
                                              )
        class_weights = dict(zip(np.unique(y_train), class_weights))
        y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
        y_test = np.asarray(y_test).astype('float32').reshape((-1,1))
        y_val= np.asarray(y_val).astype('float32').reshape((-1,1))

        print(f'Fold {fold_no} ...')

        history = model.fit(
              X_train,
              y_train,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=callbacks,
              validation_data=(X_val,y_val),
              verbose=1,
              class_weight=class_weights,
          )

        scores = model.evaluate(X_test, y_test, verbose=0)
        print(scores[2])

        acc_per_fold.append(scores[1])
        auc_per_fold.append(scores[2])
        loss_per_fold.append(scores[0])
        fold_no = fold_no + 1

    f=pd.DataFrame(columns=['subject','acc','auc', 'units'])
    f.loc[len(f.index)]=[subject, np.mean(acc_per_fold), np.mean(auc_per_fold), units]
    return f
