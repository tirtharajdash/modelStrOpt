from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import roc_auc_score
from sklearn.datasets import load_iris
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adadelta, Adam, rmsprop
import sys
import numpy as np


iris = load_iris()

X = iris.data
y = iris.target

seed = 123
val_size = int( 0.2 * len( y ) )
np.random.seed( seed )

indices = np.random.permutation(len(X))
X_train = X[ indices[:-val_size]]
y_train = y[ indices[:-val_size]]
X_val = X[ indices[-val_size:]]
y_val = y[ indices[-val_size:]]


space = {'choice': hp.choice('num_layers',
                    [ {'layers':'two', },
                    {'layers':'three',
                    'units3': hp.uniform('units3', 64,1024), 
                    'dropout3': hp.uniform('dropout3', .25,.75)}
                    ]),

            'units1': hp.uniform('units1', 64,1024),
            'units2': hp.uniform('units2', 64,1024),

            'dropout1': hp.uniform('dropout1', .25,.75),
            'dropout2': hp.uniform('dropout2',  .25,.75),

            'batch_size' : hp.uniform('batch_size', 28,128),

            'nb_epochs' :  100,
            'optimizer': hp.choice('optimizer',['adadelta','adam','rmsprop']),
            'activation': 'relu'
        }

def f_nn(params):   
    print ('Params valing: ', params)
    model = Sequential()
    model.add(Dense(output_dim=params['units1'], input_dim = X.shape[1])) 
    model.add(Activation(params['activation']))
    model.add(Dropout(params['dropout1']))

    model.add(Dense(output_dim=params['units2'], init = "glorot_uniform")) 
    model.add(Activation(params['activation']))
    model.add(Dropout(params['dropout2']))

    if params['choice']['layers']== 'three':
        model.add(Dense(output_dim=params['choice']['units3'], init = "glorot_uniform")) 
        model.add(Activation(params['activation']))
        model.add(Dropout(params['choice']['dropout3']))    

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=params['optimizer'])

    model.fit(X_train, y_train, nb_epoch=params['nb_epochs'], batch_size=params['batch_size'], verbose = 0)

    pred_auc =model.predict_proba(X_val, batch_size = 128, verbose = 0)
    acc = roc_auc_score(y_val, pred_auc)
    print('AUC:', acc)
    sys.stdout.flush() 
    return {'loss': -acc, 'status': STATUS_OK}


trials = Trials()
best = fmin(f_nn, space, algo=tpe.suggest, max_evals=50, trials=trials)
print('best: ')
print(best)
