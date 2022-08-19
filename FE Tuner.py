from tensorflow.keras.models import Model
from keras.layers import Dense, Dropout, Input, BatchNormalization, LeakyReLU
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot
import numpy as np
from Create_Preds_Runner import X_test_norm, X_train_norm, X_train_norm_ext, X_test_norm_ext
from keras import Sequential
from keras.regularizers import l1
from tensorflow.keras.models import Model, load_model
from tensorflow import keras
import tensorflow as tf
from kerastuner.tuners import Hyperband

def build_autoencoder_model(hp):
    hp_dropout1 = hp.Float('dropout1', min_value=0.0, max_value=0.9, step=0.1)
    hp_momentum1 = hp.Float('momentum1', min_value=0.5, max_value=1.0, step=0.05)
    hp_units1 = hp.Int('units1', min_value=128, max_value=16384, step=128)
    hp_dropout2 = hp.Float('dropout2', min_value=0.0, max_value=0.9, step=0.1)
    hp_momentum2 = hp.Float('momentum2', min_value=0.5, max_value=1.0, step=0.05)
    hp_units2 = hp.Int('units2', min_value=128, max_value=8192, step=128)
    hp_dropout3 = hp.Float('dropout2', min_value=0.0, max_value=0.9, step=0.1)
    hp_momentum3 = hp.Float('momentum2', min_value=0.5, max_value=1.0, step=0.05)
    hp_units3 = hp.Int('units2', min_value=128, max_value=8192, step=128)
    hp_dropout4 = hp.Float('dropout2', min_value=0.0, max_value=0.9, step=0.1)
    hp_momentum4 = hp.Float('momentum2', min_value=0.5, max_value=1.0, step=0.05)
    hp_units4 = hp.Int('units2', min_value=128, max_value=8192, step=128)
    hp_relu1 = hp.Float('relu1', min_value=0.0, max_value=0.6, step=0.1)
    hp_relu2 = hp.Float('relu2', min_value=0.0, max_value=0.6, step=0.1)
    hp_relu3 = hp.Float('relu3', min_value=0.0, max_value=0.6, step=0.1)
    hp_relu4 = hp.Float('relu4', min_value=0.0, max_value=0.6, step=0.1)

    # define encoder
    n_inputs = X_train_norm_ext.shape[1]
    visible = Input(shape=(n_inputs,))
    # encoder level 1
    e = Dense(units=hp_units1)(visible)
    e = (Dropout(rate=hp_dropout1))(e)
    e = BatchNormalization(momentum=hp_momentum1)(e)
    e = LeakyReLU(alpha=hp_relu1)(e)
    # encoder level 2
    e = Dense(units=hp_units2)(e)
    e = (Dropout(rate=hp_dropout2))(e)
    e = BatchNormalization(momentum=hp_momentum2)(e)
    e = LeakyReLU(alpha=hp_relu2)(e)
    # bottleneck
    n_bottleneck = round(float(n_inputs) / 150.0)

    # n_bottleneck = n_inputs
    bottleneck = Dense(n_bottleneck)(e)

    # define decoder, level 1
    d = Dense(units=hp_units3)(bottleneck)
    d = (Dropout(rate=hp_dropout3))(d)
    d = BatchNormalization(momentum=hp_momentum3)(d)
    d = LeakyReLU(alpha=hp_relu3)(d)
    # define decoder, level 2
    d = Dense(units=hp_units4)(d)
    d = (Dropout(rate=hp_dropout4))(d)
    d = BatchNormalization(momentum=hp_momentum4)(d)
    d = LeakyReLU(alpha=hp_relu4)(d)
    # output layer
    output = Dense(n_inputs, activation='sigmoid')(d)

    # define autoencoder model
    model = Model(inputs=visible, outputs=output)

    # compile autoencoder model
    hp_lr = hp.Choice('learning rate', values=[0.1, 0.01, 0.001, 0.0001, 0.05, 0.005, 0.0005])
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=hp_lr), loss='mse')

    return model

print('step1 complete')

tuner = Hyperband(build_autoencoder_model, objective='val_loss', factor=3, max_epochs=50, overwrite=True)

print('step2 complete')

results = tuner.search(X_train_norm_ext, X_train_norm_ext, validation_data=(X_test_norm_ext, X_test_norm_ext))

print('step3 complete')

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

best_model_count = 20
best = tuner.results_summary(num_trials=best_model_count)

model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train_norm_ext, X_train_norm_ext, epochs=200, validation_data=(X_test_norm_ext, X_test_norm_ext))

val_acc_per_epoch = history.history['val_loss']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))
