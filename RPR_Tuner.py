from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input, BatchNormalization
from Create_Preds_Runner import X_train_norm, X_test_norm, y_train, y_test, X_important_test, X_important_train
from kerastuner.tuners import Hyperband
from tensorflow import keras
import tensorflow as tf

def build_model(hp):

    # Create Model
    hp_dropout = hp.Float('dropout', min_value=0.0, max_value=0.9, step=0.1)
    hp_momentum = hp.Float('momentum', min_value=0.5, max_value=1.0, step=0.05)
    hp_units = hp.Int('units', min_value=32, max_value=4096, step=32)
    model = Sequential()
    model.add(Dense(units=hp_units, input_dim=1351, activation='relu'))
    model.add(Dropout(rate=hp_dropout))
    model.add(BatchNormalization(momentum=hp_momentum))
    model.add(Dense(units=hp_units, activation='relu'))
    model.add(Dropout(rate=hp_dropout))
    model.add(Dense(12, activation='softmax'))

    # Compile Model
    hp_lr = hp.Choice('learning rate', values=[0.1, 0.01, 0.001, 0.0001, 0.05, 0.005, 0.0005])
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=hp_lr), loss='categorical_crossentropy',
              metrics=['accuracy', 'Precision', 'Recall'])

    return model

print('step1 complete')

tuner = Hyperband(build_model, objective='val_accuracy', factor=2, max_epochs=50)

print('step2 complete')

tuner.search(X_important_train, y_train, validation_data=(X_important_test, y_test))

print('step3 complete')
