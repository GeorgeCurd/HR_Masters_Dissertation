from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input, BatchNormalization
from Create_Preds_Runner import X_train_norm, X_test_norm, y_train, y_test, X_important_test, X_important_train
from kerastuner.tuners import Hyperband
from tensorflow import keras
import tensorflow as tf

def build_model(hp):

    # Create Model
    hp_dropout1 = hp.Float('dropout1', min_value=0.0, max_value=0.9, step=0.1)
    hp_momentum1 = hp.Float('momentum1', min_value=0.5, max_value=1.0, step=0.05)
    hp_units1 = hp.Int('units1', min_value=32, max_value=2048, step=32)
    hp_dropout2 = hp.Float('dropout2', min_value=0.0, max_value=0.9, step=0.1)
    hp_momentum2 = hp.Float('momentum2', min_value=0.5, max_value=1.0, step=0.05)
    hp_units2 = hp.Int('units2', min_value=32, max_value=2048, step=32)
    model = Sequential()
    model.add(Dense(units=hp_units1, input_dim=1351, activation='relu'))
    model.add(Dropout(rate=hp_dropout1))
    model.add(BatchNormalization(momentum=hp_momentum1))
    model.add(Dense(units=hp_units2, activation='relu'))
    model.add(Dropout(rate=hp_dropout2))
    model.add(BatchNormalization(momentum=hp_momentum2))
    model.add(Dense(12, activation='softmax'))

    # Compile Model
    hp_lr = hp.Choice('learning rate', values=[0.1, 0.01, 0.001, 0.0001, 0.05, 0.005, 0.0005])
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=hp_lr), loss='categorical_crossentropy',
              metrics=['accuracy', 'Precision', 'Recall'])

    return model

print('step1 complete')

tuner = Hyperband(build_model, objective='val_accuracy', factor=3, max_epochs=50, overwrite=True)

print('step2 complete')

results = tuner.search(X_important_train, y_train, validation_data=(X_important_test, y_test))

print('step3 complete')

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

best_model_count = 10
best = tuner.results_summary(num_trials=best_model_count)

model = tuner.hypermodel.build(best_hps)
history = model.fit(X_important_train, y_train, epochs=200, validation_data=(X_important_test, y_test))

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

