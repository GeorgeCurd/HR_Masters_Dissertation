import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split
import keras
from keras.metrics import Precision, Accuracy, Recall
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input, BatchNormalization
from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_curve, PrecisionRecallDisplay
import matplotlib.pyplot as plt
# from Create_Preds_Runner import X_train_norm, X_test_norm, y_train, y_test, X_important_test, X_important_train,\
#      prob_lookup_train, prob_lookup_test, horse_lookup_train, horse_lookup_test, odds_lookup_test, odds_lookup_train
import tensorflow as tf
from matplotlib import pyplot

np.random.seed(1)

def create_rpr_model(X_train, y_train):
    a = len(X_train.columns)
    # Create Model
    model = Sequential()
    model.add(Dense(160, input_dim=a, activation='relu'))
    model.add(Dropout(rate=0.6))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(192, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(BatchNormalization(momentum=0.55))
    model.add(Dense(12, activation='softmax'))

    # Compile Model
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0005), loss='categorical_crossentropy',
              metrics=['accuracy', 'Precision', 'Recall'])

    # fit the model
    hist = model.fit(X_train, y_train, epochs=8, batch_size=50, shuffle=False)
    return model


def evaluate_model(model, X_test, y_test):
    # evaluate the model
    scores = model.evaluate(X_test, y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    print("\n%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))
    print("\n%s: %.2f%%" % (model.metrics_names[3], scores[3]*100))


def create_predictions(model, X_test):
    # Predict
    prediction = model.predict(X_test)
    prediction = pd.DataFrame(prediction)
    return prediction


# def multiclass_roc_auc_score(curve_type, y_test, y_pred, max_horses, average="macro"):
#     target = [i for i in range(1, max_horses+1)]
#     y_pred = np.asarray(y_pred)
#     y_pred = y_pred.argmax(axis=-1)
#     y_test = np.asarray(y_test)
#     y_test = y_test.argmax(axis=-1)
#     lb = LabelBinarizer()
#     lb.fit(y_test)
#     y_test = lb.transform(y_test)
#     y_pred = lb.transform(y_pred)
#
#     for (idx, c_label) in enumerate(target):
#         fpr, tpr, thresholds = curve_type(y_test[:, idx].astype(int), y_pred[:, idx])
#         c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)' % (c_label, auc(fpr, tpr)))
#     c_ax.plot(fpr, fpr, 'b-', label = 'Random Guessing')
#     return roc_auc_score(y_test, y_pred, average=average)
#
#
# # Runner
# hist, network = create_rpr_model(X_important_train, y_train, X_important_test, y_test)
# evaluate_model(network, X_important_test, y_test)
# y_pred = create_predictions(network, X_important_test)

# fig, c_ax = plt.subplots(1, 1, figsize=(12, 8))
# roc_multi = multiclass_roc_auc_score(roc_curve, y_test, y_pred, 12, average="macro")
# c_ax.legend()
# c_ax.set_xlabel('False Positive Rate')
# c_ax.set_ylabel('True Positive Rate')
# plt.show()

# # plot loss
# pyplot.plot(hist0.history['val_accuracy'], label='0 Hidden Layers')
# pyplot.plot(hist1.history['val_accuracy'], label='1 Hidden Layer')
# pyplot.plot(hist2.history['val_accuracy'], label='2 Hidden Layers')
# limits = [0, 200, 0.2, 0.3]
# pyplot.axis(limits)
# pyplot.legend()
# pyplot.xlabel("Epochs")
# pyplot.ylabel(" Validation Loss (Accuracy)")
# pyplot.show()
#
# # Find best loss and Val Loss Results
# loss = hist.history['val_accuracy']
# best_loss = max(loss)
# bl_index = loss.index(best_loss)
# print(best_loss)
# print(bl_index)

# fig, c_ax = plt.subplots(1, 1, figsize=(12, 8))
# roc_multi = multiclass_roc_auc_score(roc_curve, y_test, y_pred, 12, average="macro")
# c_ax.legend()
# c_ax.set_xlabel('False Positive Rate')
# c_ax.set_ylabel('True Positive Rate')
# plt.show()

# horse_lookup_test2 = horse_lookup_test.reset_index(inplace=False, drop=True)
# odds_lookup_test2 = odds_lookup_test.reset_index(inplace=False, drop=True)
# prob_lookup_test2 = prob_lookup_test.reset_index(inplace=False, drop=True)
# y_pred2 = y_pred.reset_index(inplace=False, drop=True)
# y_pred_bin = np.argmax(np.asarray(y_pred2), axis=-1)
# y_test2 = y_test.reset_index(inplace=False, drop=True)
# betting_info = pd.concat([horse_lookup_test2, odds_lookup_test2, prob_lookup_test2, y_pred2, y_test2], axis=1)
# betting_info.to_csv('C:/Users/e1187273/Pictures/Horse Racing Data/betting_info.csv')
# print('Complete')
