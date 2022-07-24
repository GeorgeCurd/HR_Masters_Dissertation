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
from Create_Preds_Runner import X_train_norm, X_test_norm, y_train, y_test, X_important_test, X_important_train,\
    prob_lookup_train, prob_lookup_test, horse_lookup_train, horse_lookup_test, odds_lookup_test, odds_lookup_train


def create_rpr_model(X_train, y_train):
    a = len(X_train.columns)

    # Create Model
    model = Sequential()
    model.add(Dense(1024, input_dim=a, activation='relu'))
    model.add(Dropout(rate=0.4))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(12, activation='softmax'))

    # Compile Model
    model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy', 'Precision','Recall'])

    # fit the model
    hist = model.fit(X_train, y_train, epochs=25, batch_size=50, shuffle=False)
    return hist, model


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


def multiclass_roc_auc_score(curve_type, y_test, y_pred, max_horses, average="macro"):
    target = [i for i in range(1, max_horses+1)]
    y_pred = np.asarray(y_pred)
    y_pred = y_pred.argmax(axis=-1)
    y_test = np.asarray(y_test)
    y_test = y_test.argmax(axis=-1)
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    for (idx, c_label) in enumerate(target):
        fpr, tpr, thresholds = curve_type(y_test[:, idx].astype(int), y_pred[:, idx])
        c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)' % (c_label, auc(fpr, tpr)))
    c_ax.plot(fpr, fpr, 'b-', label = 'Random Guessing')
    return roc_auc_score(y_test, y_pred, average=average)


# # Runner
# hist, network = create_rpr_model(X_train_norm, y_train)
# evaluate_model(network, X_test_norm, y_test)
# y_pred = create_predictions(network, X_test_norm)

# fig, c_ax = plt.subplots(1, 1, figsize=(12, 8))
# roc_multi = multiclass_roc_auc_score(roc_curve, y_test, y_pred, 12, average="macro")
# c_ax.legend()
# c_ax.set_xlabel('False Positive Rate')
# c_ax.set_ylabel('True Positive Rate')
# plt.show()


hist, network = create_rpr_model(X_important_train, y_train)
evaluate_model(network, X_important_test, y_test)
y_pred = create_predictions(network, X_important_test)

# fig, c_ax = plt.subplots(1, 1, figsize=(12, 8))
# roc_multi = multiclass_roc_auc_score(roc_curve, y_test, y_pred, 12, average="macro")
# c_ax.legend()
# c_ax.set_xlabel('False Positive Rate')
# c_ax.set_ylabel('True Positive Rate')
# plt.show()

horse_lookup_test.reset_index(inplace=True)
odds_lookup_test.reset_index(inplace=True)
prob_lookup_test.reset_index(inplace=True)
y_pred.reset_index(inplace=True)
y_test.reset_index(inplace=True)
betting_info = pd.concat([horse_lookup_test, odds_lookup_test, prob_lookup_test, y_pred, y_test], axis=1)
betting_info.to_csv('C:/Users/e1187273/Pictures/Horse Racing Data/betting_info.csv')
