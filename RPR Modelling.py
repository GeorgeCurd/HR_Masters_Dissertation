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


def extract_horse_cols(df, max_n_horses):
    df_new = pd.DataFrame()
    for i in range(1, max_n_horses + 1):
        last_column = df.pop('Horse_' + str(i))
        a = len(df_new.columns)
        df_new.insert(a, 'Horse_' + str(i), last_column)
    return df_new, df

def extract_odds_cols(df, max_n_horses):
    df_new = pd.DataFrame()
    df2 = df.copy()
    for i in range(1, max_n_horses + 1):
        last_column = df2.pop('ValueOdds_BetfairFormat_' + str(i))
        a = len(df_new.columns)
        df_new.insert(a, 'ValueOdds_BetfairFormat_' + str(i), last_column)
    return df_new


def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    for (idx, c_label) in enumerate(target):
        fpr, tpr, thresholds = roc_curve(y_test[:,idx].astype(int), y_pred[:,idx])
        c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
    c_ax.plot(fpr, fpr, 'b-', label = 'Random Guessing')
    return roc_auc_score(y_test, y_pred, average=average)

filename = 'C:/Users/e1187273/Pictures/Horse Racing Data/race_to_row.csv'
data = pd.read_csv(filename)
horse_lookup, data = extract_horse_cols(data, 14)
X = data[data.columns[:-14]]
X_names = X.columns
# ss = StandardScaler()
# X = pd.DataFrame(ss.fit_transform(X))
y = data[data.columns[-14:]]
y_names = y.columns

# Test Train Split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)
X_train.set_axis(X_names, axis=1, inplace=True)
X_test.set_axis(X_names, axis=1, inplace=True)
y_train.set_axis(y_names, axis=1, inplace=True)
y_train.set_axis(y_names, axis=1, inplace=True)


a = len(X.columns)
# Create Model
model = Sequential()
model.add(Dense(1024, input_dim=a, activation='relu'))
model.add(Dropout(rate=0.4))
model.add(BatchNormalization(momentum=0.9))
model.add(Dense(128, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(14, activation='softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy', 'Precision'])
print(model.summary())


# fit the model
hist = model.fit(X_train, y_train, epochs=10, batch_size=50, shuffle=False)

# evaluate the model
scores = model.evaluate(X_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("\n%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))

# Predict
a = extract_odds_cols(X_test, 14)
prediction = model.predict(X_test)
prediction = pd.DataFrame(prediction)
y_pred = prediction.iloc[:,0]

y_acts = y_test.iloc[:,0]
# y_pred = prediction.argmax(axis=-1)
# y_acts = np.asarray(y_test)
# y_acts = y_acts.argmax(axis=-1)


# PR Curve
fpr_keras, tpr_keras, thresholds_keras = precision_recall_curve(y_acts, y_pred,pos_label=1)
#auc_keras = auc(fpr_keras, tpr_keras)

#Plot ROC Curves
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(tpr_keras, fpr_keras)
# , label='Keras (area = {:.3f})'.format(auc_keras)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


