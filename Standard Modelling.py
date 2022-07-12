import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import keras
from keras.metrics import Precision, Accuracy, Recall, AUC
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input, BatchNormalization
from Data_Processing import hr_data, hr_horse_data
from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_curve, PrecisionRecallDisplay
import matplotlib.pyplot as plt


X = hr_data[hr_data.columns[:-1]]
horse_lookup = X.pop('Horse')
X_names = X.columns
# ss = StandardScaler()
# X = pd.DataFrame(ss.fit_transform(X))
y = hr_data[hr_data.columns[-1:]]
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
model.add(Dense(1, activation='sigmoid'))


model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy', 'Precision', 'Recall', AUC(num_thresholds=200, curve="ROC")])

print(model.summary())


# fit the model
hist = model.fit(X_train, y_train, epochs=10, batch_size=50, shuffle=False)

# evaluate the model
scores = model.evaluate(X_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("\n%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))
print("\n%s: %.2f%%" % (model.metrics_names[3], scores[3]*100))
print("\n%s: %.2f%%" % (model.metrics_names[4], scores[4]*100))


# Predict Test
y_pred_keras = model.predict(X_test)

# ROC Curve
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras,pos_label=1)
auc_keras = auc(fpr_keras, tpr_keras)

#Plot ROC Curves
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

# PR Curve
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_keras,pos_label=1)
precision=precision[0:-1]
recall=recall[0:-1]
auc_pr = auc(recall, precision)

#Plot PR Curves
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR curve')
plt.legend(loc='best')
plt.show()

