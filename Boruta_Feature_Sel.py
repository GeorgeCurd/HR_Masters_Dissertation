import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from Create_Preds_Runner import y_train, y_test

# Import X Data
filename = 'C:/Users/e1187273/Pictures/Horse Racing Data/X_train_merge.csv'
X_train_merge = pd.read_csv(filename)
filename = 'C:/Users/e1187273/Pictures/Horse Racing Data/X_test_merge.csv'
X_test_merge = pd.read_csv(filename)
y_train_act = np.asarray(y_train).argmax(axis=-1)
y_test_act = np.asarray(y_test).argmax(axis=-1)
print('complete import')

# Create a baseline model using all features
rf_all_features = RandomForestClassifier(random_state=1, n_estimators=1, max_depth=5)
rf_all_features.fit(X_train_merge, y_train_act)
acc = accuracy_score(y_test_act, rf_all_features.predict(X_test_merge))
prec = precision_score(y_test_act, rf_all_features.predict(X_test_merge), average='macro', zero_division=1)
rec = recall_score(y_test_act, rf_all_features.predict(X_test_merge),average='macro')
print(acc, prec, rec)

# # Create BorutaPY object using RF and rank the features
# rfc = RandomForestClassifier(random_state=1, n_estimators=1, max_depth=5)
# boruta_selector = BorutaPy(rfc, n_estimators='auto', verbose=2, random_state=1)
# boruta_selector.fit(np.array(X_train_merge), np.array(y_train_act))
# print("Ranking: ", boruta_selector.ranking_)
# print("No. of significant features: ", boruta_selector.n_features_)
#
# # Create table and show feature ranking
# selected_rf_features = pd.DataFrame({'Feature': list(X_train_merge.columns),
#                                      'Ranking': boruta_selector.ranking_})
# selected_rf_features.sort_values(by='Ranking')
