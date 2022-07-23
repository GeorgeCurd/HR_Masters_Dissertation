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
rf_all_features = RandomForestClassifier(random_state=1, n_estimators=100, max_depth=5)
rf_all_features.fit(X_train_merge, y_train_act)
acc = accuracy_score(y_test_act, rf_all_features.predict(X_test_merge))
prec = precision_score(y_test_act, rf_all_features.predict(X_test_merge), average='macro', zero_division=1)
rec = recall_score(y_test_act, rf_all_features.predict(X_test_merge),average='macro')
print(acc, prec, rec)

# Create BorutaPY object using RF and rank the features
rfc = RandomForestClassifier(random_state=1, n_estimators=50, max_depth=5)
boruta_selector = BorutaPy(rfc, n_estimators='auto', verbose=2, random_state=1, perc=100)
boruta_selector.fit(np.array(X_train_merge), np.array(y_train_act))
print("Ranking: ", boruta_selector.ranking_)
print("No. of significant features: ", boruta_selector.n_features_)
support = boruta_selector.support_
features = [column for column in X_train_merge.columns[support]]
features = pd.Index(features)
#
# Create table and show feature ranking
selected_rf_features = pd.DataFrame({'Feature': list(X_train_merge.columns),
                                     'Ranking': boruta_selector.ranking_})
selected_rf_features.sort_values(by='Ranking')
# selected_rf_features.to_csv('C:/Users/e1187273/Pictures/Horse Racing Data/FS_cols_perc100.csv')

# Select important features
X_important_train = boruta_selector.transform(np.array(X_train_merge))
X_important_test = boruta_selector.transform(np.array(X_test_merge))

# Run RF model with selected features for comparison
rf_boruta = RandomForestClassifier(random_state=1, n_estimators=100, max_depth=5)
rf_boruta.fit(X_important_train, y_train_act)


acc_FS = accuracy_score(y_test_act, rf_boruta.predict(X_important_test))
prec_FS = precision_score(y_test_act, rf_boruta.predict(X_important_test), average='macro', zero_division=1)
rec_FS = recall_score(y_test_act, rf_boruta.predict(X_important_test),average='macro')
print(acc_FS, prec_FS, rec_FS)


pd.DataFrame(X_important_train, index=None, columns=features).to_csv('C:/Users/e1187273/Pictures/Horse Racing Data/X_important_train.csv')
pd.DataFrame(X_important_test, index=None, columns=features).to_csv('C:/Users/e1187273/Pictures/Horse Racing Data/X_important_test.csv')
print('Complete')
