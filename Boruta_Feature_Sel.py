import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from Create_Preds_Runner import y_train, y_test


# Set params
random_state =1
n_estimators = 50
depth = 5


def create_and_test_RF_model(X_train, X_test, y_train, y_test, RF_rand_state, n_ests, depth):
    # Convert y multiclass into one column
    y_train = np.asarray(y_train).argmax(axis=-1)
    y_test = np.asarray(y_test).argmax(axis=-1)
    # Create an RF model for testing various feature combinations
    rf = RandomForestClassifier(random_state=RF_rand_state, n_estimators=n_ests, max_depth=depth)
    rf.fit(X_train, y_train)
    acc = accuracy_score(y_test, rf.predict(X_test))
    prec = precision_score(y_test, rf.predict(X_test), average='macro', zero_division=1)
    rec = recall_score(y_test, rf.predict(X_test), average='macro')
    print('accuracy:' + str(acc), 'precision:' + str(prec), 'recall:' + str(rec))


def create_boruta_selector_and_rank(X_train, y_train, RF_rand_state, n_ests, depth, perc):
    # Create BorutaPY object using RF and rank the features
    rfc = RandomForestClassifier(random_state=RF_rand_state, n_estimators=n_ests, max_depth=depth)
    boruta_selector = BorutaPy(rfc, n_estimators='auto', verbose=2, random_state=RF_rand_state, perc=perc)
    boruta_selector.fit(np.array(X_train), np.array(y_train))
    # Create table and show feature ranking
    selected_rf_features = pd.DataFrame({'Feature': list(X_train.columns),
                                         'Ranking': boruta_selector.ranking_})
    selected_rf_features.sort_values(by='Ranking')
    return boruta_selector, selected_rf_features


def return_selected_features(selector, X_train, X_test):
    # Return selected features as dataframe
    support = selector.support_
    features = [column for column in X_train.columns[support]]
    features = pd.Index(features)
    X_important_train = selector.transform(np.array(X_train))
    X_important_test = selector.transform(np.array(X_test))
    X_important_train  = pd.DataFrame(X_important_train, index=None, columns=features)
    X_important_test = pd.DataFrame(X_important_test, index=None, columns=features)
    return X_important_train, X_important_test


# Import X Data
filename = 'C:/Users/e1187273/Pictures/Horse Racing Data/X_train_merge.csv'
X_train_merge = pd.read_csv(filename)
filename = 'C:/Users/e1187273/Pictures/Horse Racing Data/X_test_merge.csv'
X_test_merge = pd.read_csv(filename)
print('complete import')

# Run BFS and Return Results

# sel, feats = create_boruta_selector_and_rank(X_train_merge, y_train, random_state, n_ests=n_estimators, depth=depth, perc=50)
# X_important_train, X_important_test = return_selected_features(sel, X_train_merge, X_test_merge)

# X_important_train.to_csv('C:/Users/e1187273/Pictures/Horse Racing Data/X_important_train.csv')
# X_important_test.to_csv('C:/Users/e1187273/Pictures/Horse Racing Data/X_important_test.csv')
# print('Complete')

# Read in Feature Selection Dataframes
filename = 'C:/Users/e1187273/Pictures/Horse Racing Data/X_important_train.csv'
X_important_train = pd.read_csv(filename)
X_important_train.drop(X_important_train.columns[0], axis=1, inplace=True)

filename = 'C:/Users/e1187273/Pictures/Horse Racing Data/X_important_test.csv'
X_important_test = pd.read_csv(filename)
X_important_test.drop(X_important_test.columns[0], axis=1, inplace=True)

create_and_test_RF_model(X_train_merge, X_test_merge, y_train, y_test, random_state, n_estimators,depth)
create_and_test_RF_model(X_important_train, X_important_test, y_train, y_test, random_state, n_estimators,depth)
