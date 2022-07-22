import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Create_Preds_Runner import X_train_merge, X_test_merge, y_train, y_test

# Create a baseline model using all features
rf_all_features = RandomForestClassifier(random_state=1, n_estimators=1000, max_depth=5)
rf_all_features.fit(X_train_merge, y_train)
accuracy_score(y_test, rf_all_features.predict(X_test_merge))

# Create BorutaPY object using RF and rank the features
rfc = RandomForestClassifier(random_state=1, n_estimators=1000, max_depth=5)
boruta_selector = BorutaPy(rfc, n_estimators='auto', verbose=2, random_state=1)
boruta_selector.fit(np.array(X_train_merge), np.array(y_train))
print("Ranking: ", boruta_selector.ranking_)
print("No. of significant features: ", boruta_selector.n_features_)

# Create table and show feature ranking
selected_rf_features = pd.DataFrame({'Feature': list(X_train_merge.columns),
                                     'Ranking': boruta_selector.ranking_})
selected_rf_features.sort_values(by='Ranking')
