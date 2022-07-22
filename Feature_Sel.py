import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel, f_classif, RFE, SelectKBest
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from Create_Preds_Runner import X_train_merge, X_test_merge, y_train, y_test

features_etc = 200

#Identify key features using random forest/extra trees
model = ExtraTreesClassifier(random_state=1)
model.fit(X_train_merge, y_train)
reduced = SelectFromModel(model, prefit=True, threshold=-np.inf, max_features=features_etc)
train_normX_ETC_FS = reduced.transform(X_train_merge)
test_normX_ETC_FS = reduced.transform(X_test_merge)
