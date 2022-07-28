import pandas as pd
import numpy as np
import Data_Proc_Preds as dp
import row_per_race_preds as rpr
from tensorflow.keras.models import Model, load_model
from Data_Proc_Preds import transform_data_preds

# Set Params
max_horses = 12

# Initial Transform
filename = 'C:/Users/e1187273/Pictures/Horse Racing Data/Epsom_Preds.csv'
hr_data = pd.read_csv(filename)
hr_data = hr_data.loc[0:43,:]
hr_data = dp.transform_data_preds(hr_data)
# Use this when testing with Extract
# = hr_data.iloc[0:100, :]

# Row Per Race Transform
a = rpr.RowPerRaceTransformer(hr_data, max_horses)
print('started')
b = a.transform_full_dataframe()
print('completed')
b.to_csv('C:/Users/e1187273/Pictures/Horse Racing Data/Epsom_RPR.csv')
print('Exported')


# Post RPR Transform
filename = 'C:/Users/e1187273/Pictures/Horse Racing Data/Epsom_RPR.csv'
data = pd.read_csv(filename)
data.fillna(0, inplace=True)
data = dp.order_by_date(data)
horse_lookup_pred, X_pred = dp.extract_horse_cols(data, max_horses)
odds_lookup_pred = dp.rpr_extract_odds_cols(data, max_horses)
prob_lookup_pred = dp.rpr_extract_prob_cols(data, max_horses)
X_pred_norm = dp.rpr_normalise(data.iloc[:, 3:])
X_pred_names = data.iloc[:, 3:].columns
X_pred_norm.columns = X_pred_names

# Merge Auto-encoded Cols with Existing Columns
encoder = load_model('encoder.h5')
X_pred_encode = encoder.predict(X_pred_norm)
X_pred_merge = pd.DataFrame(np.column_stack([X_pred_norm, X_pred_encode]))
# Create column names
X_encode_names = pd.Index(['AE' + str(i) for i in range(X_pred_encode.shape[1])])
X_merge_names = X_pred_names.union(X_encode_names, sort=False)
X_pred_merge.columns = X_merge_names
X_pred_merge.to_csv('C:/Users/e1187273/Pictures/Horse Racing Data/X_pred_merge.csv')
print('completed')


