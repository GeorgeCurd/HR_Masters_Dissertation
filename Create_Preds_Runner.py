import pandas as pd
import numpy as np
import Data_Processing as dp
import row_per_race as rpr
from tensorflow.keras.models import Model, load_model

# Set Params
max_horses = 12

# # Initial Transform
# filename = 'C:/Users/e1187273/Pictures/Horse Racing Data/HR_DATA_COMB_SHUFFLE.csv'
# hr_data = pd.read_csv(filename)
# hr_data = dp.transform_data(hr_data)
# # Use this when testing with Extract
# # = hr_data.iloc[0:100, :]
#
# # Row Per Race Transform
# a = rpr.RowPerRaceTransformer(hr_data, max_horses)
# print('started')
# b = a.transform_full_dataframe()
# print('completed')
# b.to_csv('C:/Users/e1187273/Pictures/Horse Racing Data/race_to_row.csv')
# print('Exported')


# Post RPR Transform
filename = 'C:/Users/e1187273/Pictures/Horse Racing Data/race_to_row.csv'
data = pd.read_csv(filename)
data.fillna(0, inplace=True)
data = dp.order_by_date(data)
X_train, X_test, y_train, y_test = dp.train_test_split_inorder(data, max_horses)
horse_lookup_train, X_train = dp.extract_horse_cols(X_train, max_horses)
horse_lookup_test, X_test = dp.extract_horse_cols(X_test, max_horses)
odds_lookup_train = dp.rpr_extract_odds_cols(X_train, max_horses)
odds_lookup_test = dp.rpr_extract_odds_cols(X_test, max_horses)
X_train_norm = dp.rpr_normalise(X_train.iloc[:, 3:])
X_test_norm = dp.rpr_normalise(X_test.iloc[:, 3:])
X_norm_names = X_test.iloc[:, 3:].columns
X_train_norm.columns = X_norm_names
X_test_norm.columns = X_norm_names

# # Take Extract for Autoencoder
# a = int(len(X_train_norm.index)/10)
# X_train_norm_ext = X_train_norm.iloc[0:a, :]
# b = int(len(X_test_norm.index)/10)
# X_test_norm_ext = X_test_norm.iloc[0:b, :]
# X_train_norm.isnull().sum().sum()


# # Merge Auto-encoded Cols with Existing Columns
# encoder = load_model('encoder.h5')
# X_train_encode = encoder.predict(X_train_norm)
# X_test_encode = encoder.predict(X_test_norm)
# X_train_merge = pd.DataFrame(np.column_stack([X_train_norm, X_train_encode]))
# X_test_merge = pd.DataFrame(np.column_stack([X_test_norm, X_test_encode]))
# # Create column names
# X_encode_names = pd.Index(['AE' + str(i) for i in range(X_test_encode.shape[1])])
# X_merge_names = X_norm_names.union(X_encode_names, sort=False)
# X_train_merge.columns = X_merge_names
# X_test_merge.columns = X_merge_names
# X_train_merge.to_csv('C:/Users/e1187273/Pictures/Horse Racing Data/X_train_merge.csv')
# X_test_merge.to_csv('C:/Users/e1187273/Pictures/Horse Racing Data/X_test_merge.csv')
# print('completed')

# Read in Feature Selection Dataframes
filename = 'C:/Users/e1187273/Pictures/Horse Racing Data/X_important_train.csv'
X_important_train = pd.read_csv(filename)
X_important_train.drop(X_important_train.columns[0], axis=1,inplace=True)

filename = 'C:/Users/e1187273/Pictures/Horse Racing Data/X_important_test.csv'
X_important_test = pd.read_csv(filename)
X_important_test.drop(X_important_test.columns[0], axis=1,inplace=True)
