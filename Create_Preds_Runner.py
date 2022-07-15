import pandas as pd
import numpy as np
import Data_Processing as dp
import row_per_race as rpr

# Set Params
max_horses = 12

# Initial Transform
filename = 'C:/Users/e1187273/Pictures/Horse Racing Data/HR_DATA_COMB_SHUFFLE.csv'
hr_data = pd.read_csv(filename)
hr_data = dp.transform_data(hr_data)
# Use this when testing with Extract
# = hr_data.iloc[0:100, :]

# Row Per Race Transform
a = rpr.RowPerRaceTransformer(hr_data, max_horses)
print('started')
b = a.transform_full_dataframe()
print('completed')
b.to_csv('C:/Users/e1187273/Pictures/Horse Racing Data/race_to_row.csv')
print('Exported')

# Post RPR Transform
filename = 'C:/Users/e1187273/Pictures/Horse Racing Data/race_to_row.csv'
data = pd.read_csv(filename)
data = dp.order_by_date(data)
col_check = data.dtypes
X_train, X_test, y_train, y_test = dp.test_train_split_inorder(data, max_horses)
horse_lookup_train, X_train = dp.extract_horse_cols(X_train, max_horses)
horse_lookup_test, X_test = dp.extract_horse_cols(X_test, max_horses)

