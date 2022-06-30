import pandas as pd
import numpy as np
import naming as nm


def check_missing_vals(df):
    missing = df.isna().sum()
    return missing


def transform_data(df):
    # Drop cols that aren't to be used for modelling
    df.drop(nm.cols_to_delete, axis=1,inplace=True)

    # Update Missing Vals
    for i in range(0, len(nm.mv_cols)):
        df[nm.mv_cols[i]].replace([np.nan], nm.mv_update_vals[i], inplace=True)

    # Encode categorical (nominal) cols as OHE variables
    df= pd.get_dummies(df,prefix=nm.nominal_cols, columns=nm.nominal_cols, drop_first=False,dtype='int64')

    # Update string values for ordinal cols (e.g. cols for previous race positions)
    for i in range(0, len(nm.ordinal_cols)):
        df.replace({nm.ordinal_cols[i]: nm.ordinal_vals}, inplace=True)

    # Create binary results column
    df["result_bin"] = np.where(df["Result"] == '1', 1, 0)
    df.drop(columns=["Result"], inplace=True)

    # Convert ordinal categorical cols into numeric
    for i in range(0, len(nm.ordinal_cols)):
        df[nm.ordinal_cols[i]] = df[nm.ordinal_cols[i]] .astype('int64')

    return df


filename = 'C:/Users/e1187273/Pictures/Horse Racing Data/HR_DATA_COMB2.csv'
hr_data = pd.read_csv(filename)
hr_data = transform_data(hr_data)
# print(hr_data['Position5RunsAgo'].unique())
# dtype = hr_data.dtypes
# dtype.to_csv('C:/Users/e1187273/Pictures/Horse Racing Data/dtypes.csv')

