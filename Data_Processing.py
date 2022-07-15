import pandas as pd
import numpy as np
import naming as nm
from sklearn.preprocessing import StandardScaler


def check_missing_vals(df):
    missing = df.isna().sum()
    return missing


def transform_data(df):
    # Drop cols that aren't to be used for modelling
    df.drop(nm.cols_to_delete, axis=1, inplace=True)

    # Update Missing Vals
    for i in range(0, len(nm.mv_cols)):
        df[nm.mv_cols[i]].replace([np.nan], nm.mv_update_vals[i], inplace=True)

    # Encode categorical (nominal) cols as OHE variables
    df = pd.get_dummies(df, prefix=nm.nominal_cols, columns=nm.nominal_cols, drop_first=False, dtype='int64')

    # Update string values for ordinal cols (e.g. cols for previous race positions)
    for i in range(0, len(nm.ordinal_cols)):
        df.replace({nm.ordinal_cols[i]: nm.ordinal_vals}, inplace=True)

    # Create binary results column
    df["result_bin"] = np.where(df["Result"] == '1', 1, 0)
    df.drop(columns=["Result"], inplace=True)

    # Convert ordinal categorical cols into numeric
    for i in range(0, len(nm.ordinal_cols)):
        df[nm.ordinal_cols[i]] = df[nm.ordinal_cols[i]].astype('int64')

    return df


def create_race_data(df):
    df_race = df[nm.joining_col]
    df_race2 = df[nm.race_cols]
    frames = [df_race, df_race2]
    df_race_merge = pd.concat(frames, axis=1)
    df_race_merge.drop_duplicates(inplace=True)
    return df_race_merge


def create_horse_data(df):
    df_horse = df.drop(nm.race_cols, axis=1, inplace=False)
    second_column = df_horse.pop(nm.ident_col[0])
    df_horse.insert(0, nm.ident_col[0], second_column)
    first_column = df_horse.pop(nm.joining_col[0])
    df_horse.insert(0, nm.joining_col[0], first_column)
    return df_horse


def rpr_extract_odds_cols(df, max_n_horses):
    df_new = pd.DataFrame()
    df2 = df.copy()
    for i in range(1, max_n_horses + 1):
        last_column = df2.pop('ValueOdds_BetfairFormat_' + str(i))
        a = len(df_new.columns)
        df_new.insert(a, 'ValueOdds_BetfairFormat_' + str(i), last_column)
    return df_new


def extract_horse_cols(df, max_n_horses):
    df_new = pd.DataFrame()
    for i in range(1, max_n_horses + 1):
        last_column = df.pop('Horse_' + str(i))
        a = len(df_new.columns)
        df_new.insert(a, 'Horse_' + str(i), last_column)
    return df_new, df


def rpr_normalise(df):
    ss = StandardScaler()
    df_new = pd.DataFrame(ss.fit_transform(df))
    return df_new


def order_by_date(df):
    df['Date'] = pd.to_datetime(df["Date"])
    df.sort_values(by='Date', inplace=True)
    # df.pop('Date')
    return df


def test_train_split_inorder(df, max_horses):
    a = len(df.index)
    b = int(a*0.8)
    X = df[df.columns[:-max_horses]]
    y = df[df.columns[-max_horses:]]
    X_train = X.head(b)
    X_test = X.drop(X_train.index)
    y_train = y.head(b)
    y_test = y.drop(X_train.index)
    return X_train, X_test, y_train, y_test








filename = 'C:/Users/e1187273/Pictures/Horse Racing Data/HR_DATA_COMB2.csv'
hr_data = pd.read_csv(filename)
hr_data = transform_data(hr_data)
hr_data_extract = hr_data.iloc[0:100, :]
# hr_data_extract.to_csv('C:/Users/e1187273/Pictures/Horse Racing Data/hr_extract.csv')
# hr_race_data = create_race_data(hr_data)
# hr_horse_data = create_horse_data(hr_data)
# hr_horse_extract = hr_horse_data.iloc[0:10, 0:10]
# test = create_index_col(hr_horse_extract)


# test = one_row_per_race(hr_horse_extract)
# test.to_csv('C:/Users/e1187273/Pictures/Horse Racing Data/pivot.csv')
# hr_horse_extract.to_csv('C:/Users/e1187273/Pictures/Horse Racing Data/hr_extract.csv')

# print(hr_data['Position5RunsAgo'].unique())
# dtype = hr_data.dtypes
# dtype.to_csv('C:/Users/e1187273/Pictures/Horse Racing Data/dtypes.csv')
# a = list(hr_data.columns)
# a= pd.Series(a)
# a.to_csv('C:/Users/e1187273/Pictures/Horse Racing Data/new_cols.csv')
# a = hr_race_data['UKHR_RaceID'].value_counts()
# a.to_csv('C:/Users/e1187273/Pictures/Horse Racing Data/UKHRRaceID_counts.csv')
