import pandas as pd
import numpy as np
import Data_Processing as dp


def create_concatenated_id(df):
    return df.astype(str).apply('_'.join, axis=1)

# 'RaceType_Chase', 'RaceType_Hurdle', 'RaceType_NH Flat', 'Going_FROZEN', 'Going_GD-SFT',  'Going_Unknown', 'Going_YIELD', 'Going_YLD-SFT',
# 'Handicap_Nursery', , 'Maiden_Maiden',

class RowPerRaceTransformer:
    def __init__(self, df, max_n_horses):
        self.df = df
        self.max_n_horses = max_n_horses
        self.suffixes = range(1, max_n_horses + 1)
        self.race_colnames = ['Date', 'RaceClass', 'Furlongs', 'Prize', 'MinAge', 'MaxAge', 'MeanWeight', 'Runners',
              'Going_GD-FM',  'Going_GD-YLD', 'Going_GOOD', 'Going_HEAVY',
             'Going_SFT-HVY', 'Going_SOFT', 'Handicap_Handicap', 'Handicap_Non Handicap',
              'Novice_Non Novice', 'Novice_Novice',  'Maiden_Non Maiden']
        self.horse_colnames = list(set(df.columns) - set(self.race_colnames))
        self.new_horse_colnames = self.create_new_horse_colnames()
        self.new_colnames = self.race_colnames + self.new_horse_colnames

    def transform_full_dataframe(self):
        df_transformed = self.df.groupby('UKHRCardRaceID').apply(
            self.transform_race_to_row)
        df_transformed.columns = self.new_colnames
        df_transformed = df_transformed.reset_index().drop(['level_1'], axis=1)
        df_transformed = self.move_horse_cols(df_transformed)
        df_transformed = self.move_result_cols(df_transformed)
        return df_transformed

    def transform_race_to_row(self, df_race, winning_label='1'):
        n_horses = df_race.shape[0]

        if n_horses > self.max_n_horses:
            return None

        race_info = df_race.iloc[0][self.race_colnames].tolist()
        horses_info = list(df_race[self.horse_colnames].values.flatten())
        padding = np.full([1, (self.max_n_horses - n_horses) *
                           len(self.horse_colnames)], 0).tolist()[0]

        full_info = []
        full_info.extend(race_info)
        full_info.extend(horses_info)
        full_info.extend(padding)

        race_row = pd.DataFrame(full_info).transpose()
        # race_row['result_bin'] = self.create_result_label(df_race, winning_label=winning_label)

        return race_row

    def create_result_label(self, df_race, winning_label='1'):
        winning_horse_number = np.flatnonzero(
            df_race['result_bin'] == winning_label)
        if len(winning_horse_number) != 1:
            return np.nan
        else:
            return winning_horse_number[0] + 1

    def create_new_horse_colnames(self):
        return ['{}_{}'.format(horse_info, suffix) for suffix in self.suffixes
                for horse_info in self.horse_colnames]

    def move_result_cols(self, df):
        for i in range(1, self.max_n_horses+1):
            last_column = df.pop('result_bin_' + str(i))
            a = len(df.columns)
            df.insert(a, 'result_bin_' + str(i), last_column)
        return df

    def move_horse_cols(self, df):
        for i in range(1, self.max_n_horses+1):
            last_column = df.pop('Horse_' + str(i))
            a = len(df.columns)
            df.insert(a, 'Horse_' + str(i), last_column)
        return df

# a = RowPerRaceTransformer(dp.hr_data,12)
# print('started')
# b = a.transform_full_dataframe()
# print('completed')
# b.to_csv('C:/Users/e1187273/Pictures/Horse Racing Data/race_to_row.csv')
# print('Exported')
