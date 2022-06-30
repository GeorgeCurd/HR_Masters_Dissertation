import pandas as pd
import numpy as np

def drop_cols(data):
    dropped = data.drop(
        ['Meeting', 'Time', 'Title', 'Horse', 'CardNumber', 'StallNumber', 'StallPercentage', 'Weight_StonesPounds',
         'Jockey', 'Trainer', 'Alarms', 'ClassPosition', 'WinClassProbability', 'WinClassProbability_Normalised',
         'ValueOdds', 'ForecastSP', 'CSVversion', 'Date', 'ElapsedDays', 'Systems', 'Time24Hour', 'BST_GMT',
         'HorseForm', 'ConnRanking', 'FrmRanking', 'LstRanking', 'ClsRanking', 'WinFRanking', 'SpdRanking', 'HCPRanking',
         'Country', 'RGoingRanking', 'RDistanceRanking', 'UKHRCardHorseID', 'UKHRCardTrainerID', 'UKHRCardJockeyID',
         'UKHRCardCourseID', 'Sire', 'UKHR_SireID', 'Dam', 'UKHR_DamID', 'Betfair Placed', 'Betfair Place S.P.', 'Betfair Win S.P.',
         'Actual Going', 'S.P.', 'Actual Runners', 'WRITE_IN_DURATION_HERE', 'UKHR_RaceID', 'UKHR_EntryID', 'UKHR_HorseID',
         'UKHR_TrainerID', 'UKHR_JockeyID', 'UKHR_CourseID', 'LengthsBehind', 'LengthsBehindTotal', 'Duration',
         'WRITE_FAVOURITE_RANKING', 'Claiming', 'Selling', 'Auction', 'HunterChase', 'Beginner', 'LengthsWonLost5RunsAgo', 'BHAclassLastType',
         'LengthsWonLost4RunsAgo', 'BHAclassLast', 'LengthsWonLost3RunsAgo', 'BHAclassToday', 'Going5RunsAgo',
         'LengthsWonLost2RunsAgo', 'Going4RunsAgo', 'Going3RunsAgo', 'LastTimePositionRaceType', 'LengthsWonLostLastRun', 'Going2RunsAgo', 'GoingLastTime', ]
        , axis=1)
    return dropped


def update_missing_vals(df,col,val):
    df.

filename = 'C:/Users/e1187273/Pictures/Horse Racing Data/HR_DATA_COMB2.csv'
df = pd.read_csv(filename)
df = drop_cols(df)
print(df['Position5RunsAgo'].unique())
a = df.isna().sum()
# a.to_csv('C:/Users/e1187273/Pictures/Horse Racing Data/missing_vals.csv')
