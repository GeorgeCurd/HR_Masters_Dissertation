import numpy as np
import pandas as pd
import RPR_Modelling as rpr_mod
from datetime import date, timedelta
from datetime import datetime
import calendar


class Backtesting:
    '''Class to iteratively build models and predict '''

    def __init__(self, race_data, result_data, odds_lookup, probs_lookup, horse_lookup, dates, create_model, start_train, end_train,
                 end_test, starting_bal):
        self.race_data = race_data
        self.result_data = result_data
        self.odds_lookup = odds_lookup
        self.probs_lookup = probs_lookup
        self.horse_lookup = horse_lookup
        self.dates = dates
        self.dates['Date'] = pd.to_datetime(dates['Date'])
        self.start_train = datetime.strptime(start_train, '%Y-%m-%d')
        self.end_train = datetime.strptime(end_train, '%Y-%m-%d')
        self.end_test = datetime.strptime(end_test, '%Y-%m-%d')
        self.create_model = create_model
        self.current_date = datetime.strptime(end_train, '%Y-%m-%d') + timedelta(days=1)
        self.starting_bal = starting_bal

    def build_model(self):
        mask = (self.dates['Date'] >= self.start_train) & (self.dates['Date'] <= self.end_train)
        X_train = self.race_data.loc[mask]
        y_train = self.result_data.loc[mask]
        model = self.create_model(X_train, y_train)
        print('initial model built')
        return model


    def daily_model_preds(self):
        output_df = pd.DataFrame()
        mth_start = self.current_date
        mth_end = date(self.current_date.year, self.current_date.month,
                       calendar.monthrange(self.current_date.year, self.current_date.month)[-1])
        mth_end = datetime.combine(mth_end, datetime.min.time())
        model = backtester.build_model()

        while mth_start <= mth_end:
            mask = (self.dates['Date'] == mth_start)
            X_test = self.race_data.loc[mask]
            y_test = self.result_data.loc[mask]
            if len(X_test.index) == 0:
                print('complete for day ' + str(mth_start) + ' no racing today')
                mth_start = mth_start + timedelta(days=1)
            else:
                preds = rpr_mod.create_predictions(model, X_test)
                print('complete for day ' + str(mth_start))
                mth_start = mth_start + timedelta(days=1)

        return preds_df

    def bet_on_best(self, preds):
        # max_prob = preds.max(axis=1)
        max_prob_idx = preds.idxmax(axis=1)
        horse_names = [self.horse_lookup[i] for i in max_prob_idx]
        return horse_names


filename = 'C:/Users/e1187273/Pictures/Horse Racing Data/X_important.csv'
X_important = pd.read_csv(filename)

filename = 'C:/Users/e1187273/Pictures/Horse Racing Data/y_full.csv'
y_full = pd.read_csv(filename)

filename = 'C:/Users/e1187273/Pictures/Horse Racing Data/odds_lookup.csv'
odd_lookup = pd.read_csv(filename)

filename = 'C:/Users/e1187273/Pictures/Horse Racing Data/prob_lookup.csv'
prob_lookup = pd.read_csv(filename)

filename = 'C:/Users/e1187273/Pictures/Horse Racing Data/horse_lookup.csv'
horse_lookup = pd.read_csv(filename)

filename = 'C:/Users/e1187273/Pictures/Horse Racing Data/dates.csv'
dates = pd.read_csv(filename)
dates['Date'] = pd.to_datetime(dates['Date'])

backtester = Backtesting(X_important, y_full, odd_lookup, prob_lookup, horse_lookup, dates,
                         rpr_mod.create_rpr_model, '2015-01-01', '2017-12-31', '2022-05-31', 1000)

# testing = backtester.daily_model_preds()

def bet_on_best(preds, horse_lookup):
    preds = np.asarray(preds)
    horse_lookup = np.asarray(horse_lookup)
    max_prob = np.max(preds, axis=1)
    max_prob_idx = np.argmax(preds, axis=1)
    horse_names = np.take_along_axis(horse_lookup, max_prob_idx[:,None], axis=1)
    return max_prob_idx, horse_names

idx,horses = bet_on_best(y_full,horse_lookup)
print('done')
