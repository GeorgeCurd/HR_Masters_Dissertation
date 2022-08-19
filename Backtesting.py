import numpy as np
import pandas as pd
import RPR_Modelling as rpr_mod
from datetime import date, timedelta
from datetime import datetime
import calendar


class Backtesting:
    """Class to iteratively build models and predict """

    def __init__(self, race_data, result_data, odds_lookup, probs_lookup, horse_lookup, race_lookup, dates, create_model, start_train, end_train,
                 end_test, starting_bal, strategy_type, strategy_size):
        self.race_data = race_data
        self.result_data = result_data
        self.odds_lookup = odds_lookup
        self.probs_lookup = probs_lookup
        self.horse_lookup = horse_lookup
        self.race_lookup = race_lookup
        self.dates = dates
        self.dates['Date'] = pd.to_datetime(dates['Date'])
        self.start_train = datetime.strptime(start_train, '%Y-%m-%d')
        self.end_train = datetime.strptime(end_train, '%Y-%m-%d')
        self.end_test = datetime.strptime(end_test, '%Y-%m-%d')
        self.create_model = create_model
        self.current_date = datetime.strptime(end_train, '%Y-%m-%d') + timedelta(days=1)
        self.starting_bal = starting_bal
        self.strategy_type = strategy_type
        self.strategy_size = strategy_size
        self.current_balance = starting_bal
        self.daily_bet_results = pd.DataFrame()

    def build_model(self):
        mask = (self.dates['Date'] >= self.start_train) & (self.dates['Date'] <= self.end_train)
        X_train = self.race_data.loc[mask]
        y_train = self.result_data.loc[mask]
        model = self.create_model(X_train, y_train)
        print('model built for dates ' + str(self.start_train) + 'and ' + str(self.end_train))
        return model


    def daily_model_preds(self):
        """ Function for daily betting and results """

        mth_start = self.current_date
        mth_end = date(self.current_date.year, self.current_date.month,
                       calendar.monthrange(self.current_date.year, self.current_date.month)[-1])
        mth_end = datetime.combine(mth_end, datetime.min.time())
        model = backtester.build_model()

        while mth_start <= mth_end:
            mask = (self.dates['Date'] == mth_start)
            X_test = self.race_data.loc[mask]
            if len(X_test.index) == 0:
                print('complete for day ' + str(mth_start) + ' no racing today')
                mth_start = mth_start + timedelta(days=1)
            else:
                preds = rpr_mod.create_predictions(model, X_test)
                spb = self.bet_strategy()
                a = self.bet_on_best(preds, mth_start, spb)
                self.daily_bet_results = self.daily_bet_results.append(a, ignore_index=True)
                self.balance_update(mth_start)
                print('complete for day ' + str(mth_start))
                mth_start = mth_start + timedelta(days=1)
            self.end_train = mth_end
            self.current_date = mth_end + timedelta(days=1)

    def bet_on_best(self, preds, dt, size_bet):
        """ Function for compiling daily bets and results """
        mask = (self.dates['Date'] == dt)
        hl = np.asarray(self.horse_lookup.loc[mask])
        ol = np.asarray(self.odds_lookup.loc[mask])
        rd = np.asarray(self.result_data.loc[mask])
        rl = np.asarray(self.race_lookup.loc[mask])
        output = pd.DataFrame()
        preds = np.asarray(preds)
        max_prob = preds.max(axis=1)
        max_prob_idx = np.argmax(preds, axis=1)
        horse_names = np.take_along_axis(hl, max_prob_idx[:,None], axis=1)
        odds = np.take_along_axis(ol, max_prob_idx[:, None], axis=1)
        result = np.take_along_axis(rd, max_prob_idx[:, None], axis=1)
        output['race_ID'] = pd.Series(rl[:, 0])
        day = np.asarray([dt for i in range(len(output.index))])
        output['date'] = pd.Series(day)
        output['horse_names'] = pd.Series(horse_names[:, 0])
        output['model_prob'] = pd.Series(max_prob)
        output['odds'] = pd.Series(odds[:, 0])
        output['actual_result'] = pd.Series(result[:, 0])
        mb = np.asarray([size_bet for i in range(len(output.index))])
        output['money_bet'] = pd.Series(mb)
        values = (output.money_bet * output.odds)-size_bet
        output['return'] = values.where(output.actual_result == 1, other=-size_bet)
        return output


    def bet_strategy(self):
        if self.strategy_type =='FS':
            stake_per_bet = self.strategy_size
        elif self.strategy_type == 'FP':
            stake_per_bet = self.current_balance/self.strategy_size
        elif self.strategy_type == 'Kelly':
            print('Add Kelly')
        else:
            print('Error: Strategy type must be FS, FP or Kelly')
        return stake_per_bet

    def balance_update(self, dt):
        a = self.daily_bet_results[self.daily_bet_results["date"] == dt].sum()["return"]
        self.current_balance = self.current_balance + a

    def full_backtest(self):
        while self.current_date <= self.end_test:
            self.daily_model_preds()
        return self.daily_bet_results




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

filename = 'C:/Users/e1187273/Pictures/Horse Racing Data/race_lookup.csv'
race_lookup = pd.read_csv(filename)

filename = 'C:/Users/e1187273/Pictures/Horse Racing Data/dates.csv'
dates = pd.read_csv(filename)
dates['Date'] = pd.to_datetime(dates['Date'])

backtester = Backtesting(X_important, y_full, odd_lookup, prob_lookup, horse_lookup, race_lookup, dates,
                         rpr_mod.create_rpr_model, '2015-01-01', '2017-12-31', '2022-05-31', 1000, 'FS', 1)

testing = backtester.full_backtest()
# testing.to_csv('C:/Users/e1187273/Pictures/Horse Racing Data/backtest_results.csv')

