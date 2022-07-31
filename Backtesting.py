import numpy as np
import pandas as pd
import RPR_Modelling as rpr_mod
from datetime import date, timedelta

class build_predict_models:

    '''Class to iteratively build models and predict '''

    def __init__(self, race_data, result_data, odds_lookup, probs_lookup, dates, create_model, start_train, end_train, end_test):
        self.race_data = race_data
        self.result_data = result_data
        self.odds_lookup = odds_lookup
        self.probs_lookup = probs_lookup
        self.dates = dates
        self.start_train = start_train
        self.end_train = end_train
        self.end_test = end_test
        self.create_model = create_model


    def build_initial_model(self):
        mask = (self.dates['Date'] >= self.start_train) & (self.dates['Date'] <= self.end_train)
        X_train = self.race_data.loc[mask]
        y_train = self.result_data.loc[mask]
        model = self.create_model(X_train, y_train)

        return model



filename = 'C:/Users/e1187273/Pictures/Horse Racing Data/X_important.csv'
X_important = pd.read_csv(filename)

filename = 'C:/Users/e1187273/Pictures/Horse Racing Data/y_full.csv'
y_full = pd.read_csv(filename)

filename = 'C:/Users/e1187273/Pictures/Horse Racing Data/odds_lookup.csv'
odd_lookup = pd.read_csv(filename)

filename = 'C:/Users/e1187273/Pictures/Horse Racing Data/prob_lookup.csv'
prob_lookup = pd.read_csv(filename)

filename = 'C:/Users/e1187273/Pictures/Horse Racing Data/dates.csv'
dates = pd.read_csv(filename)

backtester = build_predict_models(X_important, y_full, odd_lookup, prob_lookup, dates,
                                  rpr_mod.create_rpr_model, '2015-01-01', '2017-12-31', '2022-05-31')

X_trial, y_trial = backtester.build_initial_model()
