cols_to_delete = ['Meeting', 'Time', 'Title', 'CardNumber', 'StallNumber', 'StallPercentage', 'Weight_StonesPounds',
         'Jockey', 'Trainer', 'Alarms', 'ClassPosition', 'WinClassProbability', 'WinClassProbability_Normalised',
         'ValueOdds', 'ForecastSP', 'CSVversion', 'ElapsedDays', 'Systems', 'Time24Hour', 'BST_GMT',
         'HorseForm', 'ConnRanking', 'FrmRanking', 'LstRanking', 'ClsRanking', 'WinFRanking', 'SpdRanking', 'HCPRanking',
         'Country', 'RGoingRanking', 'RDistanceRanking', 'UKHRCardHorseID', 'UKHRCardTrainerID', 'UKHRCardJockeyID',
         'UKHRCardCourseID', 'Sire', 'UKHR_SireID', 'Dam', 'UKHR_DamID', 'Betfair Placed', 'Betfair Place S.P.', 'Betfair Win S.P.',
         'Actual Going', 'S.P.', 'Actual Runners', 'WRITE_IN_DURATION_HERE', 'UKHRCardRaceID', 'UKHR_EntryID', 'UKHR_HorseID',
         'UKHR_TrainerID', 'UKHR_JockeyID', 'UKHR_CourseID', 'LengthsBehind', 'LengthsBehindTotal', 'Duration',
         'WRITE_FAVOURITE_RANKING', 'Claiming', 'Selling', 'Auction', 'HunterChase', 'Beginner', 'LengthsWonLost5RunsAgo', 'BHAclassLastType',
         'LengthsWonLost4RunsAgo', 'BHAclassLast', 'LengthsWonLost3RunsAgo', 'BHAclassToday',
         'LengthsWonLost2RunsAgo',  'LastTimePositionRaceType', 'LengthsWonLostLastRun' ]

mv_cols = ['Maiden', 'Novice', 'Wearing', 'Handicap', 'Position5RunsAgo', 'Position4RunsAgo', 'Gender',
            'Position3RunsAgo', 'Position2RunsAgo', 'Prize', 'PositionLastTime', 'Going','GoingLastTime',
           'Going2RunsAgo', 'Going4RunsAgo', 'Going3RunsAgo', 'Going5RunsAgo', ]

mv_update_vals = ['Non Maiden', 'Non Novice', 'Not Wearing', 'Non Handicap', 'NA', 'NA', 'Unknown', 'NA', 'NA', 0, 'NA','Unknown',
                  'Unknown','Unknown','Unknown','Unknown','Unknown']

nominal_cols = ['RaceType', 'Going', 'Gender', 'Wearing', 'Handicap', 'Novice', 'Maiden','GoingLastTime',
           'Going2RunsAgo', 'Going4RunsAgo', 'Going3RunsAgo', 'Going5RunsAgo']

ordinal_cols = ['PositionLastTime', 'Position2RunsAgo', 'Position3RunsAgo', 'Position4RunsAgo', 'Position5RunsAgo']

ordinal_vals = {'BD': 100, 'CO': 101, 'DSQ': 102, 'F': 103, 'PU': 104,
                'REF': 105, 'RO': 106, 'RR': 107, 'SU': 108, 'UR': 109, 'VOI': 110,'NA':111,'LFT':112, 'WDU':113}

race_cols = ['Date', 'RaceClass', 'Furlongs', 'Prize', 'MinAge', 'MaxAge', 'MeanWeight', 'Runners', 'RaceType_Chase', 'RaceType_Hurdle',
             'RaceType_NH Flat', 'Going_FROZEN', 'Going_GD-FM', 'Going_GD-SFT', 'Going_GD-YLD', 'Going_GOOD', 'Going_HEAVY',
             'Going_SFT-HVY', 'Going_SOFT', 'Going_Unknown', 'Going_YIELD', 'Going_YLD-SFT', 'Handicap_Handicap', 'Handicap_Non Handicap',
             'Handicap_Nursery', 'Novice_Non Novice', 'Novice_Novice', 'Maiden_Maiden', 'Maiden_Non Maiden']

joining_col = ['UKHR_RaceID']

ident_col = ['Horse']
