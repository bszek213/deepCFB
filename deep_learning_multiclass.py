#multiclass deep-learning on college football games
from pandas import read_csv, DataFrame, concat, io
from os.path import join, exists
from os import getcwd, mkdir
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy import stats
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping
from keras_tuner.tuners import RandomSearch
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.models import Model
from pickle import dump, load
from colorama import Fore, Style
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from collect_augment_data import collect_two_teams
from numpy import nan, array, reshape, arange
from sys import argv
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import subprocess
import yaml 

def build_classifier(hp):
    model = keras.Sequential()

    # Tune the number of layers
    num_layers = hp.Int('num_layers', min_value=1, max_value=5, step=1)

    for i in range(num_layers):
        # Tune the number of units in each layer
        units = hp.Int(f'units_layer_{i}', min_value=8, max_value=512, step=32)

        # Tune the activation function
        activation = hp.Choice(f'activation_layer_{i}', values=['relu', 'tanh', 'sigmoid'])

        model.add(layers.Dense(units=units, activation=activation))

        # Tune dropout rate
        dropout_rate = hp.Float(f'dropout_layer_{i}', min_value=0.0, max_value=0.5, step=0.1)
        model.add(layers.Dropout(rate=dropout_rate))

    # Add the output layer with 'softmax' activation for multi-class classification
    model.add(layers.Dense(2, activation='softmax'))

    # Tune the optimizer
    optimizer = hp.Choice('optimizer', values=['adam', 'rmsprop', 'sgd'])
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')

    if optimizer == 'adam':
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                    loss='categorical_crossentropy',  # Change the loss function
                    metrics=['accuracy'])
    elif optimizer == 'rmsprop':
        model.compile(optimizer=RMSprop(learning_rate=learning_rate),
                    loss='categorical_crossentropy',  # Change the loss function
                    metrics=['accuracy'])
    else:
        model.compile(optimizer=SGD(learning_rate=learning_rate),
                    loss='categorical_crossentropy',  # Change the loss function
                    metrics=['accuracy'])

    return model

def create_model_classifier(hp,shape_input):
    #Feature model
    optimizer = hp.Choice('optimizer', ['adam', 'rmsprop'])
    units = hp.Int('units', min_value=5, max_value=100, step=5)
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
    inputs = Input(shape=(shape_input,))

    shared_hidden_layer = Dense(units, activation='relu')(inputs)
    shared_hidden_layer = Dense(units, activation='tanh')(shared_hidden_layer)
    shared_hidden_layer = Dense(units, activation='relu')(shared_hidden_layer)
    shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
    shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)

    output_layers = []
    for i in range(shape_input):
        output_layer = Dense(1, activation='tanh', name=f'target_{i+1}')(shared_hidden_layer)
        output_layers.append(output_layer)

    if optimizer == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    else:
        optimizer = RMSprop(learning_rate=learning_rate)
    model = Model(inputs=inputs, outputs=output_layers)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])

    return model

class deepCfbMulti():
    def __init__(self):
        print('instantiate deepCfbMulti class')

    def str_manipulations(self,df):
        #extract outcomes and scores
        df['team_1_outcome'] = df['game_result'].apply(lambda x: 1 if x[0] == 'W' else 0)
        df['team_2_outcome'] = df['game_result'].apply(lambda x: 1 if x[0] == 'L' else 0)
        df['team_1_score'] = df['game_result'].str.extract(r'(\d+)-\d+').astype(int)
        df['team_2_score'] = df['game_result'].str.extract(r'\d+-(\d+)').astype(int)
        df.drop(columns=['game_result'],inplace=True)

        return df
    
    def split_classifier(self):
        #Read in data
        self.all_data = read_csv(join(getcwd(),'all_data.csv'))
        self.all_data = concat([self.all_data, read_csv(join(getcwd(),'all_data_2023.csv'))])

        self.x_regress = read_csv(join(getcwd(),'x_feature_regression.csv')) 
        self.x_regress = concat([self.x_regress, read_csv(join(getcwd(),'x_feature_regression_2023.csv'))])

        self.y_regress = read_csv(join(getcwd(),'y_feature_regression.csv')) 
        self.y_regress = concat([self.y_regress, read_csv(join(getcwd(),'y_feature_regression_2023.csv'))])

        self.all_data = self.str_manipulations(self.all_data)
        self.x_regress = self.str_manipulations(self.x_regress)
        self.y_regress = self.str_manipulations(self.y_regress)

        self.classifier_drop = ['team_1_outcome','team_2_outcome','game_loc']
        self.y = self.all_data[['team_1_outcome','team_2_outcome']]
        self.x = self.all_data.drop(columns=self.classifier_drop)

        print(f'number of features: {len(self.x.columns)}')
        print(f'number of samples: {len(self.x)}')
        self.manual_comp = len(self.x.columns)

        #Standardize
        self.scaler = MinMaxScaler(feature_range=(0,1))
        X_std = self.scaler.fit_transform(self.x)
        #FA
        self.fa = FactorAnalysis(n_components=self.manual_comp)
        X_fa = self.fa.fit_transform(X_std)
        self.x_data = DataFrame(X_fa, columns=[f'FA{i}' for i in range(1, self.manual_comp+1)])

        #drop non-normal columns - removes columns that have no distribution (ie they are binary data) - Exploratory for now
        self.non_normal_columns = []
        for column in self.x_data.columns:
            stat, p = stats.shapiro(self.x_data[column])
            if p == 1:
                self.non_normal_columns.append(column)
        self.x_data = self.x_data.drop(self.non_normal_columns, axis=1)

        #split data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_data, self.y, train_size=0.8)
    
    def multiclass_class(self):
        if not exists('multiclass_models'):
            mkdir("multiclass_models")
        abs_path = join(getcwd(),'multiclass_models','keras_classifier_mc.h5')
        if exists(abs_path):
            self.dnn_class = keras.models.load_model(abs_path)
        else:
            tuner = RandomSearch(
                build_classifier,
                objective='val_accuracy',
                max_trials=125,  # Number of combinations to try
                directory='classifier_multiclass',  # Directory to store the results
                project_name='classifier_multiclass_project'
            )

            early_stop = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1)
            tuner.search(self.x_train, self.y_train, 
                        epochs=100, batch_size=128, 
                        validation_data=(self.x_test, self.y_test),
                        callbacks=[early_stop]) 
            
            best_model = tuner.get_best_models(1)[0]
            best_model.fit(self.x_train, self.y_train, epochs=100, batch_size=128, verbose=2,
                        validation_data=(self.x_test, self.y_test),
                        callbacks=[early_stop])
            best_model.save(abs_path)
            self.dnn_class = best_model
            validation_loss, validation_accuracy = best_model.evaluate(self.x_test, self.y_test)
            print(f'Validation Loss: {validation_loss}')
            print(f'Validation Accuracy: {validation_accuracy}')
    
    def deep_learn_features(self):
        #drop target label
        self.x_regress.drop(columns=self.classifier_drop,inplace=True)
        self.y_regress.drop(columns=self.classifier_drop,inplace=True)
        #standardize
        X_std = self.scaler.transform(self.x_regress)
        y_std = self.scaler.transform(self.y_regress)
        # FA
        X_fa = self.fa.transform(X_std)
        Y_fa = self.fa.transform(y_std)
        #create DF
        x_regress = DataFrame(X_fa, columns=[f'FA{i}' for i in range(1, self.manual_comp+1)])
        y_regress = DataFrame(Y_fa, columns=[f'FA{i}' for i in range(1, self.manual_comp+1)])

        #remove non-normal distributions
        x_regress.drop(self.non_normal_columns, axis=1, inplace=True)
        y_regress.drop(self.non_normal_columns, axis=1, inplace=True)
        
        x_train, x_test, y_train, y_test = train_test_split(x_regress, y_regress, train_size=0.8)

        #dnn feayures
        if not exists('multiclass_models'):
            mkdir("multiclass_models")
        abs_path = join(getcwd(),'multiclass_models','feature_dnn_classifier.h5')
        if exists(abs_path):
            print('load trained feature regression model')
            self.model_feature_regress_model = keras.models.load_model(abs_path)
        else:
            #FIND BEST PARAMETERS
            tuner = RandomSearch(
                lambda hp: create_model_classifier(hp,x_regress.shape[1]),
                objective='val_loss',
                max_trials=100,
                directory='feature_learning_dnn',
                project_name='model_tuning')
            tuner.search_space_summary()
            tuner.search(x_train, y_train, validation_data=(x_test, y_test), epochs=120)

            # Get the best model and summary of the best hyperparameters
            best_model = tuner.get_best_models(num_models=1)[0]
            best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
            best_model.summary()
            hyperparams = best_hyperparameters.values
            print(hyperparams)
            best_model.save(abs_path)

        lin_abs_path = join(getcwd(),'multiclass_models','feature_linear_regression.pkl')
        # if not exists(lin_abs_path):
        lin_model = LinearRegression().fit(x_train,y_train)
        y_pred = lin_model.predict(x_test)
        y_test_np = y_test.to_numpy()
        mse_error = mean_squared_error(y_test_np,y_pred)
        print(f'Linear Regression MSE: {mse_error}')
        with open(lin_abs_path, 'wb') as file:
                dump(lin_model, file)
        self.feature_linear_regression = lin_model 
        # else:
        # with open(lin_abs_path, 'rb') as file:
        #     self.feature_linear_regression = load(file)

        #random forest features
        lin_abs_path = join(getcwd(),'multiclass_models','feature_random_forest.pkl')
        if not exists(lin_abs_path):
            param_grid = {
                'n_estimators': [300, 400, 500],
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10],  # Change min_child_weight to min_samples_split
                'min_samples_leaf': [1, 2, 4],  # Change gamma to min_samples_leaf
            }

            # Train the Random Forest model and Create the GridSearchCV object
            grid_search = GridSearchCV(estimator=RandomForestRegressor(), 
                                    param_grid=param_grid, 
                                    cv=3, n_jobs=10, 
                                    verbose=3,
                                    scoring='neg_mean_squared_error')

            # Fit the GridSearchCV object to the training data
            grid_search.fit(x_train, y_train)
            
            #check validation data
            val_predictions = grid_search.predict(x_test)
            val_mse = mean_squared_error(y_test, val_predictions)

            # Print the best parameters and best score
            print("Best Parameters: ", grid_search.best_params_)
            print("Best Explained Variance: ", grid_search.best_score_)
            print(f'Valdition MSE: {val_mse}')

            # Save the trained model to a file
            with open(lin_abs_path, 'wb') as file:
                    dump(grid_search, file)
            self.feature_rf = grid_search
        else:
            with open(lin_abs_path, 'rb') as file:
                self.feature_rf = load(file)

    def test_forecast(self):
        #all teams
        # teams_list = get_teams_year(2015,2023)
        #Select certain teams
        # with open('teams_played_this_week.txt','r') as file:
        #      content = file.read()
        # teams_list = content.split("\n")
        # teams_list = [string for string in teams_list if string.strip() != ""]
        with open(join(getcwd(),'team_rankings_year.yaml')) as file:
            teams_dict_year = yaml.load(file, Loader=yaml.FullLoader)
        teams_list = teams_dict_year[2023]

        count_teams = 1
        lin_out = 0
        roll_out = 0
        dnn_out = 0
        rf_out= 0
        roll_3 = 0
        roll_ewm = 0

        for abv in tqdm(teams_list):
            str_combine = 'https://www.sports-reference.com/cfb/schools/' + abv.lower() + '/' + str(2023) + '/gamelog/'
            df_inst = collect_two_teams(str_combine,abv.lower(),2023)

            df_inst = self.str_manipulations(df_inst)
            game_result_series = df_inst[['team_1_outcome','team_2_outcome']].iloc[-1]
            df_inst.drop(columns=self.classifier_drop, inplace=True)

            #Standardize
            X_std = self.scaler.transform(df_inst)
            #FA
            X_fa = self.fa.transform(X_std)
            final_df = DataFrame(X_fa, columns=[f'FA{i}' for i in range(1, self.manual_comp+1)])
            final_df.drop(self.non_normal_columns, axis=1, inplace=True)

            _, df_forecast_second  = self.extract_features(final_df)
            feature_data = df_forecast_second.to_numpy().reshape(1, -1)

            #running median calculation
            rolling_features_2 = final_df.rolling(2).median().iloc[-1:]
            rolling_features_3 = final_df.rolling(3).median().iloc[-1:]
            rolling_features_mean_2 = final_df.ewm(span=2).mean().iloc[-1:]

            #Feature prediction
            # next_game_features_lin = self.feature_linear_regression.predict(feature_data)
            # next_game_features_dnn = self.model_feature_regress_model.predict(feature_data)
            # next_game_features_rf = self.feature_rf.predict(feature_data)

            #multi-learning output manipulation
            # dnn_list = []
            # for val in next_game_features_dnn:
            #     dnn_list.append(val[0][0])
            # dnn_list = array(dnn_list)
            # dnn_list = reshape(dnn_list, (1,len(dnn_list)))

            #Predictions
            # prediction_dnn = self.dnn_class.predict(dnn_list)
            # prediction_lin = self.dnn_class.predict(next_game_features_lin)
            # prediction_rf = self.dnn_class.predict(next_game_features_rf)
            prediction_rolling = self.dnn_class.predict(rolling_features_2)
            prediction_rolling_3 = self.dnn_class.predict(rolling_features_3)
            prediction_rolling_ewm = self.dnn_class.predict(rolling_features_mean_2)

            #check if outcome is above 0.5 for team 1
            # if prediction_dnn[0][0] > 0.5:
            #     result_dnn = 1
            # else:
            #     result_dnn = 0
            # if prediction_lin[0][0] > 0.5:
            #     result_lin = 1
            # else:
            #     result_lin = 0
            if prediction_rolling[0][0] > 0.5:
                result_rolling = 1
            else:
                result_rolling = 0
            # if prediction_rf[0][0] > 0.5:
            #     result_rf = 1
            # else:
            #     result_rf = 0
            if prediction_rolling_3[0][0] > 0.5:
                result_rolling_3 = 1
            else:
                result_rolling_3 = 0
            if prediction_rolling_ewm[0][0] > 0.5:
                result_rolling_ewm = 1
            else:
                result_rolling_ewm = 0

            # if int(game_result_series['team_1_outcome']) == result_dnn:
            #         dnn_out += 1
            # if int(game_result_series['team_1_outcome']) == result_lin:
            #         lin_out += 1
            if int(game_result_series['team_1_outcome']) == result_rolling:
                    roll_out += 1
            # if int(game_result_series['team_1_outcome']) == result_rf:
            #         rf_out += 1
            if int(game_result_series['team_1_outcome']) == result_rolling_3:
                    roll_3 += 1
            if int(game_result_series['team_1_outcome']) == result_rolling_ewm:
                    roll_ewm += 1
            
            print('=======================================')
            # print(f'DNN Accuracy out of {count_teams} teams: {dnn_out / count_teams}')
            # print(f'LinRegress Accuracy out of {count_teams} teams: {lin_out / count_teams}')
            # print(f'RandomForest Accuracy out of {count_teams} teams: {rf_out / count_teams}')
            print(f'Rolling median 2 Accuracy out of {count_teams} teams: {roll_out / count_teams}')
            print(f'Rolling median 3 Accuracy out of {count_teams} teams: {roll_3 / count_teams}')
            print(f'Rolling EWM 2 Accuracy out of {count_teams} teams: {roll_ewm / count_teams}')
            print('=======================================')
            count_teams += 1

    def predict_teams(self):
        while True:
            # try:
                self.team_1 = input('team_1: ')
                if self.team_1 == 'exit':
                    break
                self.team_2 = input('team_2: ')

                str_combine = 'https://www.sports-reference.com/cfb/schools/' + self.team_1.lower() + '/' + str(2023) + '/gamelog/'
                team_1_df = collect_two_teams(str_combine,self.team_1.lower(),2023)
                str_combine = 'https://www.sports-reference.com/cfb/schools/' + self.team_2.lower() + '/' + str(2023) + '/gamelog/'
                team_2_df = collect_two_teams(str_combine,self.team_2.lower(),2023)

                team_1_df = self.str_manipulations(team_1_df)
                team_2_df = self.str_manipulations(team_2_df)
                team_1_df.drop(columns=self.classifier_drop, inplace=True)
                team_2_df.drop(columns=self.classifier_drop, inplace=True)

                columns_to_replace = [col for col in team_2_df.columns if '_opp' not in col]
                strings_to_remove = ['team_1_score', 'team_2_score']
                columns_to_replace = [item for item in columns_to_replace if item not in strings_to_remove]

                # Ensure the length of both dfs match
                length_difference = len(team_1_df) - len(team_2_df)
                if length_difference > 0:
                    team_1_df = team_1_df.iloc[length_difference:]
                elif length_difference < 0:
                    team_2_df = team_2_df.iloc[-length_difference:]
                # if len(team_1_df) > len(team_2_df):
                #     team_2_df = team_2_df.iloc[-len(team_1_df):]
                # if len(team_2_df) < len(team_1_df):
                #     team_2_df = concat([team_2_df] * (len(team_1_df) // len(team_2_df)) + [team_2_df.iloc[:len(team_1_df) % len(team_2_df)]])
                # Replace the data from one df with the corresponding data from other df
                for col in columns_to_replace:
                    opp_col = col + '_opp'
                    if opp_col in team_1_df.columns:
                        team_1_df[opp_col] = team_2_df[col]

                team_1_df['team_2_score'] = team_2_df['team_1_score']

                # print("Non-opp columns of df2:")
                # print(team_2_df[columns_to_replace])
                # print("\nOpp columns of df1:")
                # print(team_1_df[team_1_df.columns[team_1_df.columns.str.endswith('_opp')]])
                # input()

                #Standardize and FA
                X_std = self.scaler.transform(team_1_df)
                X_fa = self.fa.transform(X_std)
                final_df_1 = DataFrame(X_fa, columns=[f'FA{i}' for i in range(1, self.manual_comp+1)])
                final_df_1.drop(self.non_normal_columns, axis=1, inplace=True)

                # X_std = self.scaler.transform(team_2_df)
                # X_fa = self.fa.transform(X_std)
                # final_df_2 = DataFrame(X_fa, columns=[f'FA{i}' for i in range(1, self.manual_comp+1)])
                # final_df_2.drop(self.non_normal_columns, axis=1, inplace=True)

                forecast_team_1, _  = self.extract_features(final_df_1)
                # forecast_team_2, _  = self.extract_features(final_df_2)
                feature_data_team_1 = forecast_team_1.to_numpy().reshape(1, -1)
                # feature_data_team_2 = forecast_team_2.to_numpy().reshape(1, -1)

                #running median calculation
                rolling_features_2_team_1 = final_df_1.rolling(2).median().iloc[-1:]
                rolling_features_3_team_1 = final_df_1.rolling(3).median().iloc[-1:]
                rolling_features_ewm = final_df_1.ewm(span=2).mean().iloc[-1:]
                rolling_low = final_df_1.rolling(window=2).quantile(0.25).iloc[-1:]
                rolling_high = final_df_1.rolling(window=2).quantile(0.75).iloc[-1:]

                #Feature prediction
                # next_game_features_lin = self.feature_linear_regression.predict(forecast_team_1)
                # next_game_features_dnn = self.model_feature_regress_model.predict(feature_data_team_1)
                # next_game_features_rf = self.feature_rf.predict(feature_data_team_1)

                # #multi-learning output manipulation
                # dnn_list = []
                # for val in next_game_features_dnn:
                #     dnn_list.append(val[0][0])
                # dnn_list = array(dnn_list)
                # dnn_list = reshape(dnn_list, (1,len(dnn_list)))

                #Predictions
                # prediction_dnn_1 = self.dnn_class.predict(dnn_list)
                # prediction_lin_1 = self.dnn_class.predict(next_game_features_lin)
                # prediction_rf_1 = self.dnn_class.predict(next_game_features_rf)
                prediction_rolling_1 = self.dnn_class.predict(rolling_features_2_team_1)
                prediction_rolling_2 = self.dnn_class.predict(rolling_features_3_team_1)
                prediction_rolling_ewm = self.dnn_class.predict(rolling_features_ewm)
                prediction_low= self.dnn_class.predict(rolling_low)
                prediction_high= self.dnn_class.predict(rolling_high)

                print('==============================')
                # print('Win Probabilities from DNN feature predictions')
                # print(Fore.YELLOW + Style.BRIGHT + f'{self.team_1} : {(prediction_dnn_1[0][0])*100} %' + Fore.CYAN + Style.BRIGHT +
                #     f' {self.team_2} : {(prediction_dnn_1[0][1])*100} %'+ Style.RESET_ALL)
                # print('Win Probabilities from LinRegress feature predictions')
                # print(Fore.YELLOW + Style.BRIGHT + f'{self.team_1} : {(prediction_lin_1[0][0])*100} %' + Fore.CYAN + Style.BRIGHT +
                #     f' {self.team_2} : {(prediction_lin_1[0][1])*100} %'+ Style.RESET_ALL)
                # print('Win Probabilities from RandomForest feature predictions')
                # print(Fore.YELLOW + Style.BRIGHT + f'{self.team_1} : {(prediction_rf_1[0][0])*100} %' + Fore.CYAN + Style.BRIGHT +
                #     f' {self.team_2} : {(prediction_rf_1[0][1])*100} %'+ Style.RESET_ALL)
                print('Win Probabilities from rolling median of 2 predictions')
                print(Fore.YELLOW + Style.BRIGHT + f'{self.team_1} : {round((prediction_rolling_1[0][0])*100,3)} %' + Fore.CYAN + Style.BRIGHT +
                    f' {self.team_2} : {round((prediction_rolling_1[0][1])*100,3)} %'+ Style.RESET_ALL)
                print('Win Probabilities from rolling median of 3 predictions')
                print(Fore.YELLOW + Style.BRIGHT + f'{self.team_1} : {round((prediction_rolling_2[0][0])*100,3)} %' + Fore.CYAN + Style.BRIGHT +
                    f' {self.team_2} : {round((prediction_rolling_2[0][1])*100,3)} %'+ Style.RESET_ALL)
                print('Win Probabilities from exponential weighted average of 2 predictions')
                print(Fore.YELLOW + Style.BRIGHT + f'{self.team_1} : {round((prediction_rolling_ewm[0][0])*100,3)} %' + Fore.CYAN + Style.BRIGHT +
                    f' {self.team_2} : {round((prediction_rolling_ewm[0][1])*100,3)} %'+ Style.RESET_ALL)
                print('Win Probabilities from 25th and 75th percentile rolling 2')
                print(Fore.YELLOW + Style.BRIGHT + f'25th: {self.team_1} : {round((prediction_low[0][0])*100,3)} %' + Fore.CYAN + Style.BRIGHT +
                    f' {self.team_2} : {round((prediction_low[0][1])*100,3)} %'+ Style.RESET_ALL)
                print(Fore.YELLOW + Style.BRIGHT + f'75th: {self.team_1} : {round((prediction_high[0][0])*100,3)} %' + Fore.CYAN + Style.BRIGHT +
                    f' {self.team_2} : {round((prediction_high[0][1])*100,3)} %'+ Style.RESET_ALL)
                print('==============================')
                #run mysrs
                print('Running my SRS analysis...')
                command = f"python3 simple_rating_system.py --all no --team_1 {self.team_1} --team_2 {self.team_2}"
                process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

                for line in process.stdout:
                    print(line, end='')

                process.wait()  # Wait for the process to finish
                del team_1_df, team_2_df
            # except Exception as e:
            #      print(f'The error: {e}. Most likely {self.team_1} or {self.team_2} do not have data')

    def extract_features(self,df):
        team_df_forecast_last = df.iloc[-1:] #last game
        try:
            team_df_forecast_second = df.iloc[-2] #2nd to last game
        except:
             team_df_forecast_second = nan
        return team_df_forecast_last, team_df_forecast_second

    def run_analysis(self):
        self.split_classifier()
        self.multiclass_class()
        self.deep_learn_features()
        if argv[1] == "test":
            self.test_forecast()
        else:
            self.predict_teams()

def main():
    deepCfbMulti().run_analysis()

if __name__ == "__main__":
    main()