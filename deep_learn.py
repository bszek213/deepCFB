from pandas import read_csv, DataFrame, concat
from os.path import join, exists
from os import getcwd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import FactorAnalysis
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from pickle import dump, load
from colorama import Fore, Style
from sklearn.linear_model import LinearRegression
from keras_tuner.tuners import RandomSearch
from keras_tuner import Objective
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop, SGD
from keras.models import Model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from collect_data import get_teams_year, html_to_df_web_scrape
from tqdm import tqdm
from numpy import nan, array, reshape, arange, isnan
from xgboost import XGBRFRegressor, XGBRFClassifier
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
from sys import argv
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# from tensorflow.keras.regularizers import l2

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
        
        # Add the output layer with sigmoid activation for binary classification
        model.add(layers.Dense(1, activation='sigmoid'))

        # Tune the optimizer
        optimizer = hp.Choice('optimizer', values=['adam', 'rmsprop', 'sgd'])
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')

        if optimizer == 'adam':
            model.compile(optimizer=Adam(learning_rate=learning_rate),
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
        elif optimizer == 'rmsprop':
            model.compile(optimizer=RMSprop(learning_rate=learning_rate),
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
        else:
            model.compile(optimizer=SGD(learning_rate=learning_rate),
                        loss='binary_crossentropy',
                        metrics=['accuracy'])

        return model

def build_model_regressor_points(hp, input_shape):
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))
    
    # Hyperparameter search space for hidden layers
    for i in range(hp.Int('num_layers', min_value=1, max_value=5)):
        model.add(layers.Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=256, step=32),
                               activation='relu'))
    
    model.add(layers.Dense(2, activation='linear'))  # Output layer for two scores
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model

class deepCfb():
    def __init__(self):
        print('instantiate deepCfb class')

    def str_manipulations(self,df):
        #game score 
        df['score'] = df['game_result'].str.replace('W','')
        df['score'] = df['score'].str.replace('L','')
        df['score'] = df['score'].str.replace('(','')
        df['score'] = df['score'].str.replace(')','')
        df['score'] = df['score'].str.split('-').str[0]
        df['score'] = df['score'].str.replace('-','')

        #game result
        df['game_result'].loc[df['game_result'].str.contains('W')] = 'W'
        df['game_result'].loc[df['game_result'].str.contains('L')] = 'L'
        df['game_result'] = df['game_result'].replace({'W': 1, 'L': 0})

        return df

    def split(self):
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

        # fig = plt.figure(figsize=(15, 15))
        # self.all_data.hist(ax=fig.gca())
        # plt.tight_layout()
        # # Show the plot
        # plt.show()

        for col in self.all_data.columns:
            if 'Unnamed' in col:
                self.all_data.drop(columns=col,inplace=True)
        #split 
        self.classifier_drop = ['game_result','game_loc']
        self.y = self.all_data['game_result']
        self.x = self.all_data.drop(columns=self.classifier_drop)
        print(f'data length: {len(self.all_data)}')
        print(f'number of features: {len(self.x.columns)}')
        self.manual_comp = len(self.x.columns) - 2
        #Standardize
        self.scaler = MinMaxScaler(feature_range=(0,1))
        X_std = self.scaler.fit_transform(self.x)
        #FA
        self.fa = FactorAnalysis(n_components=self.manual_comp)
        X_fa = self.fa.fit_transform(X_std)
        self.x_data = DataFrame(X_fa, columns=[f'FA{i}' for i in range(1, self.manual_comp+1)])
        #Do not use FA
        # self.x_data = X_std

        #split data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_data, self.y, train_size=0.8)

    def dnn_regressor(self):
        all_data = read_csv(join(getcwd(),'all_data.csv'))
        all_data = concat([all_data, read_csv(join(getcwd(),'all_data_2023.csv'))])
        
        for col in all_data.columns:
            if 'Unnamed' in col:
                all_data.drop(columns=col,inplace=True)

        all_data[['team_1_score', 'team_2_score']] = all_data['game_result'].str.extract(r'(\d+)-(\d+)').astype(int)
        self.regression_drop = ['team_1_score', 'team_2_score','game_result','game_loc']
        X = all_data.drop(columns=self.regression_drop)
        y = all_data[['team_1_score', 'team_2_score']]
        
        self.regress_scaler = MinMaxScaler(feature_range=(0,1))
        X_std = self.regress_scaler.fit_transform(X)
        #FA
        self.regress_fa = FactorAnalysis(n_components=self.manual_comp)
        X_fa = self.regress_fa.fit_transform(X_std)
        if not exists('keras_regressor.h5'):
                
            x_final = DataFrame(X_fa, columns=[f'FA{i}' for i in range(1, self.manual_comp+1)])
            
            X_train, X_test, y_train, y_test = train_test_split(x_final, y, test_size=0.2, random_state=42)

            # Create a tuner
            input_shape = X_train.shape[1]
            tuner = RandomSearch(
                lambda hp: build_model_regressor_points(hp, input_shape),  # Pass input_shape to the build_model function
                objective='val_mae',#Objective("mae", direction="min"),
                max_trials=50,
                directory='regressor'
            )
            tuner.search(X_train, y_train, validation_data=(X_test, y_test), epochs=75)
            best_model = tuner.get_best_models(num_models=1)[0]
            best_model.fit(X_train, y_train, epochs=100, batch_size=128, verbose=2,
                            validation_data=(X_test, y_test))
        
            best_model.save('keras_regressor.h5')

    def dnn_classifier(self):
        if exists('keras_classifier.h5'):
            self.dnn_class = keras.models.load_model('keras_classifier.h5')
        else:
            #model accuracy: 97.3% validation
            # optimizer = keras.optimizers.Adam(learning_rate=0.001,
            #                                 #   kernel_regularizer=regularizers.l2(0.001)
            #                                   )
            # self.dnn_class = keras.Sequential([
            #         layers.Dense(48, input_shape=(self.x_train.shape[1],)),
            #         layers.LeakyReLU(alpha=0.2),
            #         layers.BatchNormalization(),
            #         layers.Dropout(0.2),
            #         layers.Dense(44),
            #         layers.LeakyReLU(alpha=0.2),
            #         layers.BatchNormalization(),
            #         layers.Dropout(0.2),
            #         layers.Dense(40),
            #         layers.LeakyReLU(alpha=0.2),
            #         layers.BatchNormalization(),
            #         layers.Dropout(0.2),
            #         layers.Dense(36),
            #         layers.LeakyReLU(alpha=0.2),
            #         layers.BatchNormalization(),
            #         layers.Dropout(0.2),
            #         layers.Dense(32),
            #         layers.LeakyReLU(alpha=0.2),
            #         layers.BatchNormalization(),
            #         layers.Dropout(0.2),
            #         layers.Dense(28),
            #         layers.LeakyReLU(alpha=0.2),
            #         layers.BatchNormalization(),
            #         layers.Dropout(0.2),
            #         layers.Dense(1, activation='sigmoid')
            #     ])
            # self.dnn_class.compile(optimizer=optimizer,
            #     loss='binary_crossentropy',
            #     metrics=['accuracy'])
            # print('Training Classifier')
            # self.dnn_class.summary()
            # #run this to see the tensorBoard: tensorboard --logdir=./logs
            # early_stop = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)
            # self.dnn_class.fit(self.x_train,self.y_train,epochs=500, batch_size=128, verbose=2,
            #                         validation_data=(self.x_test,self.y_test),callbacks=[early_stop]) 
            # self.dnn_class.save('keras_classifier.h5')
            tuner = RandomSearch(
                build_classifier,
                objective='val_accuracy',
                max_trials=125,  # Number of combinations to try
                directory='classifier',  # Directory to store the results
                project_name='classifier_project'
            )

            # Perform the hyperparameter search
            early_stop = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1)
            tuner.search(self.x_train, self.y_train, 
                        epochs=100, batch_size=128, 
                        validation_data=(self.x_test, self.y_test),
                        callbacks=[early_stop]) 

            #get best model from search
            # early_stop = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
            # best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            # best_model = tuner.hypermodel.build(best_hps) 
            best_model = tuner.get_best_models(1)[0]
            best_model.fit(self.x_train, self.y_train, epochs=100, batch_size=128, verbose=2,
                        validation_data=(self.x_test, self.y_test),
                        callbacks=[early_stop])
            best_model.save('keras_classifier.h5')
            validation_loss, validation_accuracy = best_model.evaluate(self.x_test, self.y_test)
            print(f'Validation Loss: {validation_loss}')
            print(f'Validation Accuracy: {validation_accuracy}')

            # Get the best hyperparameters and build the final model
            # best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            # best_model = tuner.hypermodel.build(best_hps)

            # best_model.fit(self.x_train, self.y_train, epochs=100, batch_size=128, verbose=2,
            #             validation_data=(self.x_test, self.y_test),
            #             callbacks=[early_stop])

            # # Save the best model
            # best_model.save('keras_classifier.h5')

    def xgb_class(self):
            if not exists("classifier_xgb.pkl"):
                param_grid = {
                    'n_estimators': [300, 400, 500],
                    'max_depth': [None, 5, 10, 20],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'reg_alpha': [0, 0.1],
                    'tree_method': ['auto', 'exact', 'approx'],
                }

                # Train the XGBoost model
                # Create the GridSearchCV object
                grid_search = GridSearchCV(estimator=XGBRFClassifier(), 
                                        param_grid=param_grid, 
                                        cv=3, n_jobs=1, 
                                        verbose=2,
                                        scoring='accuracy')

                # Fit the GridSearchCV object to the training data
                grid_search.fit(self.x_train, self.y_train,
                                eval_set=[(self.x_test, self.y_test)], 
                                # early_stopping_rounds=100, 
                                verbose=True)

                # Print the best parameters and best score
                print("Best Parameters Classifier: ", grid_search.best_params_)
                print("Best Score Classifier: ", grid_search.best_score_)
                with open('classifier_xgb.pkl', 'wb') as file:
                        dump(grid_search, file)

    def random_forest_class(self):
        if not exists("classifier_rf.pkl"):
            param_grid = {
                'n_estimators': [300, 400, 500],
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
            }

            # Train the Random Forest model
            grid_search = GridSearchCV(estimator=RandomForestClassifier(), 
                                    param_grid=param_grid, 
                                    cv=3, n_jobs=5, 
                                    verbose=2,
                                    scoring='accuracy')

            # Fit the GridSearchCV object to the training data
            grid_search.fit(self.x_train, self.y_train)

            # Print the best parameters and best score
            print("Best Parameters Classifier: ", grid_search.best_params_)
            print("Best Score Classifier: ", grid_search.best_score_)
            with open('classifier_rf.pkl', 'wb') as file:
                dump(grid_search, file)


    
    def logistic_regression_class(self):
        if not exists("classifier_logistic_regression.pkl"):
            param_grid = {
                'penalty': ['l1', 'l2'],
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
            }

            # Train the Logistic Regression model
            # Create the GridSearchCV object
            grid_search = GridSearchCV(estimator=LogisticRegression(), 
                                    param_grid=param_grid, 
                                    cv=3, n_jobs=1, 
                                    verbose=2,
                                    scoring='accuracy')

            # Fit the GridSearchCV object to the training data
            grid_search.fit(self.x_train, self.y_train)

            # Print the best parameters and best score
            print("Best Parameters Classifier: ", grid_search.best_params_)
            print("Best Score Classifier: ", grid_search.best_score_)
            with open('classifier_logistic_regression.pkl', 'wb') as file:
                dump(grid_search, file)
   
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
        x_train, x_test, y_train, y_test = train_test_split(x_regress, y_regress, train_size=0.8)

        #linear regression
        if not exists('feature_linear_regression.pkl'):
            lin_model = LinearRegression().fit(x_train,y_train)
            y_pred = lin_model.predict(x_test)
            y_test_np = y_test.to_numpy()
            mse_error = mean_squared_error(y_test_np,y_pred)
            print(f'Linear Regression MSE: {mse_error}')
            with open('feature_linear_regression.pkl', 'wb') as file:
                    dump(lin_model, file)

        #DNN feature regressor
        if exists('feature_dnn_classifier.h5'):
            print('load trained feature regression model')
            self.model_feature_regress_model = keras.models.load_model('feature_dnn_classifier.h5')
        else:
            #FIND BEST PARAMETERS
            tuner = RandomSearch(
                lambda hp: create_model_classifier(hp,self.manual_comp),
                objective='val_loss',
                max_trials=20,
                directory='tuner_results_classifier',
                project_name='model_tuning')
            tuner.search_space_summary()
            tuner.search(x_train, y_train, validation_data=(x_test, y_test), epochs=120)

            # Get the best model and summary of the best hyperparameters
            best_model = tuner.get_best_models(num_models=1)[0]
            best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
            best_model.summary()
            hyperparams = best_hyperparameters.values
            print(hyperparams)
            best_model.save(f"feature_dnn_classifier.h5")
        
        if not exists('feature_random_forest.pkl'):
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
            with open('feature_random_forest.pkl', 'wb') as file:
                    dump(grid_search, file)

        #MLP
        if not exists('feature_mlp_model.pkl'):
            param_grid_mlp = {
                'hidden_layer_sizes': [(32, 32, 32), (64,), (32, 32), (64, 64), (64, 64, 64), (128,)],
                'activation': ['relu', 'tanh'],
                'solver': ['adam'],
                'alpha': [0.0001, 0.001, 0.01],
                'max_iter': [100, 200, 300, 400, 500, 600]
            }

            # Create the GridSearchCV object for MLPRegressor
            grid_search_mlp = GridSearchCV(estimator=MLPRegressor(), 
                                        param_grid=param_grid_mlp, 
                                        cv=5, 
                                        n_jobs=10, 
                                        verbose=3,
                                        scoring='neg_mean_squared_error')

            # Fit the GridSearchCV object to the training data
            grid_search_mlp.fit(x_train, y_train)

            # Print the best parameters and best score (RMSE)
            print("Best Parameters (MLP): ", grid_search_mlp.best_params_)
            print("Best Explained Variance Score (MLP): ", grid_search_mlp.best_score_)  # We used greater_is_better=False, so we need to negate the score

            # Save the trained model to a file
            with open('feature_mlp_model.pkl', 'wb') as file:
                    dump(grid_search_mlp, file)
        
        #XGB model
        if not exists('feature_xgb.pkl'):
            param_grid = {
                    'n_estimators': [300, 400, 500],
                    'max_depth': [None, 5, 10, 20],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'reg_alpha': [0, 0.1],
                    'tree_method': ['auto', 'exact', 'approx'],
                }

            # Train the XGBoost model
            # Create the GridSearchCV object
            grid_search = GridSearchCV(estimator=XGBRFRegressor(), 
                                       param_grid=param_grid, 
                                       cv=3, n_jobs=1, 
                                       verbose=2,
                                       scoring='neg_mean_squared_error')

            # Fit the GridSearchCV object to the training data
            grid_search.fit(x_train, y_train,
                            eval_set=[(x_test, y_test)], 
                            # early_stopping_rounds=100, 
                            verbose=False)

            # Print the best parameters and best score
            print("Best Parameters: ", grid_search.best_params_)
            print("Best Score: ", grid_search.best_score_)
            with open('feature_xgb.pkl', 'wb') as file:
                    dump(grid_search, file)

        #CatBoostRegressor
        # if not exists('feature_catboost.pkl'):
        #     param_grid = {
        #         'iterations': [100, 200, 300],
        #         'learning_rate': [0.01, 0.1, 0.2],
        #         'depth': [6, 8, 10],
        #         'l2_leaf_reg': [0.1, 1, 10],
        #         'random_strength': [0.1, 0.5, 1]
        #     }


        #     # Train the CatBoostRegressor model
        #     # Create the GridSearchCV object
        #     grid_search = GridSearchCV(estimator=CatBoostRegressor(), 
        #                             param_grid=param_grid, 
        #                             cv=3, n_jobs=1, 
        #                             verbose=2,
        #                             scoring='neg_mean_squared_error')  # Note the negative sign for neg_mean_squared_error

        #     # Fit the GridSearchCV object to the training data
        #     print(y_train)
        #     input()
        #     grid_search.fit(x_train, y_train,
        #                     eval_set=[(x_test, y_test)], 
        #                     # early_stopping_rounds=100, 
        #                     verbose=False)

        #     # Print the best parameters and best score
        #     print("Best Parameters: ", grid_search.best_params_)
        #     print("Best Score: ", grid_search.best_score_)
            
        #     # Save the best model using joblib
        #     with open('feature_catboost.pkl', 'wb') as file:
        #         dump(grid_search, file)

    def forecast_features(self,game_data):
        #standardize and PCA
        X_std_1 = self.scaler.transform(game_data)
        X_pca_1 = self.fa.transform(X_std_1)
        team_1_df2023 = DataFrame(X_pca_1, columns=[f'FA{i}' for i in range(1, self.manual_comp+1)])#self.pca.n_components_+1
        team_df_forecast_last = team_1_df2023.iloc[-1:] #last game
        try:
            team_df_forecast_second = team_1_df2023.iloc[-2] #2nd to last game
        except:
             team_df_forecast_second = nan
        return team_df_forecast_last, team_df_forecast_second, team_1_df2023
    
    def test_forecast(self):
        #all teams
        # teams_list = get_teams_year(2015,2023)

        #Select certain teams
        with open('teams_played_this_week.txt','r') as file:
             content = file.read()
        teams_list = content.split("\n")
        teams_list = [string for string in teams_list if string.strip() != ""]

        save_betting_teams = []
        model = keras.models.load_model('keras_classifier.h5')
        feature_model = keras.models.load_model('feature_dnn_classifier.h5')
        with open('feature_linear_regression.pkl', 'rb') as file:
            lin_model = load(file)
        # with open('feature_xgb_model.pkl', 'rb') as file:
        #         xgb_model = load(file)
        with open('feature_random_forest.pkl', 'rb') as file:
                rf_model = load(file)
        with open('feature_mlp_model.pkl', 'rb') as file:
                mlp_model = load(file)
        # with open('feature_catboost.pkl', 'rb') as file:
        #         cat_model = load(file)
        with open('feature_xgb.pkl', 'rb') as file:
                xgb_model = load(file)
        count_teams = 1
        lin_out = 0
        rf_out = 0
        xgb_out = 0
        cat_out = 0
        mlp_out = 0
        dnn_out = 0
        esnemble_out = 0
        rolling_out_2 = 0
        rolling_out_3 = 0
        save_over_teams = []

        for abv in tqdm(teams_list):
            try:
                # if keyboard.is_pressed('q') and keyboard.is_pressed('w'):
                #     print('Exiting for loop early')
                #     break
                print('')#tqdm thing
                print(f'current team: {abv}')
                # team = all_teams(abv)
                str_combine = 'https://www.sports-reference.com/cfb/schools/' + abv.lower() + '/' + str(2023) + '/gamelog/'
                output = html_to_df_web_scrape(str_combine,abv.lower(),2023)

                #game result
                output = self.str_manipulations(output)
                final_data = output.replace(r'^\s*$', nan, regex=True) #replace empty string with NAN

                #Actual outcomes
                game_result_series = final_data['game_result']
                final_data.drop(columns=self.classifier_drop, inplace=True)

                #standardize and FA
                _, df_forecast_second, final_data_manipulated = self.forecast_features(final_data)
                ground_truth = game_result_series.iloc[-1]
                feature_data = df_forecast_second.to_numpy().reshape(1, -1)

                #running median calculation
                rolling_features_2 = final_data_manipulated.rolling(2).median().iloc[-1:]
                # rolling_features_3 = final_data_manipulated.rolling(3).median().iloc[-1:]

                #Feature prediction
                next_game_features_rf = rf_model.predict(feature_data)
                next_game_features_dnn = feature_model.predict(feature_data)
                next_game_features_mlp = mlp_model.predict(feature_data)
                next_game_features_lin = lin_model.predict(feature_data)
                # next_game_features_cat = cat_model.predict(feature_data)
                next_game_features_xgb = xgb_model.predict(feature_data)

                #multi-learning output manipulation
                dnn_list = []
                for val in next_game_features_dnn:
                    dnn_list.append(val[0][0])
                dnn_list = array(dnn_list)
                dnn_list = reshape(dnn_list, (1,len(dnn_list)))

                #predict classification
                prediction_median_rf = model.predict(next_game_features_rf)
                prediction_median_dnn = model.predict(dnn_list)
                prediction_median_mlp = model.predict(next_game_features_mlp)
                prediction_median_lin = model.predict(next_game_features_lin)
                prediction_median_rolling_2 = model.predict(rolling_features_2)
                # prediction_median_rolling_3 = model.predict(rolling_features_3)
                # prediction_median_cat = model.predict(next_game_features_cat)
                prediction_median_xgb = model.predict(next_game_features_xgb)

                #check if outcome is above 0.5
                if prediction_median_rf[0][0] > 0.5:
                    result_median_rf = 1
                else:
                    result_median_rf = 0
                if prediction_median_dnn[0][0] > 0.5:
                    result_median_dnn = 1
                else:
                    result_median_dnn = 0
                if prediction_median_mlp[0][0] > 0.5:
                    result_median_mlp = 1
                else:
                    result_median_mlp = 0
                if prediction_median_lin[0][0] > 0.5:
                    result_median_lin= 1
                else:
                    result_median_lin = 0
                if prediction_median_rolling_2[0][0] > 0.5:
                    result_median_roll_2 = 1
                else:
                    result_median_roll_2 = 0
                # if prediction_median_rolling_3[0][0] > 0.5:
                #     result_median_roll_3 = 1
                # else:
                #     result_median_roll_3 = 0
                # if prediction_median_cat[0][0] > 0.5:
                #     result_median_cat = 1
                # else:
                #     result_median_cat = 0
                if prediction_median_xgb[0][0] > 0.5:
                    result_median_xgb = 1
                else:
                    result_median_xgb = 0

                #ensemble output
                ensemble = result_median_mlp + result_median_rf + result_median_dnn + result_median_lin

                #all models
                if ensemble >= 3:
                    result_game = 1
                else:
                    result_game = 0

                #individual models
                if int(ground_truth) == result_median_lin:
                    lin_out += 1
                if int(ground_truth) == result_median_rf:
                    rf_out += 1
                if int(ground_truth) == result_median_mlp:
                    mlp_out += 1
                if int(ground_truth) == result_median_dnn:
                    dnn_out += 1
                if int(ground_truth) == result_game:
                    esnemble_out += 1
                if int(ground_truth) == result_median_roll_2:
                    rolling_out_2 += 1
                # if int(ground_truth) == result_median_roll_3:
                #     rolling_out_3 += 1
                # if int(ground_truth) == result_median_cat:
                #     cat_out += 1
                if int(ground_truth) == result_median_xgb:
                    xgb_out += 1

                print('=======================================')
                print(f'Prediction: {result_game} vs. Actual: {int(ground_truth)}')
                print('=======================================')
                print(f'Ensemble Accuracy out of {count_teams} teams: {esnemble_out / count_teams}')
                print(f'DNN Accuracy out of {count_teams} teams: {dnn_out / count_teams}')
                print(f'LinRegress Accuracy out of {count_teams} teams: {lin_out / count_teams}')
                print(f'RandomForest Accuracy out of {count_teams} teams: {rf_out / count_teams}')
                print(f'Rolling median 2 Accuracy out of {count_teams} teams: {rolling_out_2 / count_teams}')
                # print(f'Rolling median 3 Accuracy out of {count_teams} teams: {rolling_out_3 / count_teams}')
                print(f'MLP Accuracy out of {count_teams} teams: {mlp_out / count_teams}')
                # print(f'CAT Accuracy out of {count_teams} teams: {cat_out / count_teams}')
                print(f'XGB Accuracy out of {count_teams} teams: {xgb_out / count_teams}')
                save_over_teams.append(esnemble_out / count_teams)

                #plot correct over teams
                x_data = [i for i in range(count_teams)]
                if count_teams != 1:
                    plt.plot(x_data,save_over_teams,color='tab:blue',marker='*')
                    plt.xlabel('team count')
                    plt.ylabel('porportion correct')
                    plt.show(block=False)
                    plt.pause(0.1)
                count_teams += 1
            except Exception as e:
                print(f'NO data found for {abv}, Error: {e}')
        
    def test_point_forecast(self):
         #Select certain teams
        with open('teams_played_this_week.txt','r') as file:
             content = file.read()
        teams_list = content.split("\n")
        teams_list = [string for string in teams_list if string.strip() != ""]

        save_betting_teams = []
        model = keras.models.load_model('keras_regressor.h5')

        for abv in tqdm(teams_list):
            print('')#tqdm thing
            print(f'current team: {abv}')
            # team = all_teams(abv)
            str_combine = 'https://www.sports-reference.com/cfb/schools/' + abv.lower() + '/' + str(2023) + '/gamelog/'
            all_data = html_to_df_web_scrape(str_combine,abv.lower(),2023)

            team_1_actual, team_1_predict = [], []
            team_2_actual, team_2_predict = [], []
            for col in all_data.columns:
                if 'Unnamed' in col:
                    all_data.drop(columns=col,inplace=True)

            all_data[['team_1_score', 'team_2_score']] = all_data['game_result'].str.extract(r'(\d+)-(\d+)').astype(int)
        
            X = all_data.drop(columns=self.regression_drop)
            y = all_data[['team_1_score', 'team_2_score']]

            X_std = self.regress_scaler.transform(X)
            X_fa = self.regress_fa.transform(X_std)
        
            x_final = DataFrame(X_fa, columns=[f'FA{i}' for i in range(1, self.manual_comp+1)])

            rolling_features_2 = x_final.rolling(2).median().iloc[-2:-1]

            pts_output = model.predict(rolling_features_2)

            team_1_out, team_2_out = pts_output[0][0], pts_output[0][1]
            team_1_actual.append(y['team_1_score'].iloc[-1])
            team_1_predict.append(pts_output[0][0])
            team_2_actual.append(y['team_2_score'].iloc[-1])
            team_2_predict.append(pts_output[0][1])
            
            print(f'MAPE for {abv}')
            print(pts_output)
            print(y.iloc[-1])
        print(f'team1 mape: {mean_absolute_percentage_error(team_1_actual,team_1_predict)}')
        print(f'team2 mape: {mean_absolute_percentage_error(team_2_actual,team_2_predict)}')
    
    def predict_teams(self):
        #Read in models
        model = keras.models.load_model('keras_classifier.h5')
        feature_model = keras.models.load_model('feature_dnn_classifier.h5')
        with open('feature_random_forest.pkl', 'rb') as file:
                rf_model = load(file)
        with open('feature_linear_regression.pkl', 'rb') as file:
                lin_model = load(file)
        with open('feature_mlp_model.pkl', 'rb') as file:
                mlp_model = load(file)
        with open('classifier_rf.pkl', 'rb') as file:
                rf_class  = load(file)
        with open('classifier_logistic_regression.pkl', 'rb') as file:
                log_class  = load(file)

        #predict two teams
        while True:
            try:
                self.team_1 = input('team_1: ')
                if self.team_1 == 'exit':
                    break
                self.team_2 = input('team_2: ')

                str_combine = 'https://www.sports-reference.com/cfb/schools/' + self.team_1 + '/' + str(2023) + '/gamelog/'
                team_1_df  = html_to_df_web_scrape(str_combine,self.team_1,2023)

                str_combine = 'https://www.sports-reference.com/cfb/schools/' + self.team_2 + '/' + str(2023) + '/gamelog/'
                team_2_df  = html_to_df_web_scrape(str_combine,self.team_2,2023)

                team_1_df = self.str_manipulations(team_1_df)
                team_2_df = self.str_manipulations(team_2_df)
                team_1_df_final = team_1_df.replace(r'^\s*$', nan, regex=True) #replace empty string with NAN  
                team_2_df_final = team_2_df.replace(r'^\s*$', nan, regex=True) #replace empty string with NAN

                #drop game result
                team_1_df_final.drop(columns=self.classifier_drop, inplace=True)
                team_2_df_final.drop(columns=self.classifier_drop, inplace=True)

                #Feature regression
                forecast_team_1, _, team_1_df_final_1 = self.forecast_features(team_1_df_final)
                forecast_team_2, _, team_1_df_final_2 = self.forecast_features(team_2_df_final)

                #running median calculation
                rolling_features_1 = team_1_df_final_1.rolling(2).median().iloc[-1:]
                rolling_features_2 = team_1_df_final_2.rolling(2).median().iloc[-1:]

                #Feature prediction
                next_game_features_rf_1 = rf_model.predict(forecast_team_1)
                next_game_features_dnn_1 = feature_model.predict(forecast_team_1)
                next_game_features_lin_1 = lin_model.predict(forecast_team_1)
                next_game_features_mlp_1 = mlp_model.predict(forecast_team_1)

                next_game_features_rf_2 = rf_model.predict(forecast_team_2)
                next_game_features_dnn_2 = feature_model.predict(forecast_team_2)
                next_game_features_lin_2 = lin_model.predict(forecast_team_2)
                next_game_features_mlp_2 = mlp_model.predict(forecast_team_2)

                #multi-learning output manipulation
                dnn_list_1 = []
                for val in next_game_features_dnn_1:
                    dnn_list_1.append(val[0][0])
                dnn_list_1 = array(dnn_list_1)
                dnn_list_1 = reshape(dnn_list_1, (1,len(dnn_list_1)))

                dnn_list_2 = []
                for val in next_game_features_dnn_2:
                    dnn_list_2.append(val[0][0])
                dnn_list_2 = array(dnn_list_2)
                dnn_list_2 = reshape(dnn_list_2, (1,len(dnn_list_2)))

                #predict - classifier 
                prediction_median_rf_1 = model.predict(next_game_features_rf_1)
                prediction_median_dnn_1 = model.predict(dnn_list_1)
                prediction_median_lin_1 = model.predict(next_game_features_lin_1)
                prediction_median_mlp_1 = model.predict(next_game_features_mlp_1)
                prediction_median_1 = model.predict(rolling_features_1)
                #classifier rf
                prediction_median_rf_1_rf = rf_class.predict(next_game_features_rf_1)
                prediction_median_dnn_1_rf = rf_class.predict(dnn_list_1)
                prediction_median_lin_1_rf = rf_class.predict(next_game_features_lin_1)
                prediction_median_mlp_1_rf = rf_class.predict(next_game_features_mlp_1)
                prediction_median_1_rf = rf_class.predict(rolling_features_1)
                #classifier logistic regression
                prediction_median_rf_1_log_class = log_class.predict(next_game_features_rf_1)
                prediction_median_dnn_1_log_class = log_class.predict(dnn_list_1)
                prediction_median_lin_1_log_class = log_class.predict(next_game_features_lin_1)
                prediction_median_mlp_1_log_class = log_class.predict(next_game_features_mlp_1)
                prediction_median_1_log_class = log_class.predict(rolling_features_1)
                
                #DNN
                prediction_median_rf_2 = model.predict(next_game_features_rf_2)
                prediction_median_dnn_2 = model.predict(dnn_list_2)
                prediction_median_lin_2 = model.predict(next_game_features_lin_2)
                prediction_median_mlp_2 = model.predict(next_game_features_mlp_2)
                prediction_median_2 = model.predict(rolling_features_2)
                #classifier rf
                prediction_median_rf_2_rf = rf_class.predict(next_game_features_rf_2)
                prediction_median_dnn_2_rf = rf_class.predict(dnn_list_2)
                prediction_median_lin_2_rf = rf_class.predict(next_game_features_lin_2)
                prediction_median_mlp_2_rf = rf_class.predict(next_game_features_mlp_2)
                prediction_median_2_rf = rf_class.predict(rolling_features_2)
                #classifier logistic regression
                prediction_median_rf_2_log_class = log_class.predict(next_game_features_rf_2)
                prediction_median_dnn_2_log_class = log_class.predict(dnn_list_2)
                prediction_median_lin_2_log_class = log_class.predict(next_game_features_lin_2)
                prediction_median_mlp_2_log_class = log_class.predict(next_game_features_mlp_2)
                prediction_median_2_log_class = log_class.predict(rolling_features_2)

                num_true_conditions_team1 = 0
                num_true_conditions_team2 = 0

                # Check conditions for team 1
                if round(prediction_median_rf_1[0][0]*100, 2) > 50:
                    num_true_conditions_team1 += 1

                if round(prediction_median_dnn_1[0][0]*100, 2) > 50:
                    num_true_conditions_team1 += 1
                
                if round(prediction_median_lin_1[0][0]*100, 2) > 50:
                    num_true_conditions_team2 += 1
                
                if round(prediction_median_mlp_1[0][0]*100, 2) > 50:
                    num_true_conditions_team2 += 1

                # Check conditions for team 2
                if round(prediction_median_lin_2[0][0]*100, 2) > 50:
                    num_true_conditions_team2 += 1
                
                if round(prediction_median_mlp_2[0][0]*100, 2) > 50:
                    num_true_conditions_team2 += 1

                if round(prediction_median_rf_2[0][0]*100, 2) > 50:
                    num_true_conditions_team2 += 1

                if round(prediction_median_dnn_2[0][0]*100, 2) > 50:
                    num_true_conditions_team2 += 1

                if num_true_conditions_team2 >= 3 and num_true_conditions_team1 >= 3:
                    print(Fore.GREEN + Style.BRIGHT +f'{self.team_1} Predicted Winning Probability: {round(prediction_median_lin_1[0][0]*100,2)}% LinRegress, {round(prediction_median_mlp_1[0][0]*100,2)}% MLP, {round(prediction_median_rf_1[0][0]*100,2)}% RF, {round(prediction_median_dnn_1[0][0]*100,2)}% DNN' + Style.RESET_ALL)
                    print(Fore.GREEN + Style.BRIGHT + f'{self.team_2} Predicted Winning Probability: {round(prediction_median_lin_2[0][0]*100,2)}% LinRegress, {round(prediction_median_mlp_2[0][0]*100,2)}% MLP {round(prediction_median_rf_2[0][0]*100,2)}% RF, {round(prediction_median_dnn_2[0][0]*100,2)}% DNN' + Style.RESET_ALL) 
                elif num_true_conditions_team2 < 3 and num_true_conditions_team1 < 3:
                    print(Fore.RED + Style.BRIGHT +f'{self.team_1} Predicted Winning Probability: {round(prediction_median_lin_1[0][0]*100,2)}% LinRegress, {round(prediction_median_mlp_1[0][0]*100,2)}% MLP, {round(prediction_median_rf_1[0][0]*100,2)}% RF, {round(prediction_median_dnn_1[0][0]*100,2)}% DNN' + Style.RESET_ALL)
                    print(Fore.RED + Style.BRIGHT + f'{self.team_2} Predicted Winning Probability: {round(prediction_median_lin_2[0][0]*100,2)}% LinRegress, {round(prediction_median_mlp_2[0][0]*100,2)}% MLP {round(prediction_median_rf_2[0][0]*100,2)}% RF, {round(prediction_median_dnn_2[0][0]*100,2)}% DNN' + Style.RESET_ALL)
                elif num_true_conditions_team1 >= 3:
                    print(Fore.GREEN + Style.BRIGHT +f'{self.team_1} Predicted Winning Probability: {round(prediction_median_lin_1[0][0]*100,2)}% LinRegress, {round(prediction_median_mlp_1[0][0]*100,2)}% MLP, {round(prediction_median_rf_1[0][0]*100,2)}% RF, {round(prediction_median_dnn_1[0][0]*100,2)}% DNN' + Style.RESET_ALL)
                    print(f'{self.team_2} Predicted Winning Probability: {round(prediction_median_lin_2[0][0]*100,2)}% LinRegress, {round(prediction_median_mlp_2[0][0]*100,2)}% MLP {round(prediction_median_rf_2[0][0]*100,2)}% RF, {round(prediction_median_dnn_2[0][0]*100,2)}% DNN')
                elif num_true_conditions_team2 >= 3:
                    print(f'{self.team_1} Predicted Winning Probability: {round(prediction_median_lin_1[0][0]*100,2)}% LinRegress, {round(prediction_median_mlp_1[0][0]*100,2)}% MLP, {round(prediction_median_rf_1[0][0]*100,2)}% RF, {round(prediction_median_dnn_1[0][0]*100,2)}% DNN')  
                    print(Fore.GREEN + Style.BRIGHT + f'{self.team_2} Predicted Winning Probability: {round(prediction_median_lin_2[0][0]*100,2)}% LinRegress, {round(prediction_median_mlp_2[0][0]*100,2)}% MLP {round(prediction_median_rf_2[0][0]*100,2)}% RF, {round(prediction_median_dnn_2[0][0]*100,2)}% DNN' + Style.RESET_ALL)
                
                #RF Classifier
                print(Fore.WHITE + Style.BRIGHT +f'{self.team_1} RF classifier Predicted Winning Probability: {prediction_median_rf_1_rf} RF, {prediction_median_dnn_1_rf} DNN, {prediction_median_lin_1_rf} LinRegress, {prediction_median_mlp_1_rf} MLP, {prediction_median_1_rf} Rolling average'+ Style.RESET_ALL)
                print(Fore.WHITE + Style.BRIGHT +f'{self.team_2} RF classifier Predicted Winning Probability: {prediction_median_rf_2_rf} RF, {prediction_median_dnn_2_rf} DNN, {prediction_median_lin_2_rf} LinRegress, {prediction_median_mlp_2_rf} MLP, {prediction_median_2_rf} Rolling average'+ Style.RESET_ALL)

                #Logistic Classifier
                print(Fore.WHITE + Style.BRIGHT +f'{self.team_1} Logistic classifier Predicted Winning Probability: {prediction_median_rf_1_log_class} RF, {prediction_median_dnn_1_log_class} DNN, {prediction_median_lin_1_log_class} LinRegress, {prediction_median_mlp_1_log_class} MLP, {prediction_median_1_log_class} Rolling average'+ Style.RESET_ALL)
                print(Fore.WHITE + Style.BRIGHT +f'{self.team_2} Logistic classifier Predicted Winning Probability: {prediction_median_rf_2_log_class} RF, {prediction_median_dnn_2_log_class} DNN, {prediction_median_lin_2_log_class} LinRegress, {prediction_median_mlp_2_log_class} MLP, {prediction_median_2_log_class} Rolling average'+ Style.RESET_ALL)

                #running averages
                if round(prediction_median_1[0][0]*100, 2) > round(prediction_median_2[0][0]*100, 2):
                    print(Fore.GREEN + Style.BRIGHT + f'{self.team_1} Predicted Winning Probability Rolling median (3): {round(prediction_median_1[0][0]*100, 2)} %' + Style.RESET_ALL)
                    print(f'{self.team_2} Predicted Winning Probability Rolling median (3): {round(prediction_median_2[0][0]*100, 2)} %')
                elif round(prediction_median_2[0][0]*100, 2) > round(prediction_median_1[0][0]*100, 2):
                    print(f'{self.team_1} Predicted Winning Probability Rolling median (3): {round(prediction_median_1[0][0]*100, 2)} %')
                    print(Fore.GREEN + Style.BRIGHT + f'{self.team_2} Predicted Winning Probability Rolling median (3): {round(prediction_median_2[0][0]*100, 2)} %' + Style.RESET_ALL)
            except Exception as e:
                 print(f'The error: {e}. Most likely {self.team_1} or {self.team_2} do not have data')
    def run_analysis(self):
        self.split()
        self.dnn_classifier()
        self.dnn_regressor()
        self.xgb_class()
        self.random_forest_class()
        self.logistic_regression_class()
        self.deep_learn_features()
        if argv[1] == 'test':
            self.test_point_forecast()
            self.test_forecast()
        self.predict_teams()

def main():
    deepCfb().run_analysis()

if __name__ == "__main__":
    main()