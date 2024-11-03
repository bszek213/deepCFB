#multiclass deep-learning on college football games
from pandas import read_csv, DataFrame, concat, io, to_numeric, io
from os.path import join, exists
from os import getcwd, mkdir, environ
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import FactorAnalysis, PCA, KernelPCA
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
import pickle
from colorama import Fore, Style
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from collect_augment_data import collect_two_teams
from numpy import nan, array, reshape, arange, random, zeros, argmax, isnan, shape, exp, var, save, load, concatenate, where, sort, cumsum
from sys import argv
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import shutil
import math
from gc import collect
from psutil import virtual_memory
from matplotlib.animation import FuncAnimation
from fitter import Fitter
from scipy.stats import norm, lognorm, beta, gamma, expon, uniform, weibull_min, weibull_max, pareto, t, chi2, triang, invgauss, genextreme, logistic, gumbel_r, gumbel_l, loggamma, powerlaw, rayleigh, laplace, cauchy
import joblib
from sklearn.impute import SimpleImputer
from umap import UMAP
from sklearn.feature_selection import SelectKBest, mutual_info_classif, SelectFromModel
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from collections import defaultdict
import numpy as np
import traceback

def check_ram_usage_txt(txtfile):
    ram_percent = virtual_memory().percent
    if ram_percent >= 95:
        with open(txtfile, 'w') as f:
                f.writelines('RAM Full\n')
        print(f"RAM is {ram_percent}%. Exit.")
        exit()
def check_ram_usage():
    ram_percent = virtual_memory().percent
    if ram_percent >= 92:
        print(f"RAM is {ram_percent}%. Exit.")
        exit()

def build_classifier(hp,input_shape):
    model = keras.Sequential()

    num_features = input_shape[1]
    print(f"Number of features: {num_features}")
    #number of layers
    num_layers = hp.Int('num_layers', min_value=1, max_value=10, step=1)

    #weight initialization
    kernel_initializer = hp.Choice('kernel_initializer', values=['glorot_uniform', 'he_uniform', 'random_normal'])

    #L2 regularization
    l2_reg = hp.Float('l2_reg', min_value=1e-6, max_value=1e-2, sampling='log')

    #batch normalization usage
    use_batch_norm = hp.Boolean('use_batch_norm')

    for i in range(num_layers):
        #the number of units in each layer
        units = hp.Int(f'units_layer_{i}', min_value=8, max_value=512, step=24)

        #the activation function
        activation = hp.Choice(f'activation_layer_{i}', values=['relu', 'tanh', 'sigmoid', 'swish', 'leaky_relu', 'elu', 
                                                                'selu', 'softplus', 'softsign', 'hard_sigmoid', 'gelu'])

        model.add(layers.Dense(units=units, activation=activation, 
                               kernel_initializer=kernel_initializer,
                               kernel_regularizer=regularizers.l2(l2_reg)))

        if use_batch_norm:
            model.add(layers.BatchNormalization())

        #dropout rate
        dropout_rate = hp.Float(f'dropout_layer_{i}', min_value=0.0, max_value=0.5, step=0.1)
        model.add(layers.Dropout(rate=dropout_rate))

    #the output layer with softmax activation for multi-class classification
    model.add(layers.Dense(2, activation='softmax'))

    #optimizer
    optimizer = hp.Choice('optimizer', values=['adam', 'rmsprop', 'sgd'])
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')

    #batch size
    batch_size = hp.Int('batch_size', min_value=16, max_value=128, step=16)

    #learning rate decay
    decay_steps = hp.Int('decay_steps', min_value=1000, max_value=10000, step=1000)
    decay_rate = hp.Float('decay_rate', min_value=0.1, max_value=0.9, step=0.1)

    #learning rate schedule
    lr_schedule = ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate
    )

    #gradient clipping
    clipnorm = hp.Float('clipnorm', min_value=0.1, max_value=1.0, step=0.1)

    if optimizer == 'adam':
        opt = Adam(learning_rate=lr_schedule, clipnorm=clipnorm)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=lr_schedule, clipnorm=clipnorm)
    else:
        #momentum for SGD
        momentum = hp.Float('momentum', min_value=0.0, max_value=0.9, step=0.1)
        opt = SGD(learning_rate=lr_schedule, momentum=momentum, clipnorm=clipnorm)

    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model, batch_size

#wrapper function to extract model and batch size
def build_classifier_with_batch_size(hp,input_shape):
    model, batch_size = build_classifier(hp,input_shape)
    hp.values['batch_size'] = batch_size
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

def calculate_proxy_elo(features, current_elo, game_result, k=32):
    pass_yds_diff = features['pass_yds'] - features['tot_yds_per_play_opp'] * features['tot_plays_opp']
    rush_yds_diff = features['rush_yds'] - features['tot_yds_per_play_opp'] * features['tot_plays_opp']
    turnovers_diff = features['turnovers']

    performance_score = (0.5 * pass_yds_diff + 0.3 * rush_yds_diff - 0.2 * turnovers_diff)
    expected_outcome = 1 / (1 + 10**((-performance_score) / 400))
    new_elo = current_elo + k * (game_result - expected_outcome)

    return new_elo

def add_opp(df_list_1,df_list_2):
    list_update_1 = []
    for team_1_df, team_2_df in zip(df_list_1,df_list_2):
        team_1_df_copy = team_1_df.copy()
        team_2_df_copy = team_2_df.copy()
        length_difference = len(team_1_df_copy) - len(team_2_df_copy)
        if length_difference > 0:
            team_1_df_copy = team_1_df_copy.iloc[length_difference:]
        elif length_difference < 0:
            team_2_df_copy = team_2_df_copy.iloc[-length_difference:]
        for col in [c for c in team_2_df_copy.columns if '_opp' not in c]:
            opp_col = col + '_opp'
            if opp_col in team_1_df_copy.columns:
                team_1_df_copy[opp_col] = team_2_df_copy[col]
        list_update_1.append(team_1_df_copy)
    return list_update_1

def bayesian_update(mu_prior, sigma_prior, new_data):
    for x in new_data:
        mu_post = (mu_prior / sigma_prior**2 + x / sigma_prior**2) / (1 / sigma_prior**2 + 1 / sigma_prior**2)
        sigma_post = np.sqrt(1 / (1 / sigma_prior**2 + 1 / sigma_prior**2))
        mu_prior = mu_post
        sigma_prior = sigma_post
    
    return mu_post, sigma_post

def str_manipulations(df):
    #extract outcomes and scores
    df['team_1_outcome'] = df['game_result'].apply(lambda x: 1 if x[0] == 'W' else 0)
    df['team_2_outcome'] = df['game_result'].apply(lambda x: 1 if x[0] == 'L' else 0)
    df['team_1_score'] = df['game_result'].str.extract(r'(\d+)-\d+').astype(int)
    df['team_2_score'] = df['game_result'].str.extract(r'\d+-(\d+)').astype(int)
    df2 = df.drop(columns=['game_result'])
    return df2

def bayes_calc(df_list_1,classifier_drop, selector,kpca,scaler):
    dict_prior = defaultdict(lambda: defaultdict(dict))
    dict_post = defaultdict(lambda: defaultdict(dict))
    for val in tqdm(range(0,len(df_list_1)-1)):
        # print(val)
        df_curr = df_list_1[val]
        df_fut = df_list_1[val+1]
        df_curr = str_manipulations(df_curr)
        df_fut = str_manipulations(df_fut)

        df_curr.drop(columns=classifier_drop, inplace=True)
        df_fut.drop(columns=classifier_drop, inplace=True)

        x_curr = selector.transform(kpca.transform(scaler.transform(df_curr)))
        df_fut = selector.transform(kpca.transform(scaler.transform(df_fut)))

        for i in range(x_curr.shape[1]):
            if val == 0:
                dict_prior[val][i]['mu'] = np.mean(x_curr[:,i])
                dict_prior[val][i]['std'] = np.std(x_curr[:,i])
            else:
                dict_prior[val][i]['mu'] = dict_post[val-1][i]['mu']
                dict_prior[val][i]['std'] = dict_post[val-1][i]['std']

            updated_results =  df_fut[:,i] #[35, 31, 28, 40]
            dict_post[val][i]['mu'], dict_post[val][i]['std'] = bayesian_update(dict_prior[val][i]['mu'], dict_prior[val][i]['std'], updated_results)
    return dict_post 

def monte_carlo_sim(final_dict_1_dn,dnn_class,n_simulations=10000):
    dict_data_monte = {}

    for key in list(final_dict_1_dn.keys()):
        mu = final_dict_1_dn[key]['mu']
        std = final_dict_1_dn[key]['std']
        dict_data_monte[key] = np.random.normal(loc=mu, scale=std * 5e20, size=n_simulations)
    
    df_monte = DataFrame(dict_data_monte)
    list_temp_1, list_temp_2 = [], []
    for index, row in tqdm(df_monte.iterrows()):
        row_array = row.values
        out = dnn_class.predict(row_array.reshape(1, -1), verbose=0)
        list_temp_1.append(out[0][0])
        list_temp_2.append(out[0][1])
    return list_temp_1, list_temp_2

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
        self.all_data = concat([self.all_data, read_csv(join(getcwd(),'all_data_2024.csv'))])

        for column in self.all_data.columns:
            if column != 'game_result':
                self.all_data[column] = to_numeric(self.all_data[column], errors='coerce')

        self.x_regress = read_csv(join(getcwd(),'x_feature_regression.csv')) 
        self.x_regress = concat([self.x_regress, read_csv(join(getcwd(),'x_feature_regression_2024.csv'))])

        self.y_regress = read_csv(join(getcwd(),'y_feature_regression.csv')) 
        self.y_regress = concat([self.y_regress, read_csv(join(getcwd(),'y_feature_regression_2024.csv'))])

        self.all_data = self.str_manipulations(self.all_data)
        self.x_regress = self.str_manipulations(self.x_regress)
        self.y_regress = self.str_manipulations(self.y_regress)

        self.classifier_drop = ['team_1_outcome','team_2_outcome','game_loc'
                                ] #'','team_1_score','team_2_score'

        self.y = self.all_data[['team_1_outcome','team_2_outcome']]

        self.x = self.all_data.drop(columns=self.classifier_drop)

        print(f'number of features: {len(self.x.columns)}')
        print(f'number of samples: {len(self.x)}')
        self.manual_comp = len(self.x.columns)

        #Ensemble Analysis
        data_path = 'data.npy'
        kernel_pca_model_path = join('processing_models','kernel_pca_model.joblib')
        pca_model_path = join('processing_models','pca_model.joblib')
        umapmodel_path = join('processing_models','umap_model.joblib')
        selector_model_path = join('processing_models', 'selector_model.joblib')
        fa_path = join('processing_models','fa_model.joblib')
        scaler_path = join('processing_models','scaler_model.joblib')
        if not exists(data_path):
            #Standardize
            self.scaler = StandardScaler() #MinMaxScaler(feature_range=(0,1))
            X_std = self.scaler.fit_transform(self.x)
            # self.pca = PCA(n_components=0.95)
            # X_pca_data = self.pca.fit_transform(X_std)

            # self.kpca = KernelPCA(n_components=int(self.pca.n_components_), kernel='rbf', n_jobs=1)
            self.kpca = KernelPCA(n_components=int(X_std.shape[1] * 0.95), kernel='rbf', n_jobs=1)
            # self.kpca = KernelPCA(n_components=int(X_std.shape[1] * 0.95), kernel='poly', degree=3, coef0=1, gamma=1, n_jobs=1)
            X_kpca_data = self.kpca.fit_transform(X_std)
            
            # self.umap_reducer = UMAP(n_components=int(self.pca.n_components_), random_state=42)
            # X_umap_data = self.umap_reducer.fit_transform(X_std)

            # self.fa = FactorAnalysis(n_components=int(self.pca.n_components_))
            # X_fa = self.fa.fit_transform(X_std)
            
            # X_combined = concatenate((X_pca_data, X_kpca_data, X_umap_data, X_fa), axis=1)

            #mututal information feature selection
            # mi_scores = mutual_info_classif(X_kpca_data, where(self.y['team_1_outcome'] == 1, 0, 1))
            # non_zero_mi_cols = mi_scores > 0
            # x_selected_features = X_kpca_data[:, non_zero_mi_cols]
            mi_scores = mutual_info_classif(X_kpca_data, where(self.y['team_1_outcome'] == 1, 0, 1))
            k = sum(mi_scores > 0) 
            self.selector = SelectKBest(score_func=mutual_info_classif, k=k)
            x_selected_features = self.selector.fit_transform(X_kpca_data, where(self.y['team_1_outcome'] == 1, 0, 1))
            print(f'data shape before mutual information: {X_kpca_data.shape}')
            print(f'data shape after mutual information: {x_selected_features.shape}')

            if not exists('processing_models'):
                mkdir('processing_models')
            save(data_path, x_selected_features)
            # joblib.dump(self.pca, pca_model_path)
            joblib.dump(self.kpca, kernel_pca_model_path)
            # joblib.dump(self.umap_reducer, umapmodel_path)
            # joblib.dump(self.fa, fa_path)
            joblib.dump(self.selector, selector_model_path)
            joblib.dump(self.scaler, scaler_path)
        else:
            x_selected_features = load(data_path)
            # self.fa = joblib.load(fa_path)
            # self.pca = joblib.load(pca_model_path)
            self.kpca = joblib.load(kernel_pca_model_path)
            # self.umap_reducer = joblib.load(umapmodel_path)
            self.selector = joblib.load(selector_model_path)
            self.scaler = joblib.load(scaler_path)
        
        self.manual_comp = x_selected_features.shape[1]
        
        self.x_data = DataFrame(x_selected_features, columns=[f'FA{i+1}' for i in range(x_selected_features.shape[1])])
        print('===================')
        print(f"Final data for training: {self.x_data.shape}")
        print(f"Final labels for training: {self.y.shape}")
        print('===================')
        num_columns = self.x_data.shape[1]
        grid_size = math.ceil(math.sqrt(num_columns))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        axes = axes.flatten()
        for i, col in enumerate(self.x_data.columns):
            axes[i].hist(self.x_data[col], bins=30, color='blue', alpha=0.7)
            axes[i].set_title(col)
        for i in range(num_columns, len(axes)):
            fig.delaxes(axes[i])
        plt.tight_layout()
        plt.savefig('all_histograms.png', dpi=300)
        plt.close()
        # binary_columns = self.x_data.columns[self.x_data.nunique() == 1]
        # self.x_data = self.x_data.drop(columns=binary_columns)

        #drop non-normal columns - removes columns that have no distribution (ie they are binary data) - Exploratory for now
        # self.non_normal_columns = []
        # for column in self.x_data.columns:
        #     stat, p = stats.shapiro(self.x_data[column])
        #     if p == 1:
        #         self.non_normal_columns.append(column)
        # self.x_data = self.x_data.drop(self.non_normal_columns, axis=1)

        with open('num_features.txt','w') as f:
            f.write(f'Number of features that the model will be trained on: {self.x_data.shape[1]}')

        #split data 75/15/10
        # self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_data, self.y, train_size=0.8)
        self.x_train, x_temp, self.y_train, y_temp = train_test_split(self.x_data, self.y, train_size=0.75, random_state=42)
        self.x_valid, self.x_test, self.y_valid, self.y_test = train_test_split(x_temp, y_temp, test_size=0.4, random_state=42)

        #add noise - I should tune this to find what the most optimal noise factor is 
        for col in self.x_train.columns:
            noise_factor = 0.015
            #gaussian noise, scaled by the column's standard deviation
            noise = noise_factor * random.normal(loc=0.0, scale=self.x_train[col].std(), size=self.x_train[col].shape)
            self.x_train[col] += noise
            # plt.hist(arrr,label='noise')
            # plt.hist(self.x_train[col],label='no_noise')
            # plt.title(f'{self.x_train.shape}')
            # plt.legend()
            # plt.show()

    def multiclass_class(self):
        if not exists('multiclass_models'):
            mkdir("multiclass_models")
        abs_path = join(getcwd(),'multiclass_models','keras_classifier_mc.h5')
        if exists(abs_path):
            self.dnn_class = keras.models.load_model(abs_path)
        else:
            # shutil.rmtree('classifier_multiclass', ignore_errors=True)
            input_shape = self.x_train.shape
            tuner = RandomSearch(
                lambda hp: build_classifier_with_batch_size(hp, input_shape),
                objective='val_accuracy',
                max_trials=125,
                directory='classifier_multiclass',
                project_name='classifier_multiclass_project',
                overwrite=False
            )

            early_stop = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1)
            tuner.search(self.x_train, self.y_train, 
                        epochs=200, batch_size=None, 
                        validation_data=(self.x_valid, self.y_valid),
                        callbacks=[early_stop]) 
            
            best_model = tuner.get_best_models(1)[0]
            #get best hyperparameters
            best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
            best_batch_size = best_hp.get('batch_size')
            with open('hyperparameters.txt', 'w') as f:
                for key, value in best_hp.values.items():
                    f.write(f'{key}: {value}\n')
                f.write(f'best_batch_size: {best_batch_size}\n')
            history = best_model.fit(self.x_train, self.y_train, 
                                     epochs=200, batch_size=int(best_batch_size), verbose=2,
                                     validation_data=(self.x_test, self.y_test),
                                     callbacks=[early_stop])
            best_model.save(abs_path)
            self.dnn_class = best_model
            test_loss, test_accuracy = best_model.evaluate(self.x_test, self.y_test)
            epochs = range(1, len(history.history['loss']) + 1)
            plt.figure(figsize=(14, 5))

            plt.subplot(1, 2, 1)
            plt.plot(epochs, history.history['loss'], label='Training Loss')
            plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
            plt.title(f'Training and Validation Loss: Test Loss: {test_loss}')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(epochs, history.history['accuracy'], label='Training Accuracy')
            plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy')
            plt.title(f'Training and Validation Accuracy: Test Accuracy: {test_accuracy}')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()

            plt.tight_layout()
            plt.savefig('Training.png',dpi=400)
            plt.close()

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
                max_trials=50,
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
                pickle.dump(lin_model, file)
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
                    pickle.dump(grid_search, file)
            self.feature_rf = grid_search
        else:
            with open(lin_abs_path, 'rb') as file:
                self.feature_rf = load(file)

    def test_forecast(self,teams_file='teams_played_this_week.txt'):
        with open(teams_file, 'r') as f:
                lines = f.readlines()
        idx = 0
        both_teams_out, one_team_out,count_teams = 0, 0, 0
        while idx < len(lines):
            line = lines[idx].strip()
            teams = line.split(',')
            self.team_1, self.team_2 = teams
            print(f'Currently making predictions for {self.team_1} vs. {self.team_2}')
            
            #team data
            team_1_df_2023 = collect_two_teams(f'https://www.sports-reference.com/cfb/schools/{self.team_1.lower()}/2023/gamelog/', self.team_1.lower(), 2023)
            team_1_df_2024 = collect_two_teams(f'https://www.sports-reference.com/cfb/schools/{self.team_1.lower()}/2024/gamelog/', self.team_1.lower(), 2024)
            team_1_df = concat([team_1_df_2023, team_1_df_2024])

            team_2_df_2023 = collect_two_teams(f'https://www.sports-reference.com/cfb/schools/{self.team_2.lower()}/2023/gamelog/', self.team_2.lower(), 2023)
            team_2_df_2024 = collect_two_teams(f'https://www.sports-reference.com/cfb/schools/{self.team_2.lower()}/2024/gamelog/', self.team_2.lower(), 2024)
            team_2_df = concat([team_2_df_2023, team_2_df_2024])

            #preprocess
            team_1_df = self.str_manipulations(team_1_df)
            team_2_df = self.str_manipulations(team_2_df)

            #get team outcomes
            if team_1_df['team_1_outcome'].iloc[-1] == 1:
                team_1_out = 1
                team_2_out = 0
            else:
                team_1_out = 0
                team_2_out = 1

            team_1_df.drop(columns=self.classifier_drop, inplace=True)
            team_2_df.drop(columns=self.classifier_drop, inplace=True)

            #extract everything except the last row
            team_1_df = team_1_df.iloc[:-1]
            team_2_df = team_2_df.iloc[:-1]

            #length mismatch
            length_difference = len(team_1_df) - len(team_2_df)
            if length_difference > 0:
                team_1_df = team_1_df.iloc[length_difference:]
            elif length_difference < 0:
                team_2_df = team_2_df.iloc[-length_difference:]

            #Monte Carlo simulations
            n_simulations = 20000
            all_probas_both_teams = zeros(2)
            all_probas_just_team_1 = zeros(2)

            #team 1
            team_1_df_copy = team_1_df.copy()
            team_2_df_copy = team_2_df.copy()

            for col in [c for c in team_2_df.columns if '_opp' not in c]:
                opp_col = col + '_opp'
                if opp_col in team_1_df_copy.columns:
                    team_1_df_copy[opp_col] = team_2_df_copy[col]
            team_1_df_copy['team_2_score'] = team_2_df_copy['team_1_score']

            #transformations and predictions
            X_std = self.scaler.transform(team_1_df_copy)
            X_fa = self.fa.transform(X_std)
            team_1_df_copy = DataFrame(X_fa, columns=[f'FA{i}' for i in range(1, self.manual_comp + 1)])
            team_1_df_copy.drop(self.non_normal_columns, axis=1, inplace=True)

            for _ in range(n_simulations):
                mc_sample = array([norm.rvs(loc=team_1_df_copy[col].mean(), scale=team_1_df_copy[col].std()*3) 
                                for col in team_1_df_copy.columns]).T
                probas = self.dnn_class.predict(mc_sample.reshape(1, -1),verbose=0)
                all_probas_both_teams += probas[0]
                all_probas_just_team_1 += probas[0]

            #team 2
            for col in [c for c in team_1_df.columns if '_opp' not in c]:
                opp_col = col + '_opp'
                if opp_col in team_2_df.columns:
                    team_2_df[opp_col] = team_1_df[col]
            team_2_df['team_2_score'] = team_1_df['team_1_score']

            X_std_t2 = self.scaler.transform(team_2_df)
            X_fa_t2 = self.fa.transform(X_std_t2)
            final_df_t2 = DataFrame(X_fa_t2, columns=[f'FA{i}' for i in range(1, self.manual_comp + 1)])
            final_df_t2.drop(self.non_normal_columns, axis=1, inplace=True)

            for _ in range(n_simulations):
                mc_sample = array([norm.rvs(loc=final_df_t2[col].mean(), scale=final_df_t2[col].std()*3) 
                                for col in final_df_t2.columns]).T
                probas = self.dnn_class.predict(mc_sample.reshape(1, -1),verbose=0)
                all_probas_both_teams += probas[0][::-1]  #flip probabilities

            #median probabilities
            median_probas_both_teams = all_probas_both_teams / (n_simulations * 2)
            median_probas_team_1 = all_probas_just_team_1 / (n_simulations)

            print(f'predicted probas: {median_probas_both_teams}')
            if median_probas_both_teams[0] > 0.5:
                team_1_both_result = 1
            else:
                team_1_both_result = 0

            # if median_probas_both_teams[1] > 0.5:
            #     team_2_both_result = 1
            # else:
            #     team_2_both_result = 0

            if median_probas_team_1[0] > 0.5:
                team_1_pred_team_1 = 1
            else:
                team_1_pred_team_1 = 0

            if team_1_out == team_1_both_result:
                both_teams_out += 1
            if team_1_out == team_1_pred_team_1:
                one_team_out += 1  
                
            count_teams += 1
            proportion_both_teams = both_teams_out / count_teams if count_teams > 0 else 0
            proportion_one_team = one_team_out / count_teams if count_teams > 0 else 0

            #write proportions to file
            with open('proportions_test.txt', 'a') as f:
                f.write(f"Both teams: {proportion_both_teams:.3f}, One team: {proportion_one_team:.3f}\n")
            print('=======================================')
            print(f'{self.team_1} win proba: {median_probas_both_teams[0]}')
            print(f'{self.team_2} win proba: {median_probas_both_teams[1]}')
            print(f'Prediction both {self.team_1}: {team_1_both_result}')
            print(f'Prediction single {self.team_1}: {team_1_pred_team_1}')
            print(f'did {self.team_1} win: {team_1_out}')
            print(f'Monte Carlo both teams {count_teams} teams: {both_teams_out / count_teams}')
            print(f'Monte Carlo Team 1 {count_teams} teams: {one_team_out / count_teams}')
            print('=======================================')
            lines.pop(idx)
            with open(teams_file, 'w') as f:
                f.writelines(lines)
            check_ram_usage_txt(teams_file)

            #check if RAM is the only string in the file
            with open(teams_file , 'r') as file:
                lines = file.readlines()
            if len(lines) == 1 and lines[0].strip() == "RAM Full":
                with open(teams_file, 'w') as file:
                    file.write("")
            # idx += 1

    def predict_teams(self, teams_file='teams_played_this_week.txt', results_file='results.csv'):
        live_plot = False
        num_layers = self.dnn_class.input_shape[1]
        if num_layers == self.manual_comp:
            print('Number of layers from standardization and model are the same')
            layer_diff = False
        else:
            print('Number of layers from standardization and model are diff. remove a layer')
            layer_diff = True
        #read the team names from the file
        with open(teams_file, 'r') as f:
            lines = f.readlines()

        idx = 0
        #prepare an empty list to store results for batch writing
        # results_list = []

        while idx < len(lines):
                line = lines[idx].strip()
            # try:
                teams = line.split(',')
                if len(teams) != 2:
                    print(f'Invalid format in line: {line}')
                    idx += 1
                    continue

                self.team_1, self.team_2 = teams
                print('================================================================')
                print(f'Currently making predictions for {self.team_1} vs. {self.team_2}')
                print('================================================================')
                #team data processing
                # team_1_df, team_2_df = DataFrame(), DataFrame()  # Initialize as empty DataFrames

                # for year in tqdm([2022, 2023, 2024]):
                #     # Collect data for team 1
                #     team_1_df_temp = collect_two_teams(f'https://www.sports-reference.com/cfb/schools/{self.team_1.lower()}/{year}/gamelog/', self.team_1.lower(), year)
                #     team_1_df = concat([team_1_df, team_1_df_temp], ignore_index=True)  # Concatenate team 1 data
                    
                #     # Collect data for team 2
                #     team_2_df_temp = collect_two_teams(f'https://www.sports-reference.com/cfb/schools/{self.team_2.lower()}/{year}/gamelog/', self.team_2.lower(), year)
                #     team_2_df = concat([team_2_df, team_2_df_temp], ignore_index=True)
                df_list_1, df_list_2 = [], []
                for year in tqdm([2018, 2019, 2020, 2021, 2022, 2023, 2024]):
                    try:
                        team_1_df_temp = collect_two_teams(f'https://www.sports-reference.com/cfb/schools/{self.team_1.lower()}/{year}/gamelog/', self.team_1.lower(), year)
                        team_2_df_temp = collect_two_teams(f'https://www.sports-reference.com/cfb/schools/{self.team_2.lower()}/{year}/gamelog/', self.team_2.lower(), year)
                        elo_1, elo_2 = [], []
                        team_elo = 0
                        for index, row in team_1_df_temp.iterrows():
                            game_result = 1 if 'W' in row['game_result'] else 0
                            team_elo = calculate_proxy_elo(row, team_elo, game_result)
                            elo_1.append(team_elo)
                        team_1_df_temp['team_elo'] = elo_1
                        team_elo = 0
                        for index, row in team_2_df_temp.iterrows():
                            game_result = 1 if 'W' in row['game_result'] else 0
                            team_elo = calculate_proxy_elo(row, team_elo, game_result)
                            elo_2.append(team_elo)
                        team_2_df_temp['team_elo'] = elo_2
                        df_list_1.append(team_1_df_temp)
                        df_list_2.append(team_2_df_temp)
                    except Exception as e:
                        traceback.print_exc()

                classifier_drop = ['team_1_outcome','team_2_outcome','game_loc'] #'',
                df_list_1_w_opp = add_opp(df_list_1,df_list_2)
                df_list_1_w_opp = [df for df in df_list_1_w_opp if not df.empty]
                if len(df_list_1_w_opp) < 7:
                    print('=========================')
                    print('Removed an empty df')
                    print('=========================')
                final_dict_1 = bayes_calc(df_list_1_w_opp,self.classifier_drop, self.selector,self.kpca,self.scaler)
                df_list_2_w_opp = add_opp(df_list_2,df_list_1)
                df_list_2_w_opp = [df for df in df_list_2_w_opp if not df.empty]
                if len(df_list_1_w_opp) < 7:
                    print('=========================')
                    print('df_list_2_w_opp an empty df')
                    print('=========================')
                final_dict_2 = bayes_calc(df_list_2_w_opp,self.classifier_drop, self.selector,self.kpca,self.scaler)

                final_dict_1_dn = final_dict_1[list(final_dict_1.keys())[-1]]
                final_dict_2_dn = final_dict_2[list(final_dict_2.keys())[-1]]

                n_simulations = 20000
                outcome_team_1, outcome_team_2 = monte_carlo_sim(final_dict_1_dn,n_simulations)
                outcome_team_2_2nd, outcome_team_1_2nd = monte_carlo_sim(final_dict_2_dn,n_simulations)

                outcome_1 = np.mean([round(np.mean(outcome_team_1)*100,3),round(np.mean(outcome_team_1_2nd)*100,3)])
                outcome_2 = np.mean([round(np.mean(outcome_team_2)*100,3),round(np.mean(outcome_team_2_2nd)*100,3)])
                results_dict = {
                    'Team 1': [self.team_1],
                    'Team 2': [self.team_2],
                    'Team 1 Probability': [outcome_1],
                    'Team 2 Probability': [outcome_2],
                }
                idx += 1
                with open(teams_file, 'w') as new_file:
                    new_file.writelines(lines[idx:])

                #convert list to DataFrame and write to CSV
                if results_dict:
                    if not exists(results_file):
                        results_df = DataFrame(results_dict)
                        results_df.to_csv(results_file, index=False)
                    else:
                        temp_file = read_csv(results_file)
                        concat([temp_file,DataFrame(results_dict)]).to_csv(results_file, index=False)
                #clean up and check memory
                collect()
                check_ram_usage()

                # #preprocess the data
                # team_1_df = self.str_manipulations(team_1_df)
                # team_2_df = self.str_manipulations(team_2_df)
                # team_1_df.drop(columns=self.classifier_drop, inplace=True)
                # team_2_df.drop(columns=self.classifier_drop, inplace=True)
                # team_1_df.dropna(inplace=True)
                # team_2_df.dropna(inplace=True)
                # print('============')
                # print(f'Shapes of data for each team: {team_1_df.shape} and {team_2_df.shape}')
                # print('============')
                # #handle length mismatch
                # length_difference = len(team_1_df) - len(team_2_df)
                # if length_difference > 0:
                #     team_1_df = team_1_df.iloc[length_difference:]
                # elif length_difference < 0:
                #     team_2_df = team_2_df.iloc[-length_difference:]

                # #monte Carlo simulations
                # n_simulations = 10000
                # # all_probas = zeros(2)
                # all_probas_norm = zeros(2)
                # all_probas_log = zeros(2)
                # all_probas_beta = zeros(2)
                # all_probas_best = zeros(2)

                # #team 1 processing and predictions
                # team_1_df_copy = team_1_df.copy()
                # team_2_df_copy = team_2_df.copy()

                # for col in [c for c in team_2_df.columns if '_opp' not in c]:
                #     opp_col = col + '_opp'
                #     if opp_col in team_1_df_copy.columns:
                #         team_1_df_copy[opp_col] = team_2_df_copy[col]
                # #team_1_df_copy['team_2_score'] = team_2_df_copy['team_1_score']

                # X_std = self.scaler.transform(team_1_df_copy)
                # X_std = X_std[~isnan(X_std).any(axis=1)]
                # # imputer = SimpleImputer(strategy='mean')  # You can also use 'median' or another strategy
                # # X_std = imputer.fit_transform(X_std)
                # X_fa = self.kpca.transform(X_std)
                # # X_fa = concatenate((self.umap_reducer.transform(X_std), self.kpca.transform(X_std), 
                # #                     self.pca.transform(X_std), self.fa.transform(X_std)), axis=1)           
                # X_fa = self.selector.transform(X_fa)
                # team_1_df_copy = DataFrame(X_fa, columns=[f'FA{i}' for i in range(1, X_fa.shape[1] + 1)])
                # print('============')
                # print(f'Shapes of data for predcition: {team_1_df_copy.shape}')
                # print('============')
                # # team_1_df_copy.drop(self.non_normal_columns, axis=1, inplace=True)
                # if layer_diff == True:
                #     team_1_df_copy = team_1_df_copy.iloc[:, :num_layers]
                
                # #feature forecast and monte carlo sampling
                # all_probas_norm, all_probas_log, all_probas_beta, all_probas_best, win_props_team_1 = self.monte_carlo_sampling(team_1_df_copy, all_probas_norm, 
                #                                                                                  all_probas_log, all_probas_beta, all_probas_best,
                #                                                                                  'team_1',self.team_1, self.team_2, n_simulations)
                #     # for _ in tqdm(range(n_simulations)):
                #     #     mc_sample = array([norm.rvs(loc=team_1_df_copy[col].mean(), scale=team_1_df_copy[col].std()*3) 
                #     #                     for col in team_1_df_copy.columns]).T
                #     #     probas = self.dnn_class.predict(mc_sample.reshape(1, -1), verbose=0)
                #     #     all_probas += probas[0]

                # #team 2 processing and predictions
                # for col in [c for c in team_1_df.columns if '_opp' not in c]:
                #     opp_col = col + '_opp'
                #     if opp_col in team_2_df.columns:
                #         team_2_df[opp_col] = team_1_df[col]
                # #team_2_df['team_2_score'] = team_1_df['team_1_score']
                
                # X_std_t2 = self.scaler.transform(team_2_df)
                # X_std_t2 = X_std_t2[~isnan(X_std_t2).any(axis=1)]
                # # imputer = SimpleImputer(strategy='mean')  # You can also use 'median' or another strategy
                # # X_std_t2 = imputer.fit_transform(X_std_t2)
                # x_fa_t2 = self.kpca.transform(X_std_t2)
                # # X_fa_t2 = concatenate((self.umap_reducer.transform(X_std_t2), self.kpca.transform(X_std_t2), 
                # #                     self.pca.transform(X_std_t2), self.fa.transform(X_std_t2)), axis=1)
                
                # X_fa_t2_ = self.selector.transform(x_fa_t2)
                # team_2_df = DataFrame(X_fa_t2_, columns=[f'FA{i}' for i in range(1, X_fa_t2_.shape[1] + 1)])
                # # team_2_df.drop(self.non_normal_columns, axis=1, inplace=True)
                # if layer_diff == True:
                #     team_2_df = team_2_df.iloc[:, :num_layers]

                # all_probas_norm, all_probas_log, all_probas_beta, all_probas_best, win_props_team_2 = self.monte_carlo_sampling(team_2_df, all_probas_norm, 
                #                                                                                  all_probas_log, all_probas_beta, all_probas_best,
                #                                                                                  'team_2',self.team_1, self.team_2, n_simulations)
                # print('======================')
                # print(win_props_team_1)
                # print(win_props_team_2)
                # print('======================')
                # # for _ in tqdm(range(n_simulations)):
                # #     mc_sample = array([norm.rvs(loc=team_2_df[col].mean(), scale=team_2_df[col].std()*3) 
                # #                     for col in team_2_df.columns]).T
                # #     probas = self.dnn_class.predict(mc_sample.reshape(1, -1), verbose=0)
                # #     all_probas += probas[0][::-1]  #flip probabilities for team 2

                # #calculate median probabilities and predicted winner
                # #normal dist
                # median_probas_norm = all_probas_norm / (n_simulations * 2)
                # predicted_class_norm = argmax(median_probas_norm)
                # predicted_winner_norm = self.team_1 if predicted_class_norm == 0 else self.team_2
                # #log dist
                # median_probas_log = all_probas_log / (n_simulations * 2)
                # predicted_class_log = argmax(median_probas_log)
                # predicted_winner_log = self.team_1 if predicted_class_log == 0 else self.team_2
                # #beta dist
                # median_probas_beta = all_probas_beta / (n_simulations * 2)
                # predicted_class_beta = argmax(median_probas_beta)
                # predicted_winner_beta = self.team_1 if predicted_class_beta == 0 else self.team_2
                # #best dist
                # median_probas_best = all_probas_best / (n_simulations * 2)
                # predicted_class_best = argmax(median_probas_best)
                # predicted_winner_best = self.team_1 if predicted_class_best == 0 else self.team_2

                # #add results to list
                # results_dict = {
                #     'Team 1': [self.team_1],
                #     'Team 2': [self.team_2],
                #     'Team 1 Probability Norm': [round(median_probas_norm[0] * 100, 3)],
                #     'Team 2 Probability Norm': [round(median_probas_norm[1] * 100, 3)],
                #     'Predicted Winner Norm': [predicted_winner_norm],
                #     'Team 1 Probability Log': [round(median_probas_log[0] * 100, 3)],
                #     'Team 2 Probability Log': [round(median_probas_log[1] * 100, 3)],
                #     'Predicted Winner Log': [predicted_winner_log],
                #     'Team 1 Probability Beta': [round(median_probas_beta[0] * 100, 3)],
                #     'Team 2 Probability Beta': [round(median_probas_beta[1] * 100, 3)],
                #     'Predicted Winner Beta': [predicted_winner_beta],
                #     'Team 1 Probability Best': [round(median_probas_best[0] * 100, 3)],
                #     'Team 2 Probability Best': [round(median_probas_best[1] * 100, 3)],
                #     'Predicted Winner Best': [predicted_winner_best]
                # }

                # #update teams file by removing processed line
                # idx += 1
                # with open(teams_file, 'w') as new_file:
                #     new_file.writelines(lines[idx:])

                # #convert list to DataFrame and write to CSV
                # if results_dict:
                #     if not exists(results_file):
                #         results_df = DataFrame(results_dict)
                #         results_df.to_csv(results_file, index=False)
                #     else:
                #         temp_file = read_csv(results_file)
                #         concat([temp_file,DataFrame(results_dict)]).to_csv(results_file, index=False)
                # #clean up and check memory
                # del team_1_df, team_2_df
                # collect()
                # check_ram_usage()

            # except Exception as e:
            #     print(f'The error: {e}. Most likely {self.team_1} or {self.team_2} do not have data')
            #     idx += 1

    def find_best_distribution(self,data):
        distributions = [
            'norm', 'lognorm', 'beta', 'gamma', 'expon', 'uniform', 'weibull_min', 'weibull_max',
            'pareto', 't', 'chi2', 'triang', 'invgauss', 'genextreme', 'logistic', 'gumbel_r', 'gumbel_l',
            'loggamma', 'powerlaw', 'rayleigh', 'laplace', 'cauchy'
        ]
        n_samples = 1
        f = Fitter(data, distributions=distributions)
        f.fit()
        best_dist = f.get_best(method='sumsquare_error')
        dist_name = list(best_dist.keys())[0]  # Get the name of the best distribution
        dist_params = best_dist[dist_name]     # Get the parameters of the best distribution

        # Step 2: Generate samples from the best-fit distribution
        # if dist_name == 'norm':
        #     samples = norm.rvs(loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        # elif dist_name == 'lognorm':
        #     samples = lognorm.rvs(s=dist_params['s'], loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        # elif dist_name == 'beta':
        #     samples = beta.rvs(a=dist_params['a'], b=dist_params['b'], loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        # elif dist_name == 'gamma':
        #     samples = gamma.rvs(a=dist_params['a'], loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        # elif dist_name == 'expon':
        #     samples = expon.rvs(loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        # elif dist_name == 'uniform':
        #     samples = uniform.rvs(loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        # elif dist_name == 'weibull_min':
        #     samples = weibull_min.rvs(c=dist_params['c'], loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        # elif dist_name == 'weibull_max':
        #     samples = weibull_max.rvs(c=dist_params['c'], loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        # elif dist_name == 'pareto':
        #     samples = pareto.rvs(b=dist_params['b'], loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        # elif dist_name == 't':
        #     samples = t.rvs(df=dist_params['df'], loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        # elif dist_name == 'chi2':
        #     samples = chi2.rvs(df=dist_params['df'], loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        # elif dist_name == 'triang':
        #     samples = triang.rvs(c=dist_params['c'], loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        # elif dist_name == 'invgauss':
        #     samples = invgauss.rvs(mu=dist_params['mu'], loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        # elif dist_name == 'genextreme':
        #     samples = genextreme.rvs(c=dist_params['c'], loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        # elif dist_name == 'logistic':
        #     samples = logistic.rvs(loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        # elif dist_name == 'gumbel_r':
        #     samples = gumbel_r.rvs(loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        # elif dist_name == 'gumbel_l':
        #     samples = gumbel_l.rvs(loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        # elif dist_name == 'loggamma':
        #     samples = loggamma.rvs(c=dist_params['c'], loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        # elif dist_name == 'powerlaw':
        #     samples = powerlaw.rvs(a=dist_params['a'], loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        # elif dist_name == 'rayleigh':
        #     samples = rayleigh.rvs(loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        # elif dist_name == 'laplace':
        #     samples = laplace.rvs(loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        # elif dist_name == 'cauchy':
        #     samples = cauchy.rvs(loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        # else:
        #     raise ValueError(f"Distribution {dist_name} not handled.")

        return dist_name, dist_params
    
    def sample_from_distribution(self, dist_name, dist_params, n_samples=1):
        if dist_name == 'norm':
            samples = norm.rvs(loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        elif dist_name == 'lognorm':
            samples = lognorm.rvs(s=dist_params['s'], loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        elif dist_name == 'beta':
            samples = beta.rvs(a=dist_params['a'], b=dist_params['b'], loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        elif dist_name == 'gamma':
            samples = gamma.rvs(a=dist_params['a'], loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        elif dist_name == 'expon':
            samples = expon.rvs(loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        elif dist_name == 'uniform':
            samples = uniform.rvs(loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        elif dist_name == 'weibull_min':
            samples = weibull_min.rvs(c=dist_params['c'], loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        elif dist_name == 'weibull_max':
            samples = weibull_max.rvs(c=dist_params['c'], loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        elif dist_name == 'pareto':
            samples = pareto.rvs(b=dist_params['b'], loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        elif dist_name == 't':
            samples = t.rvs(df=dist_params['df'], loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        elif dist_name == 'chi2':
            samples = chi2.rvs(df=dist_params['df'], loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        elif dist_name == 'triang':
            samples = triang.rvs(c=dist_params['c'], loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        elif dist_name == 'invgauss':
            samples = invgauss.rvs(mu=dist_params['mu'], loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        elif dist_name == 'genextreme':
            samples = genextreme.rvs(c=dist_params['c'], loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        elif dist_name == 'logistic':
            samples = logistic.rvs(loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        elif dist_name == 'gumbel_r':
            samples = gumbel_r.rvs(loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        elif dist_name == 'gumbel_l':
            samples = gumbel_l.rvs(loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        elif dist_name == 'loggamma':
            samples = loggamma.rvs(c=dist_params['c'], loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        elif dist_name == 'powerlaw':
            samples = powerlaw.rvs(a=dist_params['a'], loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        elif dist_name == 'rayleigh':
            samples = rayleigh.rvs(loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        elif dist_name == 'laplace':
            samples = laplace.rvs(loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        elif dist_name == 'cauchy':
            samples = cauchy.rvs(loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        else:
            raise ValueError(f"Distribution {dist_name} not handled.")

        return samples

    def monte_carlo_sampling(self, df, all_probas_norm, all_probas_log, all_probas_beta, all_probas_best, team, team_1, team_2, n_simulations=10000):
        #init the team wins
        team1_wins_norm, team1_wins_log, team1_wins_beta, team1_wins_best = 0, 0, 0, 0
        team2_wins_norm, team2_wins_log, team2_wins_beta, team2_wins_best = 0, 0, 0, 0

        #find the best dist for each feature
        data_best_fit = {}
        # axes = df.hist(bins=20, figsize=(10, 8))
        axes = df.plot()
        plt.tight_layout()
        plt.savefig('histogram_example_used_for_fitting.png')
        plt.close()

        self.forecasts = {}
        for column in df.columns:
            # model = ARIMA(df[column], order=(1, 1, 1))
            model = auto_arima(df[column], seasonal=False,
                            suppress_warnings=True,
                            stepwise=True,
                            error_action="ignore") #13 week season  m=13,
            forecast = model.predict(n_periods=1).values  # Forecast for the next game
            # if column not in self.forecasts:
            #     self.forecasts[column] = []
            # self.forecasts[column].append(forecast)
            self.forecasts[column] = forecast
        forecasted_data = DataFrame(self.forecasts)
        forecasted_pred = self.dnn_class.predict(forecasted_data.to_numpy().reshape(1, -1), verbose=0)
        # print(self.forecasts)
        # for col in df.columns:
        #     plt.figure()
        #     plt.plot(arange(0,len(df[column])),df[column])
        #     plt.scatter([len(df)] * len(self.forecasts[column]),self.forecasts[column],color='green')
        #     plt.show()

        # for col in df.columns:
        #     dist_name, dist_params = self.find_best_distribution(df[col])
        #     data_best_fit[col] = (dist_name, dist_params) 

        # for _ in tqdm(range(n_simulations)):
        #     mc_sample_norm = array([norm.rvs(loc=df[col].mean(), scale=df[col].std()*3) 
        #                         for col in df.columns]).T
        #     mc_sample_log = array([lognorm.rvs(s=df[col].std(), scale=exp(df[col].mean()))
        #                         for col in df.columns]).T
        #     mc_sample_beta = array(beta.rvs(a, b) * (df.min() - df.max()) + df.min())

        #     #sample from distribution
        #     mc_sample_best_fit = []
        #     for col in df.columns:
        #         dist_name, dist_params = data_best_fit[col]
        #         sample = self.sample_from_distribution(dist_name, dist_params)
        #         mc_sample_best_fit.append(sample)
        #     mc_sample_best_fit = array(mc_sample_best_fit).reshape(1, -1)

        #     #predictions
        #     probas_norm = self.dnn_class.predict(mc_sample_norm.reshape(1, -1), verbose=0)
        #     probas_log = self.dnn_class.predict(mc_sample_log.reshape(1, -1), verbose=0)
        #     probas_beta = self.dnn_class.predict(mc_sample_beta.reshape(1, -1), verbose=0)
        #     probas_best = self.dnn_class.predict(mc_sample_best_fit, verbose=0)

        #     #save probas
        #     if team == 'team_1':
        #         all_probas_norm += probas_norm[0]
        #         all_probas_log += probas_log[0]
        #         all_probas_beta += probas_beta[0]
        #         all_probas_best += probas_best[0]

        #         #add how many times team_1 beat team_2
        #         team1_wins_norm += probas_norm[0][0] > probas_norm[0][1]
        #         team1_wins_log += probas_log[0][0] > probas_log[0][1]
        #         team1_wins_beta += probas_beta[0][0] > probas_beta[0][1]
                
        #         team2_wins_norm += probas_norm[0][1] > probas_norm[0][0]
        #         team2_wins_log += probas_log[0][1] > probas_log[0][0]
        #         team2_wins_beta += probas_beta[0][1] > probas_beta[0][0]
        #     else:
        #         all_probas_norm += probas_norm[0][::-1]
        #         all_probas_log += probas_log[0][::-1]
        #         all_probas_beta += probas_beta[0][::-1]
        #         all_probas_best += probas_best[0][::-1]

        #         flip_norm = probas_norm[0][::-1]
        #         flip_log = probas_log[0][::-1]
        #         flip_beta = probas_beta[0][::-1]
                
        #         team1_wins_norm += flip_norm[0] > flip_norm[1]
        #         team1_wins_log += flip_log[0] > flip_log[1]
        #         team1_wins_beta += flip_beta[0] > flip_beta[1]
                
        #         team2_wins_norm += flip_norm[1] > flip_norm[0]
        #         team2_wins_log += flip_log[1] > flip_log[0]
        #         team2_wins_beta += flip_beta[1] > flip_beta[0]
        
        # #calculate win proportions
        # team1_win_prop_norm = team1_wins_norm / n_simulations
        # team1_win_prop_log = team1_wins_log / n_simulations
        # team1_win_prop_beta = team1_wins_beta / n_simulations
        
        # team2_win_prop_norm = team2_wins_norm / n_simulations
        # team2_win_prop_log = team2_wins_log / n_simulations
        # team2_win_prop_beta = team2_wins_beta / n_simulations
        
        # win_proportions = {
        #     f'{team_1}': {
        #         'norm': team1_win_prop_norm*100,
        #         'log': team1_win_prop_log*100,
        #         'beta': team1_win_prop_beta*100
        #     },
        #     f'{team_2}': {
        #         'norm': team2_win_prop_norm*100,
        #         'log': team2_win_prop_log*100,
        #         'beta': team2_win_prop_beta*100
        #     }
        # }
        return all_probas_norm, all_probas_log, all_probas_beta, all_probas_best, win_proportions
    # def predict_teams(self, teams_file='team_names_played_this_week.txt', results_file='results.txt'):
    #     try:
    #         with open(teams_file, 'r') as f:
    #             lines = f.readlines()

    #         idx = 0
    #         with open(results_file, 'a') as results:
    #             while idx < len(lines):
    #                 line = lines[idx].strip()
    #                 try:
    #                     teams = line.split(',')
    #                     if len(teams) != 2:
    #                         results.write(f'Invalid format in line: {line}\n')
    #                         idx += 1
    #                         continue
                        
    #                     self.team_1, self.team_2 = teams
    #                     print(f'Currently making predictions for {self.team_1} vs. {self.team_2}')
                        
    #                     #team data
    #                     team_1_df_2023 = collect_two_teams(f'https://www.sports-reference.com/cfb/schools/{self.team_1.lower()}/2023/gamelog/', self.team_1.lower(), 2023)
    #                     team_1_df_2024 = collect_two_teams(f'https://www.sports-reference.com/cfb/schools/{self.team_1.lower()}/2024/gamelog/', self.team_1.lower(), 2024)
    #                     team_1_df = concat([team_1_df_2023, team_1_df_2024])

    #                     team_2_df_2023 = collect_two_teams(f'https://www.sports-reference.com/cfb/schools/{self.team_2.lower()}/2023/gamelog/', self.team_2.lower(), 2023)
    #                     team_2_df_2024 = collect_two_teams(f'https://www.sports-reference.com/cfb/schools/{self.team_2.lower()}/2024/gamelog/', self.team_2.lower(), 2024)
    #                     team_2_df = concat([team_2_df_2023, team_2_df_2024])

    #                     #preprocess
    #                     team_1_df = self.str_manipulations(team_1_df)
    #                     team_2_df = self.str_manipulations(team_2_df)
    #                     team_1_df.drop(columns=self.classifier_drop, inplace=True)
    #                     team_2_df.drop(columns=self.classifier_drop, inplace=True)

    #                     #length mismatch
    #                     length_difference = len(team_1_df) - len(team_2_df)
    #                     if length_difference > 0:
    #                         team_1_df = team_1_df.iloc[length_difference:]
    #                     elif length_difference < 0:
    #                         team_2_df = team_2_df.iloc[-length_difference:]

    #                     #Monte Carlo simulations
    #                     n_simulations = 5000
    #                     all_probas = zeros(2)

    #                     #team 1
    #                     team_1_df_copy = team_1_df.copy()
    #                     team_2_df_copy = team_2_df.copy()

    #                     for col in [c for c in team_2_df.columns if '_opp' not in c]:
    #                         opp_col = col + '_opp'
    #                         if opp_col in team_1_df_copy.columns:
    #                             team_1_df_copy[opp_col] = team_2_df_copy[col]
    #                     team_1_df_copy['team_2_score'] = team_2_df_copy['team_1_score']

    #                     #transformations and predictions
    #                     X_std = self.scaler.transform(team_1_df_copy)
    #                     X_fa = self.fa.transform(X_std)
    #                     team_1_df_copy = DataFrame(X_fa, columns=[f'FA{i}' for i in range(1, self.manual_comp + 1)])
    #                     team_1_df_copy.drop(self.non_normal_columns, axis=1, inplace=True)

    #                     for _ in tqdm(range(n_simulations)):
    #                         mc_sample = array([norm.rvs(loc=team_1_df_copy[col].mean(), scale=team_1_df_copy[col].std()*3) 
    #                                         for col in team_1_df_copy.columns]).T
    #                         probas = self.dnn_class.predict(mc_sample.reshape(1, -1))
    #                         all_probas += probas[0]

    #                     #team 2
    #                     for col in [c for c in team_1_df.columns if '_opp' not in c]:
    #                         opp_col = col + '_opp'
    #                         if opp_col in team_2_df.columns:
    #                             team_2_df[opp_col] = team_1_df[col]
    #                     team_2_df['team_2_score'] = team_1_df['team_1_score']

    #                     X_std_t2 = self.scaler.transform(team_2_df)
    #                     X_fa_t2 = self.fa.transform(X_std_t2)
    #                     final_df_t2 = DataFrame(X_fa_t2, columns=[f'FA{i}' for i in range(1, self.manual_comp + 1)])
    #                     final_df_t2.drop(self.non_normal_columns, axis=1, inplace=True)

    #                     for _ in tqdm(range(n_simulations)):
    #                         mc_sample = array([norm.rvs(loc=final_df_t2[col].mean(), scale=final_df_t2[col].std()*3) 
    #                                         for col in final_df_t2.columns]).T
    #                         probas = self.dnn_class.predict(mc_sample.reshape(1, -1))
    #                         all_probas += probas[0][::-1]  #flip probabilities

    #                     #median probabilities
    #                     median_probas = all_probas / (n_simulations * 2)
    #                     predicted_class = argmax(median_probas)

    #                     rolling_features_2_team_1 = team_1_df_copy.rolling(2).median().iloc[-1:]
    #                     rolling_features_3_team_1 = team_1_df_copy.rolling(3).median().iloc[-1:]
    #                     rolling_features_ewm = team_1_df_copy.ewm(span=2).mean().iloc[-1:]
    #                     rolling_low = team_1_df_copy.rolling(window=2).quantile(0.25).iloc[-1:]
    #                     rolling_high = team_1_df_copy.rolling(window=2).quantile(0.75).iloc[-1:]

    #                     #predictions for rolling statistics
    #                     predictions = [
    #                         ('rolling median of 2', self.dnn_class.predict(rolling_features_2_team_1)),
    #                         ('rolling median of 3', self.dnn_class.predict(rolling_features_3_team_1)),
    #                         ('exponential weighted average', self.dnn_class.predict(rolling_features_ewm)),
    #                         ('25th percentile', self.dnn_class.predict(rolling_low)),
    #                         ('75th percentile', self.dnn_class.predict(rolling_high))
    #                     ]

    #                     #results
    #                     results.write('==============================\n')
    #                     results.write(f'Win Probabilities from Monte Carlo Simulation with {n_simulations*2} simulations\n')
    #                     results.write(f'{self.team_1} : {round(median_probas[0] * 100, 3)}%\n')
    #                     results.write(f'{self.team_2} : {round(median_probas[1] * 100, 3)}%\n')

    #                     for label, pred in predictions:
    #                         results.write(f'Win Probabilities from {label}\n')
    #                         results.write(f'{self.team_1} : {round(pred[0][0] * 100, 3)}%\n')
    #                         results.write(f'{self.team_2} : {round(pred[0][1] * 100, 3)}%\n')

    #                     results.write('==============================\n')

    #                     #update teams file by removing processed line
    #                     idx += 1
    #                     with open(teams_file, 'w') as new_file:
    #                         new_file.writelines(lines[idx:])

    #                     #clean up and check memory
    #                     del team_1_df, team_2_df
    #                     collect()
    #                     check_ram_usage()

    #                 except Exception as e:
    #                     results.write(f'The error: {e}. Most likely {self.team_1} or {self.team_2} do not have data\n')
    #                     idx += 1

    #     except Exception as e:
    #         print(f"Error: {e}")

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
        # self.deep_learn_features()
        if argv[1] == "test":
            self.test_forecast()
        else:
            self.predict_teams()

def main():
    deepCfbMulti().run_analysis()

if __name__ == "__main__":
    main()
