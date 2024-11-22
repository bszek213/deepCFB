#multiclass deep-learning on college football games
from pandas import read_csv, DataFrame, concat, io, to_numeric, io
from os.path import join, exists
from os import getcwd, mkdir, environ, makedirs
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
import seaborn as sns

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



def monte_carlo_sim(final_dict_1_dn,dnn_class,actual_winner=None,team_1=None,team_2=None,pred_type='ewm',n_simulations=5000):
    makedirs('predictions',exist_ok=True)
    dict_data_monte = {}
    list_final_1, list_final_2 = [], []
    for key in list(final_dict_1_dn.keys()):
        mu = final_dict_1_dn[key]['mu']
        std = final_dict_1_dn[key]['std']
        dict_data_monte[key] = np.random.normal(loc=mu, scale=std, size=n_simulations)

    df_monte = DataFrame(dict_data_monte)
    # plt.hist(df_monte[df_monte.columns[0]])
    # plt.show()
    list_temp_1, list_temp_2 = [], []
    for index, row in tqdm(df_monte.iterrows()):
        row_array = row.values
        out = dnn_class.predict(row_array.reshape(1, -1), verbose=0)
        list_temp_1.append(out[0][0])
        list_temp_2.append(out[0][1])
    list_final_1.append(round(np.mean(list_temp_1),3))
    list_final_2.append(round(np.mean(list_temp_2),3))

    df_monte['prob_class_0'] = list_temp_1
    df_monte['prob_class_1'] = list_temp_2

    # Determine predicted winner for each simulation
    df_monte['predicted_winner'] = np.where(df_monte['prob_class_1'] > df_monte['prob_class_0'], 1, 0)
    
    if actual_winner != None:
        # Add column to indicate correct prediction
        df_monte['is_correct'] = np.where(df_monte['predicted_winner'] == actual_winner, 1, 0)

        feature_columns = df_monte.columns.difference(['prob_class_0', 'prob_class_1', 'predicted_winner', 'is_correct'])
        n_features = len(feature_columns)
        n_cols = 3  # Number of columns in the plot grid
        n_rows = (n_features // n_cols) + (n_features % n_cols > 0)

        plt.figure(figsize=(15, n_rows * 4))

        for idx, feature in enumerate(feature_columns, start=1):
            plt.subplot(n_rows, n_cols, idx)

            sns.histplot(df_monte[feature], color='blue', label='normal', kde=True, bins=30)
            #correct predictions in green
            sns.histplot(df_monte[df_monte['is_correct'] == 1][feature], color='green', label='Correct', kde=True, bins=30)

            plt.title(f"Feature: {feature}")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.legend()

        plt.suptitle(f"Monte Carlo Simulation {pred_type}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'figures/correct_predictions_{pred_type}.png')
        plt.close()
    else:
        first_feature = df_monte.columns[0]
        plt.figure(figsize=(10, 6))

        sns.histplot(
            df_monte[first_feature],
            color='grey', label='Full Distribution', kde=True, bins=50, alpha=0.5
        )
        
        sns.histplot(
            df_monte[df_monte['predicted_winner'] == 0][first_feature],
            color='red', label=f'{team_1} Wins', kde=True, bins=50
        )
        
        sns.histplot(
            df_monte[df_monte['predicted_winner'] == 1][first_feature],
            color='blue', label=f'{team_2} Wins', kde=True, bins=50
        )
        
        plt.title(f"Histogram of {first_feature} - Team Wins Distribution")
        plt.xlabel("Feature Value")
        plt.ylabel("Frequency")
        plt.legend()
        
        plt.savefig(f'predictions/first_feature_wins_histogram_{pred_type}_{team_1}.png')
        plt.close()

    return list_final_1, list_final_2, np.sum(df_monte['predicted_winner'] == 0) / n_simulations, np.sum(df_monte['predicted_winner'] == 1) / n_simulations

# def monte_carlo_sim(final_dict_1_dn,dnn_class,n_simulations=5000):
#     dict_data_monte = {}
#     # std_mult = [19448624389.373577,10000000.0,441005945.4176732,19448624389.373577,171132830.41617775]
#     # min_value = np.log10(min(std_mult))
#     # max_value = np.log10(max(std_mult))
#     # num_points = int(len(std_mult) * 1.5) # You can set this to any number of points you want
#     # log_scale = np.logspace(min_value, max_value, num=num_points)
#     list_final_1, list_final_2 = [] , []
#     # for multi in tqdm(log_scale):
#     for key in list(final_dict_1_dn.keys()):
#         mu = final_dict_1_dn[key]['mu']
#         std = final_dict_1_dn[key]['std']
#         dict_data_monte[key] = np.random.normal(loc=mu, scale=std * 1, size=n_simulations)
        
#     df_monte = DataFrame(dict_data_monte)
#     list_temp_1, list_temp_2 = [], []
#     for index, row in tqdm(df_monte.iterrows()):
#         row_array = row.values
#         out = dnn_class.predict(row_array.reshape(1, -1), verbose=0)
#         list_temp_1.append(out[0][0])
#         list_temp_2.append(out[0][1])
#     list_final_1.append(np.mean(list_temp_1))
#     list_final_2.append(np.mean(list_temp_2))
#     return list_final_1, list_final_2

def kalman_filter_update(data, initial_mu, initial_std, process_var=1e-3, measurement_var=1e-2):
    kalman_results = {}
    
    #measurement variance
    ewm_smoothed = data.ewm(alpha=0.3,adjust=False).mean()
    residuals = data - ewm_smoothed
    measurement_var = residuals.var().mean()

    #process variance
    smoothed_data = data.ewm(alpha=0.3,adjust=False).mean()
    process_diff = smoothed_data.diff().dropna()
    process_var = process_diff.var().mean()

    #init each feature's mean and variance
    for feature in data.columns:
        mu = initial_mu[feature]
        var = initial_std[feature]**2  # Variance is the square of std deviation
        kalman_results[feature] = {'mu': mu, 'var': var}

    #run Kalman Filter
    for i in range(len(data)):
        for feature in data.columns:
            #measurement update
            measurement = data[feature].iloc[i]
            pred_mu = kalman_results[feature]['mu']

            #predicted variance with process noise
            pred_var = kalman_results[feature]['var'] + process_var  
            
            kalman_gain = pred_var / (pred_var + measurement_var)
            updated_mu = pred_mu + kalman_gain * (measurement - pred_mu)
            updated_var = (1 - kalman_gain) * pred_var
            
            #update the Kalman filter
            kalman_results[feature] = {'mu': updated_mu, 'var': updated_var}

    #final mu and std for each feature
    final_dict_1_dn = {
        feature: {'mu': kalman_results[feature]['mu'], 'std': np.sqrt(kalman_results[feature]['var'])}
        for feature in data.columns
    }
    
    return final_dict_1_dn

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
                for year in tqdm([2023, 2024]): #2018, 2019, 2020, 2021, 2022, 
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

                # classifier_drop = ['team_1_outcome','team_2_outcome','game_loc'] #'',
                df_list_1_w_opp = add_opp(df_list_1,df_list_2)
                df_list_1_w_opp = [df for df in df_list_1_w_opp if not df.empty]
                if len(df_list_1_w_opp) < 2:
                    print('=========================')
                    print('Removed an empty df')
                    print('=========================')

                #Kalman filter team 1
                df_curr = str_manipulations(concat(df_list_1_w_opp, ignore_index=True))
                df_curr = df_curr.drop(columns=self.classifier_drop)

                x_curr = self.selector.transform(self.kpca.transform(self.scaler.transform(df_curr)))
                x_curr = DataFrame(x_curr)
                ewm_mean = x_curr.ewm(alpha=0.3,adjust=False).mean().iloc[-1]
                ewm_std = x_curr.ewm(alpha=0.3,adjust=False).std().iloc[-1]
                final_dict_1_dn_ewm = {}
                for col in ewm_mean.index:
                    final_dict_1_dn_ewm[col] = {
                        'mu': ewm_mean[col],
                        'std': ewm_std[col] * 1.96
                    }
                final_dict_1_dn_kal = kalman_filter_update(x_curr, x_curr.mean(), x_curr.std())
                # final_dict_1 = bayes_calc(df_list_1_w_opp,self.classifier_drop, self.selector,self.kpca,self.scaler)

                df_list_2_w_opp = add_opp(df_list_2,df_list_1)
                df_list_2_w_opp = [df for df in df_list_2_w_opp if not df.empty]
                if len(df_list_2_w_opp) < 2:
                    print('=========================')
                    print('df_list_2_w_opp an empty df')
                    print('=========================')
                
                #Kalman filter team 2
                df_curr_2 = str_manipulations(concat(df_list_2_w_opp, ignore_index=True))
                df_curr_2 = df_curr_2.drop(columns=self.classifier_drop)

                x_curr_2 = self.selector.transform(self.kpca.transform(self.scaler.transform(df_curr_2)))
                x_curr_2 = DataFrame(x_curr_2)
                ewm_mean = x_curr.ewm(alpha=0.3,adjust=False).mean().iloc[-1]
                ewm_std = x_curr.ewm(alpha=0.3,adjust=False).std().iloc[-1]
                final_dict_2_dn_ewm = {}
                for col in ewm_mean.index:
                    final_dict_2_dn_ewm[col] = {
                        'mu': ewm_mean[col],
                        'std': ewm_std[col] * 1.96
                    }
                final_dict_2_dn_kal = kalman_filter_update(x_curr, x_curr.mean(), x_curr.std())
                # final_dict_2 = bayes_calc(df_list_2_w_opp,self.classifier_drop, self.selector,self.kpca,self.scaler)

                # final_dict_1_dn = final_dict_1[list(final_dict_1.keys())[-1]]
                # final_dict_2_dn = final_dict_2[list(final_dict_2.keys())[-1]]

                n_simulations = 2000
                outcome_team_1_ewm, outcome_team_2_ewm, predict_team_1_prop_1, predict_team_2_prop_1 = monte_carlo_sim(final_dict_1_dn_ewm,dnn_class=self.dnn_class,n_simulations=n_simulations,team_1=self.team_1,team_2=self.team_2,pred_type='ewm')
                # outcome_team_2_2nd_ewm, outcome_team_1_2nd_ewm, predict_team_2_prop_2, predict_team_1_prop_2 = monte_carlo_sim(final_dict_2_dn_ewm,dnn_class=self.dnn_class,n_simulations=n_simulations,team_1=self.team_2,team_2=self.team_1,pred_type='ewm')

                final_prop_team_1_ewm = round(np.mean([predict_team_1_prop_1])*100,3)
                final_prop_team_2_ewm = round(np.mean([predict_team_2_prop_1])*100,3)

                outcome_team_1_kal, outcome_team_2_kal, predict_team_1_prop_1, predict_team_2_prop_1 = monte_carlo_sim(final_dict_1_dn_kal,dnn_class=self.dnn_class,n_simulations=n_simulations,team_1=self.team_1,team_2=self.team_2,pred_type='kalman')
                # outcome_team_2_2nd_kal, outcome_team_1_2nd_kal, predict_team_2_prop_2, predict_team_1_prop_2  = monte_carlo_sim(final_dict_2_dn_kal,dnn_class=self.dnn_class,n_simulations=n_simulations,team_1=self.team_2,team_2=self.team_1,pred_type='kalman')

                final_prop_team_1_kalm = round(np.mean([predict_team_1_prop_1])*100,3)
                final_prop_team_2_kalm = round(np.mean([predict_team_2_prop_1])*100,3)

                # outcome_team_2_2nd, outcome_team_1_2nd = monte_carlo_sim(final_dict_2_dn,self.dnn_class,n_simulations)
                print('============================================')
                print(f'Probas ewm: {outcome_team_1_ewm}, {outcome_team_2_ewm}')
                print(f'Probas kalman: {outcome_team_1_kal}, {outcome_team_2_kal}')
                print('============================================')
                outcome_1_ewm = round(np.mean(outcome_team_1_ewm)*100,3)
                outcome_2_ewm = round(np.mean(outcome_team_2_ewm)*100,3)

                outcome_1_kal = round(np.mean(outcome_team_1_kal)*100,3)
                outcome_2_kal = round(np.mean(outcome_team_2_kal)*100,3)
                # outcome_1 = np.mean([round(np.mean(outcome_team_1)*100,3),round(np.mean(outcome_team_1_2nd)*100,3)])
                # outcome_2 = np.mean([round(np.mean(outcome_team_2)*100,3),round(np.mean(outcome_team_2_2nd)*100,3)])
                results_dict = {
                    'Team 1': [self.team_1],
                    'Team 2': [self.team_2],
                    'Team 1 Probability EWM': [outcome_1_ewm],
                    'Team 2 Probability EWM': [outcome_2_ewm],
                    'Team 1 Probability KAL': [outcome_1_kal],
                    'Team 2 Probability KAL': [outcome_2_kal],
                    'Team 1 Probability EWM Prop': [final_prop_team_1_ewm],
                    'Team 2 Probability EWM Prop': [final_prop_team_2_ewm],
                    'Team 1 Probability KAL Prop': [final_prop_team_1_kalm],
                    'Team 2 Probability KAL Prop': [final_prop_team_2_kalm],
                    
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
