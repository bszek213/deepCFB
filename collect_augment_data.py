#collect data and sort them for multi-class classification
#I am going to try this first without cfbd data

from collect_data import read_api_key_from_yaml
import cfbd
import requests
from bs4 import BeautifulSoup
from pandas import DataFrame, read_csv, concat
import cfbd
from numpy import nan, where
from time import sleep
from os.path import join, exists
from os import getcwd, remove
from tqdm import tqdm
import yaml

def requests_sports_ref(URL):
    while True:
        try:
            page = requests.get(URL)
            soup = BeautifulSoup(page.content, "html.parser")
            break
        except:
            print('HTTPSConnectionPool(host="www.sports-reference.com", port=443): Max retries exceeded. Retry in 10 seconds')
            sleep(10)
    return soup

def augment_team_names(team):
    team = team.replace(' ', '-').lower()
    if '.' in team:
        team = team.replace(".", "")
    if 'the' in team:
        team = team.replace("the-", "")
    if '&' in team:
        team = team.replace("&", "")
    if '(' in team and ')' in team:
        team = team.replace("(", "")
        team = team.replace(")", "")
    if "'" in team:
        team = team.replace("'", "")
    return team

def extract_features_from_other_team(team,team_playing,year):
    sleep(2)
    try:
        URL = 'https://www.sports-reference.com/cfb/schools/' + team + '/' + str(year) + '/gamelog/'
        soup = requests_sports_ref(URL)
        table = soup.find(id="div_offense")
        tbody = table.find('tbody')
    except:
        print(f'data for {team} does not exist')
        return None
    tr_body = tbody.find_all('tr')
    game_result = []
    turnovers = []
    pass_cmp = []
    pass_att = []
    pass_yds = []
    pass_td = []
    rush_att = []
    rush_yds = []
    rush_td = []
    rush_yds_per_att = []
    tot_plays = []
    tot_yds_per_play = []
    first_down_pass = []
    first_down_rush = []
    first_down_penalty = []
    first_down = []
    penalty = []
    penalty_yds = []
    fumbles_lost = []
    pass_int = []
    game_result = []
    #gets all games
    for trb in tr_body:
        #gets invidual game data
        for td in trb.find_all('td'):
            if td.get('data-stat') == "opp_name":
                if '*' in td.get_text():
                    text_data = td.get_text().replace('*','')
                else:
                    text_data = td.get_text()
                opp_team = augment_team_names(text_data)
                # print('=====')
                # print(opp_team)
                # print(team_playing)
                # print('=====')
                # input()
                # if opp_team == team_playing:
            # if td.get('data-stat') == "game_result":
            #     result = td.get_text()
                
            if td.get('data-stat') == "turnovers":
                turn = td.get_text()
                
            if td.get('data-stat') == "pass_cmp":
                pass_c = td.get_text()
                
            if td.get('data-stat') == "pass_att":
                pass_a = td.get_text()
                
            if td.get('data-stat') == "pass_cmp_pct":
                pass_c_p = td.get_text()
                
            if td.get('data-stat') == "pass_yds":
                pass_y= td.get_text()
                
            if td.get('data-stat') == "pass_td":
                pass_t = td.get_text()
                
            if td.get('data-stat') == "rush_att":
                rush_a = td.get_text()
                
            if td.get('data-stat') == "rush_yds":
                rush_y = td.get_text()
                
            if td.get('data-stat') == "rush_yds_per_att":
                rush_y_p_a = td.get_text()
                
            if td.get('data-stat') == "rush_td":
                rush_t = td.get_text()
                
            if td.get('data-stat') == "tot_plays":
                tot_p = td.get_text()
                
            if td.get('data-stat') == "tot_yds_per_play":
                tot_y_p_a = td.get_text()
                
            if td.get('data-stat') == "first_down_pass":
                first_d_p = td.get_text()
                
            if td.get('data-stat') == "first_down_rush":
                first_d_r = td.get_text()
                
            if td.get('data-stat') == "first_down_penalty":
                first_d_p = td.get_text()
                
            if td.get('data-stat') == "first_down":
                first_d = td.get_text()
                
            if td.get('data-stat') == "penalty":
                pen = td.get_text()
                
            if td.get('data-stat') == "penalty_yds":
                pen_y = td.get_text()
                
            if td.get('data-stat') == "pass_int":
                pass_i = td.get_text()
                
            if td.get('data-stat') == "fumbles_lost":
                fum_lost = td.get_text()
                
        if opp_team == team_playing:
            # game_result.append(result)
            turnovers.append(turn)
            pass_cmp.append(pass_c)
            pass_att.append(pass_a)
            pass_att.append(pass_c_p)
            pass_yds.append(pass_y)
            pass_td.append(pass_t)
            rush_att.append(rush_a)
            rush_yds.append(rush_y)
            rush_yds_per_att.append(rush_y_p_a)
            rush_td.append(rush_t)
            tot_plays.append(tot_p)
            first_down_penalty.append(first_d_p)
            tot_yds_per_play.append(tot_y_p_a)
            first_down_pass.append(first_d_p)
            first_down_rush.append(first_d_r)
            first_down.append(first_d)
            penalty.append(pen)
            penalty_yds.append(pen_y)
            pass_int.append(pass_i)
            fumbles_lost.append(fum_lost)
            return DataFrame(list(zip(turnovers,pass_cmp,pass_att,pass_yds,
                                pass_td,
                                rush_att,
                                rush_yds,
                                rush_td, 
                                rush_yds_per_att,
                                tot_plays,
                                tot_yds_per_play,
                                first_down_pass,
                                first_down_rush,
                                first_down_penalty,
                                first_down,
                                penalty,
                                penalty_yds,
                                fumbles_lost,
                                pass_int)),
                                columns =['turnovers_opp', 'pass_cmp_opp', 'pass_att_opp', 'pass_yds_opp', 'pass_td_opp', 'rush_att_opp', 
                                'rush_yds_opp', 'rush_td_opp', 'rush_yds_per_att_opp', 'tot_plays_opp', 'tot_yds_per_play_opp',
                                'first_down_pass_opp', 'first_down_rush_opp', 'first_down_penalty_opp', 'first_down_opp', 'penalty_opp', 'penalty_yds_opp', 'fumbles_lost_opp',
                                'pass_int_opp'])


def collect_two_teams(URL,team,year):
    soup = requests_sports_ref(URL)
    table = soup.find(id="div_offense")
    tbody = table.find('tbody')
    tr_body = tbody.find_all('tr')
    #current_team features
    final_df = DataFrame()
    for trb in tr_body:
        game_result = []
        turnovers = []
        pass_cmp = []
        pass_att = []
        pass_yds = []
        pass_td = []
        rush_att = []
        rush_yds = []
        rush_td = []
        rush_yds_per_att = []
        tot_plays = []
        tot_yds_per_play = []
        first_down_pass = []
        first_down_rush = []
        first_down_penalty = []
        first_down = []
        penalty = []
        penalty_yds = []
        fumbles_lost = []
        pass_int = []
        game_loc = []
        for td in trb.find_all('td'):
            if td.get('data-stat') == "opp_name":
                if '*' in td.get_text():
                    text_data = td.get_text().replace('*','')
                else:
                    text_data = td.get_text()
                opp_team = augment_team_names(text_data)
                print(f'opponent: {opp_team}')
                opp_df = extract_features_from_other_team(opp_team,team,year)
            #current team feature assignment
            if td.get('data-stat') == "game_result":
                print(f'outcome: {td.get_text()}')
                result = td.get_text()

            if td.get('data-stat') == "turnovers":
                turn = td.get_text()
                
            if td.get('data-stat') == "pass_cmp":
                pass_c = td.get_text()
                
            if td.get('data-stat') == "pass_att":
                pass_a = td.get_text()
                
            if td.get('data-stat') == "pass_cmp_pct":
                pass_c_p = td.get_text()
                
            if td.get('data-stat') == "pass_yds":
                pass_y= td.get_text()
                
            if td.get('data-stat') == "pass_td":
                pass_t = td.get_text()
                
            if td.get('data-stat') == "rush_att":
                rush_a = td.get_text()
                
            if td.get('data-stat') == "rush_yds":
                rush_y = td.get_text()
                
            if td.get('data-stat') == "rush_yds_per_att":
                rush_y_p_a = td.get_text()
                
            if td.get('data-stat') == "rush_td":
                rush_t = td.get_text()
                
            if td.get('data-stat') == "tot_plays":
                tot_p = td.get_text()
                
            if td.get('data-stat') == "tot_yds_per_play":
                tot_y_p_a = td.get_text()
                
            if td.get('data-stat') == "first_down_pass":
                first_d_p = td.get_text()
                
            if td.get('data-stat') == "first_down_rush":
                first_d_r = td.get_text()
                
            if td.get('data-stat') == "first_down_penalty":
                first_d_p = td.get_text()
                
            if td.get('data-stat') == "first_down":
                first_d = td.get_text()
                
            if td.get('data-stat') == "penalty":
                pen = td.get_text()
                
            if td.get('data-stat') == "penalty_yds":
                pen_y = td.get_text()
                
            if td.get('data-stat') == "pass_int":
                pass_i = td.get_text()
                
            if td.get('data-stat') == "fumbles_lost":
                fum_lost = td.get_text()
            if td.get('data-stat') == "game_location":
                if td.get_text() == '@':
                    loc = 0 #away
                else:
                    loc = 1 #home
            
            #team feature assignment
            #if the data are not there for opp do not save
        if opp_df is not None and isinstance(opp_df, DataFrame):
            game_result.append(result)
            turnovers.append(turn)
            pass_cmp.append(pass_c)
            pass_att.append(pass_a)
            pass_att.append(pass_c_p)
            pass_yds.append(pass_y)
            pass_td.append(pass_t)
            rush_att.append(rush_a)
            rush_yds.append(rush_y)
            rush_yds_per_att.append(rush_y_p_a)
            rush_td.append(rush_t)
            tot_plays.append(tot_p)
            first_down_penalty.append(first_d_p)
            tot_yds_per_play.append(tot_y_p_a)
            first_down_pass.append(first_d_p)
            first_down_rush.append(first_d_r)
            first_down.append(first_d)
            penalty.append(pen)
            penalty_yds.append(pen_y)
            pass_int.append(pass_i)
            fumbles_lost.append(fum_lost)
            game_loc.append(loc)
            df = DataFrame(list(zip(game_result,turnovers,pass_cmp,pass_att,pass_yds,
                                    pass_td,
                                    rush_att,
                                    rush_yds,
                                    rush_td, 
                                    rush_yds_per_att,
                                    tot_plays,
                                    tot_yds_per_play,
                                    first_down_pass,
                                    first_down_rush,
                                    first_down_penalty,
                                    first_down,
                                    penalty,
                                    penalty_yds,
                                    fumbles_lost,
                                    pass_int,
                                    game_loc)),
                                    columns =['game_result','turnovers', 'pass_cmp', 'pass_att', 'pass_yds', 'pass_td', 'rush_att', 
                                    'rush_yds', 'rush_td', 'rush_yds_per_att', 'tot_plays', 'tot_yds_per_play',
                                    'first_down_pass', 'first_down_rush', 'first_down_penalty', 'first_down', 'penalty', 'penalty_yds', 'fumbles_lost',
                                    'pass_int','game_loc'])
            df = concat([df, opp_df], axis=1)
            final_df = concat([final_df, df])
            print(final_df)
            sleep(4)
    return final_df
def get_teams():
        year_list_find = []
        year_list = [2023,2022,2021,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010]#,2009,2008,2007,2006,2005,2004,2003,2002,2001,2000]
        #all teams with data
        # all_teams = get_teams_year(min(year_list),2023)
        #select only the top 30 teams
        with open('top_30_teams.txt','r') as file:
             content = file.read()
        all_teams = content.split("\n")
        all_teams = [string for string in all_teams if string.strip() != ""]

        if exists(join(getcwd(),'year_count.yaml')):
            with open(join(getcwd(),'year_count.yaml')) as file:
                year_counts = yaml.load(file, Loader=yaml.FullLoader)
        else:
            year_counts = {'year':year_list_find}
        #remove the current year to always update with the latest games
        # year_counts['year'].remove(2023)
        if year_counts['year']:
            year_list_check =  year_counts['year']
            year_list_find = year_counts['year']
            year_list = [i for i in year_list if i not in year_list_check]
            print(f'Need data for year: {year_list}')
        if year_list:
            for year in year_list:
                # all_teams = Teams(year)
                # team_names = all_teams.dataframes.abbreviation
                # team_names = team_names.sort_values()   
                final_list = []
                x_feature_regress, y_feature_regress = [], []
                for abv in tqdm(all_teams):  
                    try:  
                        print('')#tqdm thing
                        print(f'current team: {abv}, year: {year}')
                        # team = all_teams(abv)
                        str_combine = 'https://www.sports-reference.com/cfb/schools/' + abv.lower() + '/' + str(year) + '/gamelog/'
                        df_inst = collect_two_teams(str_combine,abv.lower(),year)
                        print(df_inst)
                        final_list.append(df_inst)
                        if len(df_inst) % 2 != 0:
                            df_inst = df_inst.iloc[:-1]
                        x_feature_regress.append(df_inst.iloc[::2])  # Odd rows
                        y_feature_regress.append(df_inst.iloc[1::2])  # Even rows
                    except:
                        print(f'{abv} has no data for {year}')
                output = concat(final_list)
                final_data_x_regress = concat(x_feature_regress)
                final_data_y_regress = concat(y_feature_regress)
                final_data = output.replace(r'^\s*$', nan, regex=True) #replace empty string with NAN

                if year == 2023:
                    #Remove old data so that you can update with the most recent games
                    if exists(join(getcwd(),'all_data_2023.csv')):
                        remove('all_data_2023.csv')
                        remove('x_feature_regression_2023.csv')
                        remove('y_feature_regression_2023.csv')
                    if not exists(join(getcwd(),'all_data_2023.csv')):
                        final_data.to_csv(join(getcwd(),'all_data_2023.csv'),index=False)
                        final_data_x_regress.to_csv(join(getcwd(),'x_feature_regression_2023.csv'),index=False)
                        final_data_y_regress.to_csv(join(getcwd(),'y_feature_regression_2023.csv'),index=False)
                else:
                    if exists(join(getcwd(),'all_data.csv')):
                        all_data = read_csv(join(getcwd(),'all_data.csv'))  
                        all_data = concat([all_data, final_data.dropna()])
                        all_data.to_csv(join(getcwd(),'all_data.csv'),index=False)
                        x_regress = read_csv(join(getcwd(),'x_feature_regression.csv')) 
                        x_regress = concat([x_regress, final_data_x_regress])
                        x_regress.to_csv(join(getcwd(),'x_feature_regression.csv'),index=False)
                        y_regress = read_csv(join(getcwd(),'y_feature_regression.csv')) 
                        y_regress = concat([y_regress, final_data_y_regress])
                        y_regress.to_csv(join(getcwd(),'y_feature_regression.csv'),index=False)

                    if not exists(join(getcwd(),'all_data.csv')):
                        final_data.to_csv(join(getcwd(),'all_data.csv'),index=False)
                        final_data_x_regress.to_csv(join(getcwd(),'x_feature_regression.csv'),index=False)
                        final_data_y_regress.to_csv(join(getcwd(),'y_feature_regression.csv'),index=False)

                year_list_find.append(year)
                print(f'year list after loop: {year_list_find}')
                with open(join(getcwd(),'year_count.yaml'), 'w') as write_file:
                    yaml.dump(year_counts, write_file)
                    print(f'writing {year} to yaml file')
def main():
    get_teams()  
if __name__ == "__main__":
    main()