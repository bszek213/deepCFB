#collect data from SportsReference.com
# -*- coding: utf-8 -*-
"""
html parse code - cfb
"""
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
# from cfbd.rest import ApiException

def get_teams_year(year_min,year_max):
    #Read in from csv
    teams_save = []
    teams = read_csv('all_schools.csv')
    filtered_df = teams[(teams["From"] <= year_min) & (teams["To"] == year_max)]
    for team in filtered_df['School']:
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
        teams_save.append(team)
    return teams_save

def html_to_df_web_scrape(URL,team,year):
    configuration = cfbd.Configuration()
    configuration.api_key['Authorization'] = 'UK4ikHBmxuHDyMlNngTZS8sokyl8Kr4FExP2NRb9G8qaFOUrUhX3xy6+OxQv4oEX'
    configuration.api_key_prefix['Authorization'] = 'Bearer'
    api_instance = cfbd.GamesApi(cfbd.ApiClient(configuration))
    api_game = cfbd.GamesApi(cfbd.ApiClient(configuration))
    # URL EXAMPLE: URL = "https://www.sports-reference.com/cfb/schools/georgia/2021/gamelog/"
    while True:
        try:
            page = requests.get(URL)
            soup = BeautifulSoup(page.content, "html.parser")
            break
        except:
            print('HTTPSConnectionPool(host="www.sports-reference.com", port=443): Max retries exceeded. Retry in 10 seconds')
            sleep(10)
    table = soup.find(id="div_offense")
    tbody = table.find('tbody')
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
    game_loc = []
    havoc = []
    start_field = []
    scoring_opp = []
    power_success = []
    second_level_yards = []
    open_field_yards = []
    line_yards = []
    success_rate = []
    explosiveness = []
    for trb in tr_body:
        for td in trb.find_all('td'):
            if td.get('data-stat') == "opp_name":
                if '*' in td.get_text():
                    text_data = td.get_text().replace('*','')
                else:
                    text_data = td.get_text()
                # print('opp:',text_data)
                # print('team:',team)
                if text_data == 'Miami (FL)':
                    text_data = 'Miami'
                elif text_data == 'Mississippi':
                    text_data = 'Ole Miss'
                elif text_data == 'Louisiana-state':
                    text_data = 'LSU'
                elif text_data == 'Louisiana State':
                    text_data = 'LSU'
                elif text_data == 'Nevada-Las Vegas':
                    text_data = 'UNLV'
                elif text_data == 'Bowling Green State':  
                    text_data = 'Bowling Green'
                elif text_data == 'UTSA':
                    text_data = 'UT San Antonio'
                elif text_data   == 'Brigham Young':
                    text_data = 'BYU'
                elif text_data   == 'Southern California':
                    text_data = 'USC'
                elif text_data   == 'Massachusetts': 
                    text_data = 'Umass'
                elif text_data   == 'Central Florida': 
                    text_data = 'UCF'
                elif text_data   == 'North Carolina State': 
                    text_data = 'NC State'
                elif text_data == 'Alabama-Birmingham':
                    text_data = 'UAB'
                elif text_data == 'Southern Methodist':
                    text_data = 'SMU'
                elif text_data == 'Middle Tennessee State':
                    text_data = 'Middle Tennessee'
                elif text_data == 'San Jose State':
                    text_data = 'San José State'
                elif text_data == 'Hawaii': 
                    text_data = "Hawai'i"
                elif text_data == 'St. Francis (PA)':
                    text_data = 'St Francis (PA)'
                elif text_data == 'Long Island':
                    text_data = 'Long Island University' 
                elif text_data == 'Grambling State':
                    text_data = 'Grambling'
                elif text_data == 'Virginia Military Institute': 
                    text_data = 'VMI'
                elif text_data == 'Nicholls State': 
                    text_data = 'Nicholls'
                elif text_data == 'McNeese State': 
                    text_data = 'McNeese'
                elif text_data == 'Central Connecticut State': 
                    text_data = 'Central Connecticut'
                elif text_data == 'Prairie View A&M': 
                    text_data = 'Prairie View'
                elif text_data == 'California-Davis': 
                    text_data = 'UC Davis'
                elif text_data == 'Tennessee-Martin': 
                    text_data = 'UT Martin'
                elif text_data == 'Presbyterian': 
                    text_data = 'Presbyterian College'
                else:
                    text_data = text_data
                if '-' in text_data:
                    text_data = text_data.replace('-',' ')
                if '-' in team:
                    team = team.replace('-',' ')
                #Execptions since no one can chose a syntax and stick to it
                if text_data == 'Arkansas Pine Bluff':
                    text_data = 'Arkansas-Pine Bluff'
                if text_data == 'Bethune Cookman':
                    text_data = 'Bethune-Cookman'
                if text_data == 'Gardner Webb':
                    text_data = 'Gardner-Webb'
                #Fix team names
                if team == 'alabama birmingham':
                    team = 'UAB'
                if team == 'bowling green state':
                    team = 'Bowling Green'
                if team == 'brigham young':
                    team = 'BYU'
                if team == 'central florida':
                    team = 'UCF'
                if team == 'hawaii':
                    team = "Hawai'i"
                if team == 'louisiana lafayette':
                    team = "louisiana"
                if team == 'louisiana state':
                    team = "LSU"
                if team == 'massachusetts':
                    team = "Umass"
                if team == 'southern methodist':
                    team = "SMU"
                if team == 'texas christian':
                    team = "TCU" 
                if team == 'texas san antonio':
                    team = "UTSA"
                if team == 'texas el paso':
                    team = "UTEP"
                if team == 'nevada las vegas':
                    team = "UNLV"
                if team == 'southern california':
                    team = "USC"
                if team == 'miami oh':
                    team = "miami (oh)"
                if team == 'miami fl':
                    team = "Miami"
                if team == 'texas am':
                    team = "texas a&m"
                if team == 'middle tennessee state':
                    team = "middle tennessee"
                if team == 'mississippi':
                    team = "ole miss"
                if team == 'north carolina state':
                    team = "nc state"
                if team == 'san jose state': 
                    team = 'San José State'
                if team == 'texas-san-antonio':
                    team = 'UT San Antonio'
                if team == 'UTSA':
                    team = 'UT San Antonio'
                if '*' in text_data:
                    text_data = text_data.replace('*','')
                print('opp:',text_data)
                print('team:',team)

                while True:
                    try:
                        api_response = api_instance.get_games(year, season_type='regular',
                                                              
                                                              home=text_data, away=team,  #add a if statement here to say if null switch home and away
                                                              )
                        if not api_response:
                            api_response = api_instance.get_games(year, season_type='regular',
                                                                  
                                                                  home=team, away=text_data,  #add a if statement here to say if null switch home and away
                                                                  )
                        if not api_response:
                            api_response = api_instance.get_games(year, season_type='postseason',
                                                                  
                                                                  home=team, away=text_data,  #add a if statement here to say if null switch home and away
                                                                  )
                        if not api_response:
                            api_response = api_instance.get_games(year, season_type='postseason',
                                                                  
                                                                  home=text_data, away=team,  #add a if statement here to say if null switch home and away
                                                                  )
                        # print(api_response)
                        break
                    except:
                        print('Reason: Unauthorized, retry in 10 seconds')
                        sleep(10)
                retry_count = 0
                while True:
                    try:
                        api_response_2 = api_game.get_advanced_box_score(api_response[0].id)
                        break
                    except:
                        print('Reason: Internal Server Error, retry in 2 seconds')
                        retry_count += 1
                        sleep(2)   
                    if retry_count == 1:
                        print('Maximum number of retries reached. Exiting loop.')
                        api_response_2 = []
                        break
                # print(api_response_2.teams['successRates'])
                # if api_response_2:
                    #Example
                    # [{'line_yards': 103.0,
                    #      'line_yards_average': 2.6,
                    #      'open_field_yards': 1,
                    #      'open_field_yards_average': 0.0,
                    #      'power_success': 0.727,
                    #      'second_level_yards': 25,
                    #      'second_level_yards_average': 0.6,
                    #      'stuff_rate': 0.154,
                    #      'team': 'Rutgers'}, {'line_yards': 41.0,
                    #      'line_yards_average': 1.7,
                    #      'open_field_yards': 46,
                    #      'open_field_yards_average': 1.9,
                    #      'power_success': 1.0,
                    #      'second_level_yards': 17,
                    #      'second_level_yards_average': 0.7,
                    #      'stuff_rate': 0.25,
                    #      'team': 'Syracuse'}]
                    # sleep(1000)
                try:
                    #havoc
                    if api_response_2.teams.havoc[0].team.capitalize() == team.capitalize():
                        havoc.append(api_response_2.teams.havoc[0].total)
                    else:
                        havoc.append(api_response_2.teams.havoc[1].total)
                    #field_position
                    if api_response_2.teams.field_position[0].team.capitalize() == team.capitalize():
                        start_field.append(api_response_2.teams.field_position[0].average_start)
                    else:
                        start_field.append(api_response_2.teams.field_position[1].average_start)
                    #scoring_opportunities
                    if api_response_2.teams.scoring_opportunities[0].team.capitalize() == team.capitalize():
                        scoring_opp.append(api_response_2.teams.scoring_opportunities[0].points_per_opportunity)
                    else:
                        scoring_opp.append(api_response_2.teams.scoring_opportunities[1].points_per_opportunity)
                    #power_success
                    if api_response_2.teams.rushing[0].team.capitalize() == team.capitalize():
                        power_success.append(api_response_2.teams.rushing[0].power_success)
                    else:
                        power_success.append(api_response_2.teams.rushing[1].power_success)
                    #second_level_yards
                    if api_response_2.teams.rushing[0].team.capitalize() == team.capitalize():
                        second_level_yards.append(api_response_2.teams.rushing[0].second_level_yards)
                    else:
                        second_level_yards.append(api_response_2.teams.rushing[1].second_level_yards)
                    #open_field_yards
                    if api_response_2.teams.rushing[0].team.capitalize() == team.capitalize():
                        open_field_yards.append(api_response_2.teams.rushing[0].open_field_yards)
                    else:
                        open_field_yards.append(api_response_2.teams.rushing[1].open_field_yards)
                    #line_yards
                    if api_response_2.teams.rushing[0].team.capitalize() == team.capitalize():
                        line_yards.append(api_response_2.teams.rushing[0].line_yards)
                    else:
                        line_yards.append(api_response_2.teams.rushing[1].line_yards)
                    #Success rate   
                    if api_response_2.teams.rushing[0].team.capitalize() == team.capitalize():
                        success_rate.append(api_response_2.teams.success_rates[0].overall.total)
                    else:
                        success_rate.append(api_response_2.teams.success_rates[1].overall.total)
                    #Explosiveness   
                    if api_response_2.teams.rushing[0].team.capitalize() == team.capitalize():
                        explosiveness.append(api_response_2.teams.explosiveness[0].overall.total)
                    else:
                        explosiveness.append(api_response_2.teams.explosiveness[1].overall.total)
                except:
                    print('Key error - team. most likely the there are no data. return NaN')
                    havoc.append(nan)
                    start_field.append(nan)
                    scoring_opp.append(nan)
                    power_success.append(nan)
                    second_level_yards.append(nan)
                    open_field_yards.append(nan)
                    line_yards.append(nan)
                    success_rate.append(nan)
                    explosiveness.append(nan)
                # else:
                #     havoc.append(nan)
                #     start_field.append(nan)
                #     scoring_opp.append(nan)
                #     start_field.append(nan)
                #     power_success.append(nan)
                #     second_level_yards.append(nan)
                #     open_field_yards.append(nan)
                #     line_yards.append(nan)
            if td.get('data-stat') == "game_result":
                game_result.append(td.get_text())
            if td.get('data-stat') == "turnovers":
                turnovers.append(td.get_text())
            if td.get('data-stat') == "pass_cmp":
                pass_cmp.append(td.get_text())
            if td.get('data-stat') == "pass_att":
                pass_att.append(td.get_text())
            if td.get('data-stat') == "pass_cmp_pct":
                pass_att.append(td.get_text())
            if td.get('data-stat') == "pass_yds":
                pass_yds.append(td.get_text())
            if td.get('data-stat') == "pass_td":
                pass_td.append(td.get_text())
            if td.get('data-stat') == "rush_att":
                rush_att.append(td.get_text())
            if td.get('data-stat') == "rush_yds":
                rush_yds.append(td.get_text())
            if td.get('data-stat') == "rush_yds_per_att":
                rush_yds_per_att.append(td.get_text())
            if td.get('data-stat') == "rush_td":
                rush_td.append(td.get_text())
            if td.get('data-stat') == "tot_plays":
                tot_plays.append(td.get_text())
            if td.get('data-stat') == "tot_yds_per_play":
                tot_yds_per_play.append(td.get_text())
            if td.get('data-stat') == "first_down_pass":
                first_down_pass.append(td.get_text())
            if td.get('data-stat') == "first_down_rush":
                first_down_rush.append(td.get_text())
            if td.get('data-stat') == "first_down_penalty":
                first_down_penalty.append(td.get_text())
            if td.get('data-stat') == "first_down":
                first_down.append(td.get_text())
            if td.get('data-stat') == "penalty":
                penalty.append(td.get_text())
            if td.get('data-stat') == "penalty_yds":
                penalty_yds.append(td.get_text())
            if td.get('data-stat') == "pass_int":
                pass_int.append(td.get_text())
            if td.get('data-stat') == "fumbles_lost":
                fumbles_lost.append(td.get_text())
            if td.get('data-stat') == "game_location":
                if td.get_text() == '@':
                    game_loc.append(0) #away
                else:
                    game_loc.append(1) #home
        sleep(2)
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
    pass_int,havoc,
    game_loc,
    scoring_opp,
    start_field,
    power_success,
    second_level_yards,
    open_field_yards,
    line_yards,
    success_rate,
    explosiveness)),
                columns =['game_result','turnovers', 'pass_cmp', 'pass_att', 'pass_yds', 'pass_td', 'rush_att', 
                   'rush_yds', 'rush_td', 'rush_yds_per_att', 'tot_plays', 'tot_yds_per_play',
                   'first_down_pass', 'first_down_rush', 'first_down_penalty', 'first_down', 'penalty', 'penalty_yds', 'fumbles_lost',
                   'pass_int','havoc','game_loc','points_per_opp','average_field_start','power_success','second_level_yards',
                   'open_field_yards','line_yards','success_rate','explosiveness'])
    # print(df)
    # input()
    return df.dropna()

def get_teams():
        year_list_find = []
        year_list = [2023,2022,2021,2019,2018,2017,2016,2015]#,2014,2013,,2012,2011,2010,2009,2008,2007,2006,2005,2004,2003,2002,2001,2000]
        all_teams = get_teams_year(min(year_list),2023)
        if exists(join(getcwd(),'year_count.yaml')):
            with open(join(getcwd(),'year_count.yaml')) as file:
                year_counts = yaml.load(file, Loader=yaml.FullLoader)
        else:
            year_counts = {'year':year_list_find}
        #remove the current year to always update with the latest games
        year_counts['year'].remove(2023)
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
                        df_inst = html_to_df_web_scrape(str_combine,abv.lower(),year)
                        # df_inst = html_to_df_web_scrape(str_combine)
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
        # else:
        #     all_data = read_csv(join(getcwd(),'all_data.csv'))
        # print('length of data: ', len(all_data))

def main():
    get_teams()
if __name__ == "__main__":
    main()