#Simple rating systme to assess how "good" a team is

from collect_data import data_for_srs
from collect_augment_data import augment_team_names
from numpy import median, absolute
from math import isnan
from pandas import DataFrame, read_csv
from tqdm import tqdm
from matplotlib.pyplot import show, xticks, tight_layout, savefig
import argparse
from os.path import exists
from colorama import Fore, Style
"""
Rating = Team's Average Point Differential - Strength of Schedule
and then rank each team
strength of schedule is the opposing team's point differential
"""

def get_df(team):
    if team == 'tcu':
        team = 'texas-christian'
    if team == 'lafayette':
        team = 'louisiana-lafayette'
    if team == 'utsa':
        team = 'texas-san-antonio'
    if team == 'utep':
        team = 'texas-el-paso'
    if team == 'sam-houston':
        team = "sam-houston-state"
    if team == 'louisiana':
        team = 'louisiana-lafayette'
    str_combine = 'https://www.sports-reference.com/cfb/schools/' + team + '/' + str(2023) + '/gamelog/'
    team, opps  = data_for_srs(str_combine,team,2023)
    return team, opps

def teams_to_test():
    with open('top_30_teams.txt','r') as file:
        content = file.read()
    all_teams = content.split("\n")
    all_teams = [string for string in all_teams if string.strip() != ""]
    return all_teams

def get_pt_diff_team_1(df):
    df[['team_1_score', 'team_2_score']] = df['game_result'].str.extract(r'(\d+)-(\d+)').astype(int)
    df['differential'] = df['team_1_score'] - df['team_2_score']
    return df['differential'].median()

def fix_school_names(input_list):
    processed_school_names = []
    for name in input_list:
        # Remove parentheses and ampersands
        cleaned_name = name.replace("(", "").replace(")", "").replace("&", "")
        lowercase_name = cleaned_name.lower()
        processed_name = lowercase_name.replace(" ", "-")
        if processed_name == 'tcu':
            processed_name = 'texas-christian'
        if processed_name == 'lafayette':
            processed_name = 'louisiana-lafayette'
        if processed_name == 'utsa':
            processed_name = 'texas-san-antonio'
        # Append the processed name to the new list
        processed_school_names.append(processed_name)
    return processed_school_names

def get_pt_diff_team_2(team):
    team[['team_1_score', 'team_2_score']] = team['game_result'].str.extract(r'(\d+)-(\d+)').astype(int)
    team['differential'] = team['team_1_score'] - team['team_2_score']
    #may attempt to help improve on SRS - only include games that team lost to detract from team_1's overall pt diff
    update_values = team['differential'][team['differential'] < 0].to_numpy()
    return median(absolute(update_values))

def main():
    parser = argparse.ArgumentParser(description="SRS - my version")
    parser.add_argument('--all', type=str, choices=['yes', 'no'], help='Perform a teams_to_test() function when "yes" is provided, otherwise if no, input team names.')
    parser.add_argument('--team_1', type=str, help='Name of team 1')
    parser.add_argument('--team_2', type=str, help='Name of team 2')
    args = parser.parse_args()
    #save outputs
    team_dict = {}
    if args.all == 'yes':
        team_output_list = (teams_to_test())
    else:
        team_output_list = []
        # Input the two teams of interests
        team_output_list.append(args.team_1)#input('input team_1: ')
        team_output_list.append(args.team_2)#input('input team_2: ')
    if not exists('my_srs.csv') and args.all == 'yes':
        for teams in tqdm(team_output_list):
            print(f'current team: {teams}')
            team, opps = get_df(teams)
            team1_pt_diff = get_pt_diff_team_1(team)
            processed_school_names = fix_school_names(opps)
            # print(processed_school_names)
            opp_team_averages = []
            for opp_team in processed_school_names:
                try:
                    team, opps = get_df(opp_team)
                    opp_diff = float(get_pt_diff_team_2(team))
                    # print(f'{opp_team}: {opp_diff}')
                    if isnan(opp_diff):
                        opp_diff = 0
                    # print(f'{opp_team}: {opp_diff}')
                    opp_team_averages.append(opp_diff)
                except:
                    print(f'{opp_team} does not not have data. Check spelling or some teams do not have data')

            #Calc SRS
            srs = team1_pt_diff - median(opp_team_averages)
            team_dict[teams] = srs
            print(team_dict)
        #df and sorting
        final_df = DataFrame({'Teams': list(team_dict.keys()), 'SRS': list(team_dict.values())})
        sorted_teams = final_df.sort_values(by='SRS', ascending=False)
        if args.all == 'yes':
            sorted_teams.to_csv('my_srs.csv',index=False)
        print('=======================')
        print(f'SRS output')
        print(team_dict)
        print('=======================')
    else:
        sorted_teams = read_csv('my_srs.csv')
        for teams in team_output_list:
            print(f'current team: {teams}')
            team, opps = get_df(teams)
            team1_pt_diff = get_pt_diff_team_1(team)
            processed_school_names = fix_school_names(opps)
            # print(processed_school_names)
            opp_team_averages = []
            for opp_team in processed_school_names:
                try:
                    team, opps = get_df(opp_team)
                    opp_diff = float(get_pt_diff_team_2(team))
                    # print(f'{opp_team}: {opp_diff}')
                    if isnan(opp_diff):
                        opp_diff = 0
                    # print(f'{opp_team}: {opp_diff}')
                    opp_team_averages.append(opp_diff)
                except:
                    print(f'{opp_team} does not not have data. Check spelling or some teams do not have data')

            #Calc SRS
            srs = team1_pt_diff - median(opp_team_averages)
            team_dict[teams] = srs
        print(Fore.YELLOW + Style.BRIGHT + '=======================')
        print(f'Two teams SRS output')
        print(team_dict)
        print('=======================' + Style.RESET_ALL)
    if args.all == 'yes':
        sorted_teams.plot.bar(x='Teams',y='SRS',figsize=(20, 6))
        tight_layout()
        xticks(rotation=45)
        savefig('my_srs.png',dpi=400)

if __name__ == "__main__":
    main()