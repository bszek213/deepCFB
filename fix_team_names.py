import re

def format_team_names():
    file_name = "teams_played_this_week.txt"
    with open(file_name, 'r') as file:
        input_string = file.read()
    # Split the input string into lines
    lines = input_string.split('\n')
    
    formatted_lines = []
    for line in lines:
        if '@' in line:
            #split the line into two teams
            teams = line.split('@')

            #remove whitespace and remove numbers
            team1 = re.sub(r'^\d+', '', teams[0].strip()).strip()
            team2 = re.sub(r'^\d+', '', teams[1].strip()).strip()

            # Replace spaces with hyphens and convert to lowercase
            team1 = team1.replace(' ', '-').lower()
            team2 = team2.replace(' ', '-').lower()
            
            # Handle special abbreviations
            team1 = team1.replace('a&m', 'am')
            team2 = team2.replace('a&m', 'am')
            team1 = team1.replace('&', 'and') #have not run into this yet
            team2 = team2.replace('&', 'and')
            # team1 = team1.replace('(', '')
            # team1 = team1.replace(')', '')
            # team1 = team2.replace('(', '')
            # team2 = team2.replace(')', '')
            
            #common abbreviations
            abbr_dict = {
                'ucf': 'central-florida',
                'uab': 'alabama-birmingham',
                'utsa': 'texas-san-antonio',
                'utep': 'texas-el-paso',
                'ul': 'louisiana',
                'lsu': 'louisiana-state',
                'lsu': 'louisiana-state',
                'nc': 'north-carolina',
                'tcu': 'texas-christian',
                'usc': 'southern-california',
                'ole-miss': 'mississippi',
                "hawai'i":"hawaii",
                "miami-(oh)":"miami-oh",
                "miami":"miami-fl",
                "app-state":"appalachian-state",
                "uconn":"connecticut",
                "byu":'brigham-young',
                "southern-miss":"southern-mississippi",
                "vmi":"virginia-military-institute",
                "unlv":"nevada-las-vegas",
                "smu":"southern-methodist",
                "bowling-green":"bowling-green-state",
                "san-josé-state": "san-jose-state",
                "louisiana": "louisiana-lafayette"
            }
            
            for abbr, full in abbr_dict.items():
                team1 = re.sub(rf'\b{abbr}\b', full, team1)
                team2 = re.sub(rf'\b{abbr}\b', full, team2)
                # team1 = team1.replace(abbr, full)
                # team2 = team2.replace(abbr, full)
            
            formatted_line = f"{team1},{team2}"
            formatted_lines.append(formatted_line)

    formatted_output = '\n'.join(formatted_lines)
    with open(file_name, 'w') as file:
        file.write(formatted_output)

format_team_names()