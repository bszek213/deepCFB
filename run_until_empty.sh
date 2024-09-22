#!/bin/bash
teams_file="teams_played_this_week.txt"

#go unitl teams.txt file has no more lines
while [ -s "$teams_file" ]; do    
    python3 deep_learning_multiclass.py train
    
    #check if the teams file is now empty
    if [ ! -s "$teams_file" ]; then
        echo "No more teams left to process."
        break
    fi
    sleep 1
done

echo "All teams processed"