# College Football Game Predictions

machine learning that predicts the outcome of any Division I college football game. Data are from 2015 - 2023 seasons.
My DNN has an accuracy of 84% on the validation data. I use multi-class learning to use the prior week's feature data
to predict next week's feature data. I used multiple models to achieve this. Data are from [SportsReference](https://www.sports-reference.com/cfb/) and 
[CFBD](https://collegefootballdata.com/). The .pkl files I could not upload to GitHub due to size issues.

THE BIG PROBLEM:
-this is trained on every college football teams. The models have bias from the bad teams, thus every top 25 team
it assums that they are just going to win every game, as their feature values are substantially higher than the bad teams.
Therefore, when a bad team wins with "low" feature scores, and a top 25 team wins with "high" feature scores, the model assumes 
then that if the good team for a week plays bad, they will still win.

## Usage

```bash
python3 collect_augment_data.py #update the data for 2023 every week
python3 deep_learning_multiclass.py test #evaluate the model on the "test" data, which is the top 25 teams last week's outcomes
python3 deep_learning_multiclass.py notest #Predict the outcomes between two teams
```
### Current prediction accuracies
```bash
# Classification accuracy on predicting last week's outcomes for each model. I used the feature learning approach and a rolling average of 2
# Current model validation loss and validation:
number of features: 40
number of samples: 6053
Validation Loss: 0.07
Validation Accuracy: 0.98
=======================================
Rolling median 2 Accuracy out of 38 teams: 0.8947368421052632
Rolling median 3 Accuracy out of 38 teams: 0.868421052631579
Rolling EWM 2 Accuracy out of 38 teams: 0.8947368421052632
=======================================
```
### Outputs
example out when you input two teams
```bash
==============================
Win Probabilities from DNN feature predictions
syracuse : 95.01610398292542 % florida-state : 4.983900114893913 %
Win Probabilities from LinRegress feature predictions
syracuse : 10.211130976676941 % florida-state : 89.7888720035553 %
Win Probabilities from rolling median predictions
syracuse : 0.19387530628591776 % florida-state : 99.80612397193909 %
==============================
```

### My Simple Rating System
I created a simple rating system that uses the median point differential of one team and subtracts that value
by all median point differential of all other teams they played against. The only caveat is that I only count 
the point differentials of the other teams when they lose, as I want to be able to assess how much each team is 
losing by. The theory behind this is that good teams will not lose often and when they do it will most likely be 
by small margins. The higher the srs value, the "better" the team is.
![](https://github.com/bszek213/deepCFB/blob/main/my_srs.png) 

### Creating CFBD api key
got to [CFBD](https://collegefootballdata.com/key) and enter your email address. they will send you an API key. 
Create a file called `api_key.yaml` and store the API key like this:
```bash
api_key:
  Authorization: asdifnasdofnasdpvnapionmfaspidfasodnfkajdslmalskdm
```
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
