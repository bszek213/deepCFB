# College Football Game Predictions

machine learning that predicts the outcome of any Division I college football game. Data are from 2015 - 2023 seasons.
My DNN has an accuracy of 84% on the validation data. I use multi-class learning to use the prior week's feature data
to predict next week's feature data. I used multiple models to achieve this. Data are from Sports-reference.com [SportsRef](https://www.sports-reference.com/cfb/) and 
[CFBD](https://collegefootballdata.com/). The .pkl files I could not update to GitHub due to size issues.

THE BIG PROBLEM:
-this is trained on every college football teams. The models have bias from the bad teams, thus every top 25 team
it assums that they are just going to win every game, as their feature values are substantially higher than the bad teams.
Therefore, when a bad team wins with "low" feature scores, and a top 25 team wins with "high" feature scores, the model assumes 
then that if the good team for a week plays bad, they will still win.

## Usage

```bash
python3 collect_data.py #update the data for 2023 every week
python3 deep_learn.py test #evaluate the model on the "test" data, which is the top 25 teams last week's outcomes
python3 deep_learn.py notest #Predict the outcomes between two teams
```
### Current prediction accuracies
```bash
# Classification accuracy on predicting last week's outcomes for each model. I used the feature learning approach and a rolling average of 2
=======================================
Ensemble Accuracy out of 25 teams: 0.92
DNN Accuracy out of 25 teams: 0.92
LinRegress Accuracy out of 25 teams: 0.92
RandomForest Accuracy out of 25 teams: 0.92
Rolling median 2 Accuracy out of 25 teams: 0.92
MLP Accuracy out of 25 teams: 0.8
XGB Accuracy out of 25 teams: 0.92
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
