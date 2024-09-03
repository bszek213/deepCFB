# College Football Game Predictions

machine learning that predicts the outcome of any Division I college football game. Data are from 2015 - 2024 seasons.
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
python3 collect_augment_data.py #update the data for 2024 every week
python3 deep_learning_multiclass.py test #evaluate the model on the "test" data, which is the top 25 teams last week's outcomes
python3 deep_learning_multiclass.py notest #Predict the outcomes between two teams
```
### Current prediction accuracies
![](https://github.com/bszek213/deepCFB/blob/main/Training.png) 

### Outputs
example out from the `results.txt`
```bash
==============================
Win Probabilities from Monte Carlo Simulation with 10000 simulations
louisiana-lafayette : 99.999 %buffalo : 0.001 %
Win Probabilities from rolling median of 2 predictions
louisiana-lafayette : 100.0 %buffalo : 0.0 %
Win Probabilities from rolling median of 3 predictions
louisiana-lafayette : 99.99 %buffalo : 0.01 %
Win Probabilities from exponential weighted average of 2 predictions
louisiana-lafayette : 100.0 %buffalo : 0.0 %
Win Probabilities from 25th and 75th percentile rolling 2
25th: louisiana-lafayette : 100.0 %buffalo : 0.0 %
75th: louisiana-lafayette : 99.999 %buffalo : 0.001 %
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
