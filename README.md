# College Football Game Predictions

machine learning that predicts the outcome of any Division I college football game. Data are from 2015 - 2023 seasons.
My DNN has an accuracy of 84% on the validation data. I use multi-class learning to use the prior week's feature data
to predict next week's feature data. I used multiple models to achieve this. Data are from Sports-reference.com (https://www.sports-reference.com/cfb/) and 
cfbd (https://collegefootballdata.com/). The .pkl files I could not update to GitHub due to size issues.

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
Ensemble Accuracy out of 23 teams: 0.9565217391304348
DNN Accuracy out of 23 teams: 0.9565217391304348
LinRegress Accuracy out of 23 teams: 0.9130434782608695
RandomForest Accuracy out of 23 teams: 0.9565217391304348
Rolling median 2 Accuracy out of 23 teams: 0.8695652173913043
MLP Accuracy out of 23 teams: 0.9565217391304348
XGB Accuracy out of 23 teams: 0.9565217391304348
```
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
