# ml-perf-metric
This repo contains a python script to calculate standard performance metrics used in machine learning applications.

The file **myfunctions.py** contains **perfMetrics(pred,Y)** which takes the predictions (pred) and true labels (Y) to calculate:

- AUC ROC (Area Under the Receiver Operating Curve).
- AUC PRC (Area Under the Precision Recall Curve).
- Acuracy (normalized and not).
- F1_score (binary, micro, and macro) : The harmonic mean of Precision and Recall.
- TPR_at_FPR0p0 : True Positive Rate at 0% False Positive Rate.
- TPR_at_FPR0p1 : True Positive Rate at 1% False Positive Rate.
- TPR_at_FPR5 : True Positive Rate at 5% False Positive Rate.
- Brier Score: mean squared error.
- HLtest (Hosmer-Lemeshow Test) : evaluates goodness-of-fit of model, i.e. if the model predicts 10% probability of a positive class for an observation, is the model correctly predicting the positive class 10% of the time?
