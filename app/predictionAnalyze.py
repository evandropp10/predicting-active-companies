import pandas as pd
from sklearn import metrics

def analyze(df):

    score = metrics.accuracy_score(df['Test'], df['Prediction'])

    confMatrix = metrics.confusion_matrix(df['Test'], df['Prediction'])

    # Verifying if predicts one class, in this case points = 0

    if confMatrix[0][0] == 0 or confMatrix[1][1] == 0:
        score = 0

    return score