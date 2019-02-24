# Decision Tree

import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split


def trainTest(dftt, x, y):
    # dftt - DF Train and Test
    ## SPlit Dataset to train and test
    x_train, x_test, y_train, y_test = train_test_split(dftt[x], dftt[y], test_size=0.20, random_state=101)
    
    model = DecisionTreeClassifier()
    
    model.fit(x_train, y_train)
    
    prediction = model.predict(x_test)
   # predict_prob = model.predict_proba(x_test)
    
    result = pd.DataFrame(columns=['Test', 'Prediction', 'Prob'])
    result['Test'] = y_test
    result['Prediction'] = prediction
    result['Prob'] = 0
    
    return result
    
def trainPredict(dft, dfp, x, y):
    # dft - DF Train,
    # dfp - DF Pprediction
    model = DecisionTreeClassifier()
    
    model.fit(dft[x], dft[y])
    
    prediction = model.predict(dfp[x])
    
    result = pd.DataFrame(columns=['Id', 'Prediction', 'Prob'])
    result['Id'] = dfp['id']
    result['Prediction'] = prediction
    result['Prob'] = 0

    result.to_csv('prediction.csv')
    