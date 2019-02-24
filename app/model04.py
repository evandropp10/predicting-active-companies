# Support Vector Machine

import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


def trainTest(dftt):
    # dftt - DF Train and Test
    ## SPlit Dataset to train and test
    x_train, x_test, y_train, y_test = train_test_split(dftt[["Qtd Vínculos CLT", "Qtd Vínculos Ativos", "Qtd Vínculos Estatutários", "Ind Rais Negativa"]], 
                                                        dftt["Ind Atividade Ano"], test_size=0.20)
    
    
    param_grid = {'Ind Atividade Ano': [0,1]} 
    model = GridSearchCV(SVC(), param_grid)
    
    model.fit(x_train, y_train)
    
    prediction = model.predict(x_test)
    predict_prob = model.predict_proba(x_test)
    
    result = pd.DataFrame(columns=['Test', 'Prediction', 'Prob'])
    result['Test'] = y_test
    result['Prediction'] = prediction
    result['Prob'] = predict_prob[:,1]
    
    return result
    
def trainPredict(dft, dfp):
    # dft - DF Train,
    # dfp - DF Pprediction
    param_grid = {'Ind Atividade Ano': [0, 1]}
    model = GridSearchCV(SVC(), param_grid)
    
    model.fit(dft[["Qtd Vínculos CLT", "Qtd Vínculos Ativos", "Qtd Vínculos Estatutários", "Ind Rais Negativa"]], 
                dft["Ind Atividade Ano"])
    
    prediction = model.predict(dfp[["Qtd Vínculos CLT", "Qtd Vínculos Ativos", "Qtd Vínculos Estatutários", "Ind Rais Negativa"]])
    
    result = pd.DataFrame(columns=['Test', 'Prediction'])
    result['Test'] = dfp['Ind Atividade Ano']
    result['Prediction'] = prediction
    
    result.to_csv("prediction.csv")
    