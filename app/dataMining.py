import pandas as pd

import preProcessor
import model01 # Logistic Regression
import model02 # KNN
import model03 # Gaussian Naive Bayes Classifier
import model04 # Support Vector Machine
import model05 # Random Forest
import predictionAnalyze

print(">>> Process Start!!")

# Pre processing train data
print(">>> Pre processing train data ...")
df = pd.read_csv("train.csv", sep=";")
dfTrain = preProcessor.preProcess(df)

# Set x and y variable to use in the models
x = ["VinculoAtivo", "Ind Rais Negativa", "Ind Simples"]
y = "Ind Atividade Ano"

# Training and Testing the models to Choose best one
print(">>> Training and Testing the models ...")


print(">>> Model01 - Logistic Regression")
dfPrModel01 = model01.trainTest(dfTrain, x, y)
score01 = predictionAnalyze.analyze(dfPrModel01)

print(">>> Model02 - KNN")
dfPrModel02 = model02.trainTest(dfTrain, x, y)
score02 = predictionAnalyze.analyze(dfPrModel02)

print(">>> Model03 - Gaussian Naive Bayes")
dfPrModel03 = model03.trainTest(dfTrain, x, y)
score03 = predictionAnalyze.analyze(dfPrModel03)

print(">>> Model04 - Decision Tree")
dfPrModel05 = model05.trainTest(dfTrain, x, y)
score05 = predictionAnalyze.analyze(dfPrModel05)

print(">>> Choose the best one ...")

## Real Training and Predicting output

# Pre processing the dataset to predict
print(">>> Pre processing test data ...")

df = pd.read_csv("test.csv", sep=";")
dfTest = preProcessor.preProcess(df)


print(">>> Predicting ...")

if score01 > score02 and score01 > score03 and score01 > score05:
    print(">>> Logistic Regression ...")
    model01.trainPredict(dfTrain, dfTest, x, y)

if score02 > score01 and score02 > score03 and score02 > score05:
    print(">>> KNN ...")
    model02.trainPredict(dfTrain, dfTest, x, y)

if score03 > score01 and score03 > score02 and score03 > score05:
    print(">>> Gaussian Naive Bayes ...")
    model03.trainPredict(dfTrain, dfTest, x, y)

if score05 > score01 and score05 > score02 and score05 > score03:
    print(">>> Decision Tree ...")
    model05.trainPredict(dfTrain, dfTest, x, y)

print(">>> Process complete!! Please check prediction.csv.")
