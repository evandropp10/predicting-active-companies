import pandas as pd

# Function to change "{ñ class}" to 0
def cleanNclass(var):
    var = var.strip()
    if var.find("{ñ clas") != -1:
        return "0"
    else:
        return var

def applyVinculo(var1):
    if var1 > 0:
        return 1
    else:
        return 0


# Main Function
def preProcess(df):

    df["BairrosSP"] = df["Bairros SP"].apply(cleanNclass)
    df["BairrosFortaleza"] = df["Bairros Fortaleza"].apply(cleanNclass)
    df["BairrosRJ"] = df["Bairros RJ"].apply(cleanNclass)
    df["DistritosSP"] = df["Distritos SP"].apply(cleanNclass)
    
    df.drop(["Bairros SP", "Bairros Fortaleza", "Bairros RJ", "Distritos SP"], axis=1, inplace=True)

    df["QtdVinculosAtivos"] = df["Qtd Vínculos Ativos"] + df["Qtd Vínculos CLT"] + df["Qtd Vínculos Estatutários"] + df["Ind CEI Vinculado"]

    df["VinculoAtivo"] = df["QtdVinculosAtivos"].apply(applyVinculo)
    
    return(df)