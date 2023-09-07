import pandas as pd
import numpy as np
import seaborn as sns

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 500)
df = pd.read_csv("Telco-Customer-Churn.csv")

def check_df(df, head=5):
    print("##################### Shape #####################")
    print(df.shape)
    print("##################### Types #####################")
    print(df.dtypes)
    print("##################### Head #####################")
    print(df.head(head))
    print("##################### Tail #####################")
    print(df.tail(head))
    print("##################### NA #####################")
    print(df.isnull().sum())
    print("##################### Quantiles #####################")
    print(df.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)



check_df(df)


def ColumnClassifier(df, report = True):

    numericalColumns = [colName for colName in df.columns if df[colName].dtypes in ["int64","int32","float64","float32"] and df[colName].nunique() > 30]
    categoricalColumns = [colName for colName in df.columns if df[colName].dtypes in ["O"] and df[colName].nunique() < 10]
    catButCardinal = [colName for colName in df.columns if df[colName].dtypes in ["O"] and df[colName].nunique() >= 10]
    numButCat = [colName for colName in df.columns if df[colName].dtypes in ["int64","int32"] and df[colName].nunique() < 30]
    numofClassified= len(numericalColumns) + len(categoricalColumns) + len(catButCardinal) + len(numButCat)


    if report:
        print(f"\nI'M REPORTING SIR! \n"
              f"*********\n"
              f"The number of Numerical Columns: {len(categoricalColumns)}\n"
              f"The number of Numerical Columns: {len(numericalColumns)}\nThe number of Cardinal Columns: {len(catButCardinal)}\n"
              f"The number of numButCat Columns:{len(numButCat)}\n"
              f"SUM OF THESE COLUMNS IS {numofClassified} SIR\n"
              f"NUMBER OF THE COLUMNS IN DATASET IS {df.shape[1]}\n"
              f"We converted numButCat to Categorical Columns SIR!\n")

    categoricalColumns.append(numButCat)
    if report:
        if (numofClassified == df.shape[1]):
            print("***\nOperation HAMMEROFTHEMADGOD flawlessly accomplished SIR!\n***")
        else:
            print("***\nOperation failed.\n***")


    return numericalColumns, categoricalColumns, catButCardinal


numCols, catCols, catButCarCols = ColumnClassifier(df)

#I've discovered that "TotalCharges" labaled as Categorical Cardinal however it should have been numerical.
df["TotalCharges"].nunique()
#Indeed it's cardinal but it should have been and integer or float.
df.info()

df["TotalCharges"].dtypes
# TotalCharges 7043 non - null object
# So it will be wise to check all the object types.

df["TotalCharges"] = df["TotalCharges"].astype("float64")
# ValueError: could not convert string to float: ' '
# This means we have hidden missing values

df.loc[df["TotalCharges"]==' ',"TotalCharges"] = np.nan
df["TotalCharges"] = df["TotalCharges"].astype("float64")