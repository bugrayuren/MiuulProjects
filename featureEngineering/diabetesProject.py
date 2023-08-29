import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#Veri temizliği için gerekli adımlar sırasıyla
# Outliers
# Missing Values
# Encoding

df = pd.read_csv("diabetes.csv")
df.info()


#################### OUTLIERS ########################
def ColumnClassifier(df):

    numericalColumns = [colName for colName in df.columns if df[colName].dtypes in ["int64","int32","float64","float32"] and df[colName].nunique() > 30]
    categoricalColumns = [colName for colName in df.columns if df[colName].dtypes in ["O"] and df[colName].nunique() < 10]
    catButCardinal = [colName for colName in df.columns if df[colName].dtypes in ["O"] and df[colName].nunique() >= 10]
    numButCat = [colName for colName in df.columns if df[colName].dtypes in ["int64","int32"] and df[colName].nunique() < 30]
    numofClassified= len(numericalColumns) + len(categoricalColumns) + len(catButCardinal) + len(numButCat)



    print(f"\nI'M REPORTING SIR! \n"
          f"*********\n"
          f"The number of Numerical Columns: {len(categoricalColumns)}\n"
          f"The number of Numerical Columns: {len(numericalColumns)}\nThe number of Cardinal Columns: {len(catButCardinal)}\n"
          f"The number of numButCat Columns:{len(numButCat)}\n"
          f"SUM OF THESE COLUMNS IS {numofClassified} SIR\n"
          f"NUMBER OF THE COLUMNS IN DATASET IS {df.shape[1]}\n"
          f"We converted numButCat to Categorical Columns SIR!\n")

    categoricalColumns.append(numButCat)

    if (numofClassified == df.shape[1]):
        print("***\nOperation HAMMEROFTHEMADGOD flawlessly accomplished SIR!\n***")
    else:
        print("***\nOperation failed.\n***")


    return numericalColumns, categoricalColumns, catButCardinal


numCol, catCol, carCol = ColumnClassifier(df)
# OUTPUT:
# The number of Numerical Columns: 0
# The number of Numerical Columns: 7
# The number of Cardinal Columns: 0
# The number of numButCat Columns:2
# SUM OF THESE COLUMNS IS 9 SIR
# NUMBER OF THE COLUMNS IN DATASET IS 9
# We converted numButCat to Categorical Columns SIR!
# ***
# Operation HAMMEROFTHEMADGOD flawlessly accomplished SIR!
# ***


def TresholdSelector(df, numCol, up=0.75, low=0.25, iqrCoef = 1.5):
    upQuartile = df[numCol].quantile(up)
    lowQuartile = df[numCol].quantile(low)
    iqr = (upQuartile - lowQuartile)
    upLimit = upQuartile + (iqrCoef*iqr)
    lowLimit = lowQuartile - (iqrCoef*iqr)
    return lowLimit, upLimit

low, up = TresholdSelector(df, "BloodPressure")


def OutlierCather(df , numCol, index = True):
    low, up = TresholdSelector(df, numCol)
    numOutlier = df[((df[numCol]<low) | (df[numCol]>up))].shape[0]
    print(f"\n Number of outliers in {numCol} is {numOutlier} SIR!\n")
    if index:
        return df[((df[numCol]<low) | (df[numCol]>up))].index

indexOutliers = OutlierCather(df, "BloodPressure")


def OutlierRatio(df):
