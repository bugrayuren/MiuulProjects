import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor

matplotlib.use('Qt5Agg')
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 500)
df = pd.read_csv("machineLearning/hitters.csv")


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


df.info()

df.isnull().sum()


# There are missing values in salary. Since salary is our target variable, they should be
# deleted

def ColumnClassifier(df, report=True):
    numericalColumns = [colName for colName in df.columns if
                        df[colName].dtypes in ["int64", "int32", "float64", "float32"] and df[colName].nunique() > 15]
    categoricalColumns = [colName for colName in df.columns if
                          df[colName].dtypes in ["O"] and df[colName].nunique() < 10]
    catButCardinal = [colName for colName in df.columns if df[colName].dtypes in ["O"] and df[colName].nunique() >= 10]
    numButCat = [colName for colName in df.columns if
                 df[colName].dtypes in ["int64", "int32"] and df[colName].nunique() <= 15]
    numofClassified = len(numericalColumns) + len(categoricalColumns) + len(catButCardinal) + len(numButCat)

    if report:
        print(f"\nI'M REPORTING SIR! \n"
              f"*********\n"
              f"The number of Categorical Columns: {len(categoricalColumns)}\n"
              f"The number of Numerical Columns: {len(numericalColumns)}\nThe number of Cardinal Columns: {len(catButCardinal)}\n"
              f"The number of numButCat Columns:{len(numButCat)}\n"
              f"SUM OF THESE COLUMNS IS {numofClassified} SIR\n"
              f"NUMBER OF THE COLUMNS IN DATASET IS {df.shape[1]}\n"
              f"We converted numButCat to Categorical Columns SIR!\n")

    categoricalColumns = categoricalColumns + numButCat
    if report:
        if (numofClassified == df.shape[1]):
            print("***\nOperation HAMMEROFTHEMADGOD flawlessly accomplished SIR!\n***")
        else:
            print("***\nOperation failed.\n***")

    return numericalColumns, categoricalColumns, catButCardinal


numCols, catCols, carCols = ColumnClassifier(df)


# The number of Categorical Columns: 3
# The number of Numerical Columns: 15
# The number of Cardinal Columns: 0
# The number of numButCat Columns:2

def cat_target_analysis(df, catCols, target):
    for col in catCols:
        print(df.groupby(col)[target].apply("mean"), end="\n\n\n")


cat_target_analysis(df, catCols, "Salary")


# Almost no difference in League and NewLeague. Remarkable difference in Division
# League
# A    541.999547
# N    529.117500
# Name: Salary, dtype: float64
# Division
# E    624.271364
# W    450.876873
# Name: Salary, dtype: float64
# NewLeague
# A    537.113028
# N    534.553852
# Name: Salary, dtype: float64

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


num_summary(df, numCols, plot=True)

sns.histplot(df, x="Salary", hue="Division")

corr = df[numCols].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr, cmap="RdBu", annot=True)
plt.show()


# There is a problematic correlation between "C_" variables and year. They have to be converted to 'yearly_avarage'.
# atBar Hits is also problematic (Runs)
# Catbar and Chits is more problematic

# finding carrer-long columns

# c_columns = [col for col in df.columns if col.startswith("c") or col.startswith("C")]

# Creating Avarage Columns


# for col in c_columns:
#     df["AVG_" + col[1:]] = df[col] / df["Years"]
#
# df.head()
#
#
# corr = df.corr()
# plt.figure(figsize=(10, 6))
# sns.heatmap(corr, cmap="RdBu",annot=True)
# plt.show()

############ Missing Values ##################


def TresholdSelector(df, numCol, up=0.75, low=0.25, iqrCoef=1.5):
    upQuartile = df[numCol].quantile(up)
    lowQuartile = df[numCol].quantile(low)
    iqr = (upQuartile - lowQuartile)
    upLimit = upQuartile + (iqrCoef * iqr)
    lowLimit = lowQuartile - (iqrCoef * iqr)
    return lowLimit, upLimit


def OutlierCather(df, numCol, index=True, report=True):
    low, up = TresholdSelector(df, numCol)
    numOutlier = df[((df[numCol] < low) | (df[numCol] > up))].shape[0]
    if report:
        print(f"\n Number of outliers in {numCol} is {numOutlier} SIR!\n")
    if index:
        return df[((df[numCol] < low) | (df[numCol] > up))].index


def OutlierRatio(df):
    all_outlier_indices = []
    num_outlier_by_column = {}
    numericalColumns, categoricalColumns, catButCardinal = ColumnClassifier(df, report=False)
    for col in numericalColumns:
        outlier_indices = list(OutlierCather(df, col, report=False))
        all_outlier_indices += outlier_indices
        num_outlier_by_column[col] = len(outlier_indices)
    outliers_array = np.array(all_outlier_indices)
    ratio = len(np.unique(outliers_array)) / df.shape[0] * 100
    print(f"\n"
          f"Outliers ratio to whole dataset: {round(ratio, 2)}"
          f"\n")
    if (ratio > 5):
        print("LARGE AMOUNT OF OUTLIER. REMOVAL IS NOT RECOMENDED")
    return num_outlier_by_column

# Outliers proportion to whole dataset: 33.54

# I do'nt want to lose that amount of large data.
# LOF Analysis may reduce the amount of data loss

# LOF Outlier Analysis
# LOF requires integer/float and non-missing data.

#So frame our data to meet these requirements
#First drop categorical columns:
lof_data = df.copy()
for catCol in catCols:
    lof_data = lof_data.drop(catCol, axis = 1)

lof_data

# then drop Salary since it contains null values.

lof_data.drop("Salary", axis=1, inplace= True)

lof_data.isnull().sum()

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(lof_data)

lof_scores = clf.negative_outlier_factor_
lof_scores[0:5]
# df_scores = -df_scores
np.sort(lof_scores)[0:5]

scores = pd.DataFrame(np.sort(lof_scores))
scores.plot(stacked=True, xlim=[0, 100], style='.-')
plt.show()
# [0:12]

th = np.sort(lof_scores)[12]

df[lof_scores < th]

df[df_scores < th].shape


df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

df[df_scores < th].index

df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)


