import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from sklearn.neighbors import LocalOutlierFactor

#The following argument is placed to fix "not responding" problem caused by mathplotlib or my IDE.
#It works for me but I'm not aware of the details.
matplotlib.use('Qt5Agg')


#Veri temizliği için gerekli adımlar sırasıyla
# Outliers
# 1-) Remove
# 2-) Re-assignment with tresholds
# Local Outlier analysis?
# Missing Values
# Encoding
pd.set_option('display.width', 100)
pd.set_option('display.max_columns', 500)
df = pd.read_csv("diabetes.csv")
df.info()
#  #   Column                    Non-Null Count  Dtype
# ---  ------                    --------------  -----
#  0   Pregnancies               768 non-null    int64
#  1   Glucose                   768 non-null    int64
#  2   BloodPressure             768 non-null    int64
#  3   SkinThickness             768 non-null    int64
#  4   Insulin                   768 non-null    int64
#  5   BMI                       768 non-null    float64
#  6   DiabetesPedigreeFunction  768 non-null    float64
#  7   Age                       768 non-null    int64
#  8   Outcome                   768 non-null    int64

df.describe()

df["BloodPressure"].hist()
plt.show()

# I've to confess that I wasn't even inspired by the course I take for following function
# as I do in other functions in this project. Differing from them I totally copied
# it from Miuul's documents.
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

##################### NA #####################
# Pregnancies                 0
# Glucose                     0
# BloodPressure               0
# SkinThickness               0
# Insulin                     0
# BMI                         0
# DiabetesPedigreeFunction    0
# Age                         0
# Outcome                     0
# dtype: int64

#By nature, these variables except for 'Pregnancies'
# and 'Outcome' (which is out target boolean variable)
# cannot be 0. As mentioned in metadata, these values
# should be assumed NA values

# An adhoc solution for this problem
target_columns = [col for col in df.columns if col not in ["Pregnancies", "Outcome"]]

# ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
for col in target_columns:
    df[col] = df[col].apply(lambda x: np.nan if (x == 0) else x)

check_df(df)
# New number of null values make more sense
# ##################### NA #####################
# Pregnancies                   0
# Glucose                       5
# BloodPressure                35
# SkinThickness               227
# Insulin                     374
# BMI                          11
# DiabetesPedigreeFunction      0
# Age                           0
# Outcome                       0
# dtype: int64



########################################
##############  OUTLIERS  ##############
########################################
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


def OutlierCather(df , numCol, index = True, report = True):
    low, up = TresholdSelector(df, numCol)
    numOutlier = df[((df[numCol]<low) | (df[numCol]>up))].shape[0]
    if report:
        print(f"\n Number of outliers in {numCol} is {numOutlier} SIR!\n")
    if index:
        return df[((df[numCol]<low) | (df[numCol]>up))].index

indexOutliers = OutlierCather(df, "BloodPressure")


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
          f"Outliers ratio to whole dataset: {round(ratio,2)}"
          f"\n")
    if (ratio > 5):
        print("LARGE AMOUNT OF OUTLIER. REMOVAL IS NOT RECOMENDED")
    return num_outlier_by_column
#Önemli. Çıktı şu anda dict halinde. Pandas Dataframe haline getirilirse daha iyi olur.
out_dict = OutlierRatio(df)
df.shape[0]


def removeOutliers(df, numColName):
    indices = OutlierCather(df, numColName, report=False)
    return df.drop(indices)
#Since number of outliers in Blood Pressure is 45, difference between number of rows original df and 'without outlier df' must be 45
without_outlier = removeOutliers(df, "BloodPressure")
df.shape[0] - without_outlier.shape[0]

def replace_with_tresholds(df, numColName, inplace = False):
    low, up = TresholdSelector(df, numColName)
    if inplace:
        df.loc[df[numColName] < low, numColName] = low
        df.loc[df[numColName] > up, numColName] = up

    else:
        copy_df = df.copy()
        copy_df.loc[copy_df[numColName] < low, numColName] = low
        copy_df.loc[copy_df[numColName] > up, numColName] = up
        return copy_df

new_df = replace_with_tresholds(df, "BloodPressure")
OutlierCather(new_df, numCol="BloodPressure", index=False)
# Number of outliers 0. Test has been passed.

#Lets try a more complex outlier detection technique to see whether there is a significant difference.
#Local Outlier Factor

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)
df_scores = clf.negative_outlier_factor_
scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')
plt.show()
th = np.sort(df_scores)[10]
lof_implemented_df =df[df_scores<th]
lof_implemented_df.shape[0]
#It has found just 10 outliers

# 10/df.shape[0] is 0.13 so it would be a clever decision to take care of the outliers only found in this method
# since other method directs me to remove or change at least 16% of the whole data points

df = df[df_scores >= th]

df.shape[0]


########################################
###########  MISSING VALUES  ###########
########################################


def missing_values_report(df):
    col_with_missing_values = [col for col in df.columns if df[col].isnull().sum() > 0]

df.isnull().sum()
#there is no missing values.