import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

#The following argument is placed to fix "not responding" problem caused by mathplotlib or my IDE.
#It works for me but I'm not aware of the details.
matplotlib.use('Qt5Agg')

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

df.describe().T


def ColumnClassifier(df, report = True):

    numericalColumns = [colName for colName in df.columns if df[colName].dtypes in ["int64","int32","float64","float32"] and df[colName].nunique() > 30]
    categoricalColumns = [colName for colName in df.columns if df[colName].dtypes in ["O"] and df[colName].nunique() < 10]
    catButCardinal = [colName for colName in df.columns if df[colName].dtypes in ["O"] and df[colName].nunique() >= 10]
    numButCat = [colName for colName in df.columns if df[colName].dtypes in ["int64","int32"] and df[colName].nunique() < 30]
    numofClassified= len(numericalColumns) + len(categoricalColumns) + len(catButCardinal) + len(numButCat)


    if report:
        print(f"\nI'M REPORTING SIR! \n"
              f"*********\n"
              f"The number of Categorical Columns: {len(categoricalColumns)}\n"
              f"The number of Numerical Columns: {len(numericalColumns)}\nThe number of Cardinal Columns: {len(catButCardinal)}\n"
              f"The number of numButCat Columns:{len(numButCat)}\n"
              f"SUM OF THESE COLUMNS IS {numofClassified} SIR\n"
              f"NUMBER OF THE COLUMNS IN DATASET IS {df.shape[1]}\n"
              f"We converted numButCat to Categorical Columns SIR!\n")

    categoricalColumns= categoricalColumns + numButCat
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

#df["TotalCharges"] = df["TotalCharges"].astype("float64")
# ValueError: could not convert string to float: ' '
# This means we have hidden missing values

df.loc[df["TotalCharges"]==' ',"TotalCharges"] = np.nan
df["TotalCharges"] = df["TotalCharges"].astype("float64")


#Önceki değişiklikler sebebiyle tekrar kategorize edelim
numCols, catCols, catButCarCols = ColumnClassifier(df)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


#### Adım 3 Hedef Değişken analizi ###

def cat_target_analysis(df, catCols, target):
    for col in catCols:
        print(df.groupby(col)[target].apply("mean"), end="\n\n\n")


df["Churn"].value_counts()
#Includes "Yes" "and "No"

#The churn should be turned to integer to evaluate in terms of mean.
df["Churn"] = df["Churn"].apply(lambda x: 1 if (x=="Yes") else 0)
cat_target_analysis(df, catCols, "Churn")

# Tech Support (no higher),  Device Protection (related to tech support), OnlineBackup, Online Security
# grubu tamamen benzer oranlar gösteriyor ve ınternet Service ile yakından alakalı. Correlation map sırasında
# bunları çıkarmak iyi olabilir.
# Etkisi yokmuş gibi gözükenleri yazmak daha kolay:
# Gender, Phone Service, Multiple Lines

def num_target_analysis(df, numCols, target):
    for col in numCols:
        print(df.groupby(target)[col].apply("mean"),end="\n\n\n")


num_target_analysis(df, numCols, "Churn")

#Etkisi olmayabilecek olanlar: Monthly Charges şüpheli.


########## Adım 5 Outliers #############
def TresholdSelector(df, numCol, up=0.75, low=0.25, iqrCoef = 1.5):
    upQuartile = df[numCol].quantile(up)
    lowQuartile = df[numCol].quantile(low)
    iqr = (upQuartile - lowQuartile)
    upLimit = upQuartile + (iqrCoef*iqr)
    lowLimit = lowQuartile - (iqrCoef*iqr)
    return lowLimit, upLimit


def OutlierCather(df , numCol, index = True, report = True):
    low, up = TresholdSelector(df, numCol)
    numOutlier = df[((df[numCol]<low) | (df[numCol]>up))].shape[0]
    if report:
        print(f"\n Number of outliers in {numCol} is {numOutlier} SIR!\n")
    if index:
        return df[((df[numCol]<low) | (df[numCol]>up))].index

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


results = OutlierRatio(df)
#no outliers

####### Missing values #####
df.isnull().sum().sum()/df.shape[0]

#11 missing values. only .0001 of the data. Can be discarded without hesitation.

df.dropna(inplace=True)

df.isnull().sum().sum()
# 0

#########Adım 7 Correlation Analysis ###########

corr = df[numCols].corr()
sns.heatmap(corr, cmap="RdBu")
plt.show()

########### Cross-Tab ##############
cross_tab = pd.crosstab(df)
print(cross_tab)
catCols
# Tech Support (no higher),  Device Protection (related to tech support), OnlineBackup, Online Security
# grubu tamamen benzer oranlar gösteriyor ve ınternet Service ile yakından alakalı. Göz atılmalı.

###### Feature Engineering #####

df.head()
df["customerID_0"] = df["customerID"].apply(lambda x: x.split("-")[0])
df["customerID_1"] = df["customerID"].apply(lambda x: x.split("-")[1])
df["customerID_1"].shape[0]-df["customerID_1"].nunique()
# Harfli kısımda sadece 3 tekrar eden değer var. Buradan bir şey çıkmayabilir
df["customerFirstLetter"] = df["customerID_1"].apply(lambda x: x[0])
df["customerFirstLetter"].value_counts()
df.groupby("customerFirstLetter")["Churn"].apply("mean").sort_values()
# Looks totally random.
df.drop(["customerID_0","customerID_1", "customerFirstLetter"], axis = 1, inplace=True)

#Herhangi bir internet bazlı teknik hizmet alıp almaamış olmasına göre bir kategori oluşturulabilir

internetRelatedCats = ["OnlineSecurity", "OnlineBackup", "DeviceProtection","TechSupport", "StreamingTV" , "StreamingMovies"]
df["is_ExtraInternetService"] = df.loc[:,internetRelatedCats].apply(lambda row: 'Yes' if 'Yes' in row.values else ("No internet service" if "No internet service" in row.values else "No"), axis=1)
df["is_ExtraInternetService"].value_counts()
df.groupby("is_ExtraInternetService")["Churn"].agg("mean")

streamingCols = ["StreamingTV" , "StreamingMovies"]
df["anyStreaming"] = df.loc[:,streamingCols].apply(lambda row: 'Yes' if 'Yes' in row.values else ("No internet service" if "No internet service" in row.values else "No"), axis=1)
df["anyStreaming"].value_counts()
df.groupby("anyStreaming")["Churn"].agg("mean")


numCols, catCols, catButCarCols = ColumnClassifier(df)

df.drop("customerID", axis=1, inplace= True)

# 3 kategorili column sayımız çok fazla ve bunun kattığı değer o kadar da yüksek değil. 20 tane column ile one hot encoder
# uygularsak en az 40 column olacak. No internet service içeren data pointlerin sayısı çok fazla ve churn oranları
# Evet ve Hayır seçeneklerine göre çok düşük. Eğer bunu hayır ile birleştirirsem hayır'ın oranında büyük bir düşüş gözlenecek
# Ama mantıken gerçekten de "Hayır" durumundalar. Sonuçta bu servisten faydalanmıyorlar. Haliyle deneysel yaklaşıp önce hepsini hayır'a
# çevirip sonucu göreceğim
ddf = df.copy()
for col in catCols:
    ddf[col] = ddf[col].apply(lambda x: "No" if (x == "No internet service") else x)

ddf["anyStreaming"].value_counts()
ddf.groupby("anyStreaming")["Churn"].agg("mean")
# after merge
# anyStreaming
# No     0.228442
# Yes    0.303577
# Name: Churn, dtype: float64

#before merge
# anyStreaming
# No                     0.344571
# No internet service    0.074342
# Yes                    0.303577
# Name: Churn, dtype: float64

#data completely changed.





