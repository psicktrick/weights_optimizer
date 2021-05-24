import pandas as pd
import numpy as np

from ga_weights_optimizer import WeightsOptimizer
from sklearn.linear_model import LinearRegression, LassoCV, Ridge, LassoLarsCV,ElasticNetCV
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from copy import  deepcopy


class CreateModels:
    def __init__(self):
        self.train = pd.read_csv(r"C:\Users\sickt\Downloads\data\train.csv")
        self.test = pd.read_csv(r"C:\Users\sickt\Downloads\data\test.csv")
        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocess_data()
        self.predictions = self.get_predictions()


    def get_predictions(self):
        lassocv = LassoCV(eps=1e-7)
        ridge = Ridge(alpha=1e-6)
        lassolarscv = LassoLarsCV()
        elasticnetcv = ElasticNetCV(eps=1e-15)

        lassocv.fit(self.X_train, self.y_train)
        ridge.fit(self.X_train, self.y_train)
        lassolarscv.fit(self.X_train, self.y_train)
        elasticnetcv.fit(self.X_train, self.y_train)

        lassocv_pred = lassocv.predict(self.X_test)
        ridge_pred = ridge.predict(self.X_test)
        lassolarscv_pred = lassolarscv.predict(self.X_test)
        elasticnetcv_pred = elasticnetcv.predict(self.X_test)

        df=pd.DataFrame()
        df["Lasso"] = lassocv_pred
        df["Ridge"] = ridge_pred
        df["LassoLars"] = lassolarscv_pred
        df["Elasticnetcv"] = elasticnetcv_pred
        df["Y"] = self.y_test.reset_index(drop=True)
        return df


    def preprocess_data(self):
        train = self.train.drop(labels=["Id"], axis=1)
        # test = self.test.drop(labels=["Id"], axis=1)
        train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index).reset_index(drop=True)
        train_len = len(train)
        # dataset = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)
        dataset = train.fillna(np.nan)
        dataset["Alley"] = dataset["Alley"].fillna("No")
        dataset["MiscFeature"] = dataset["MiscFeature"].fillna("No")
        dataset["Fence"] = dataset["Fence"].fillna("No")
        dataset["PoolQC"] = dataset["PoolQC"].fillna("No")
        dataset["FireplaceQu"] = dataset["FireplaceQu"].fillna("No")
        dataset["Utilities"] = dataset["Utilities"].fillna("AllPub")
        dataset["BsmtCond"] = dataset["BsmtCond"].fillna("No")
        dataset["BsmtQual"] = dataset["BsmtQual"].fillna("No")
        dataset["BsmtFinType2"] = dataset["BsmtFinType2"].fillna("No")
        dataset["BsmtFinType1"] = dataset["BsmtFinType1"].fillna("No")
        dataset.loc[dataset["BsmtCond"] == "No", "BsmtUnfSF"] = 0
        dataset.loc[dataset["BsmtFinType1"] == "No", "BsmtFinSF1"] = 0
        dataset.loc[dataset["BsmtFinType2"] == "No", "BsmtFinSF2"] = 0
        dataset.loc[dataset["BsmtQual"] == "No", "TotalBsmtSF"] = 0
        dataset.loc[dataset["BsmtCond"] == "No", "BsmtHalfBath"] = 0
        dataset.loc[dataset["BsmtCond"] == "No", "BsmtFullBath"] = 0
        dataset["BsmtExposure"] = dataset["BsmtExposure"].fillna("No")
        dataset["SaleType"] = dataset["SaleType"].fillna("WD")
        dataset["MSZoning"] = dataset["MSZoning"].fillna("RL")
        dataset["KitchenQual"] = dataset["KitchenQual"].fillna("TA")
        dataset["GarageType"] = dataset["GarageType"].fillna("No")
        dataset["GarageFinish"] = dataset["GarageFinish"].fillna("No")
        dataset["GarageQual"] = dataset["GarageQual"].fillna("No")
        dataset["GarageCond"] = dataset["GarageCond"].fillna("No")
        dataset.loc[dataset["GarageType"] == "No", "GarageYrBlt"] = dataset["YearBuilt"][dataset["GarageType"] == "No"]
        dataset.loc[dataset["GarageType"] == "No", "GarageCars"] = 0
        dataset.loc[dataset["GarageType"] == "No", "GarageArea"] = 0
        dataset["GarageArea"] = dataset["GarageArea"].fillna(dataset["GarageArea"].median())
        dataset["GarageCars"] = dataset["GarageCars"].fillna(dataset["GarageCars"].median())
        dataset["GarageYrBlt"] = dataset["GarageYrBlt"].fillna(dataset["GarageYrBlt"].median())
        dataset["Functional"] = dataset["Functional"].fillna("Typ")
        dataset["Exterior2nd"] = dataset["Exterior2nd"].fillna("VinylSd")
        dataset["Exterior1st"] = dataset["Exterior1st"].fillna("VinylSd")
        dataset["Electrical"] = dataset["Electrical"].fillna("SBrkr")
        dataset["MasVnrType"] = dataset["MasVnrType"].fillna("None")
        dataset.loc[dataset["MasVnrType"] == "None", "MasVnrArea"] = 0
        dataset = dataset.replace({'MSSubClass': {20: 'SubClass_20', 30: 'SubClass_30', 40: 'SubClass_40',
                                                  45: 'SubClass_45', 50: 'SubClass_50', 60: 'SubClass_60',
                                                  70: 'SubClass_70',
                                                  75: 'SubClass_75', 80: 'SubClass_80', 85: 'SubClass_85',
                                                  90: 'SubClass_90',
                                                  120: 'SubClass_120', 150: 'SubClass_150', 160: 'SubClass_160',
                                                  180: 'SubClass_180',
                                                  190: 'SubClass_190'}})
        dataset = dataset.replace({'MoSold': {1: 'Jan', 2: 'Feb', 3: 'Mar',
                                              4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct',
                                              11: 'Nov', 12: 'Dec'}})
        dataset['YrSold'] = dataset['YrSold'].astype(str)
        dataset["BsmtCond"] = dataset["BsmtCond"].astype("category", categories=['No', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
                                                         ordered=True).cat.codes
        dataset["BsmtExposure"] = dataset["BsmtExposure"].astype("category", categories=['No', 'Mn', 'Av', 'Gd'],
                                                                 ordered=True).cat.codes
        dataset["BsmtFinType1"] = dataset["BsmtFinType1"].astype("category",
                                                                 categories=['No', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ',
                                                                             'GLQ'], ordered=True).cat.codes
        dataset["BsmtFinType2"] = dataset["BsmtFinType2"].astype("category",
                                                                 categories=['No', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ',
                                                                             'GLQ'], ordered=True).cat.codes
        dataset["BsmtQual"] = dataset["BsmtQual"].astype("category", categories=['No', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
                                                         ordered=True).cat.codes
        dataset["ExterCond"] = dataset["ExterCond"].astype("category", categories=['Po', 'Fa', 'TA', 'Gd', 'Ex'],
                                                           ordered=True).cat.codes
        dataset["ExterQual"] = dataset["ExterQual"].astype("category", categories=['Po', 'Fa', 'TA', 'Gd', 'Ex'],
                                                           ordered=True).cat.codes
        dataset["Fence"] = dataset["Fence"].astype("category", categories=['No', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv'],
                                                   ordered=True).cat.codes
        dataset["FireplaceQu"] = dataset["FireplaceQu"].astype("category", categories=['No', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
                                                               ordered=True).cat.codes
        dataset["Functional"] = dataset["Functional"].astype("category",
                                                             categories=['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2',
                                                                         'Min1', 'Typ'], ordered=True).cat.codes
        dataset["GarageCond"] = dataset["GarageCond"].astype("category", categories=['No', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
                                                             ordered=True).cat.codes
        dataset["GarageFinish"] = dataset["GarageFinish"].astype("category", categories=['No', 'Unf', 'RFn', 'Fin'],
                                                                 ordered=True).cat.codes
        dataset["GarageQual"] = dataset["GarageQual"].astype("category", categories=['No', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
                                                             ordered=True).cat.codes
        dataset["HeatingQC"] = dataset["HeatingQC"].astype("category", categories=['Po', 'Fa', 'TA', 'Gd', 'Ex'],
                                                           ordered=True).cat.codes
        dataset["KitchenQual"] = dataset["KitchenQual"].astype("category", categories=['Po', 'Fa', 'TA', 'Gd', 'Ex'],
                                                               ordered=True).cat.codes
        dataset["PavedDrive"] = dataset["PavedDrive"].astype("category", categories=['N', 'P', 'Y'], ordered=True).cat.codes
        dataset["PoolQC"] = dataset["PoolQC"].astype("category", categories=['No', 'Fa', 'TA', 'Gd', 'Ex'],
                                                     ordered=True).cat.codes
        dataset["Utilities"] = dataset["Utilities"].astype("category", categories=['ELO', 'NoSeWa', 'NoSewr', 'AllPub'],
                                                           ordered=True).cat.codes
        dataset = pd.get_dummies(dataset, columns=["Alley", "BldgType", "CentralAir",
                                                   "Condition1", "Condition2", "Electrical", "Exterior1st", "Exterior2nd",
                                                   "Foundation",
                                                   "GarageType", "Heating", "HouseStyle", "LandContour", "LandSlope",
                                                   "LotConfig", "LotShape",
                                                   "MSZoning", "MasVnrType", "MiscFeature", "Neighborhood", "RoofMatl",
                                                   "RoofStyle",
                                                   "SaleCondition", "SaleType", "Street", "MSSubClass", 'MoSold', 'YrSold'],
                                 drop_first=True)
        dataset = dataset.drop(labels=[ 'Condition2_PosN',
                                       'MSSubClass_SubClass_160'], axis=1)
        skewed_features = ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "GarageArea", "MasVnrArea"
            , "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "3SsnPorch", "EnclosedPorch",
                           "GrLivArea", "LotArea", "LowQualFinSF", "OpenPorchSF", "PoolArea",
                           "ScreenPorch", "WoodDeckSF"]
        for feature in skewed_features:
            dataset[feature] = np.log1p(dataset[feature])
        dataset["SalePrice"] = np.log1p(dataset["SalePrice"])
        y = dataset["SalePrice"]
        X = dataset.drop(labels="SalePrice", axis=1)
        X = X.drop(labels="LotFrontage", axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

        return X_train, X_test, y_train, y_test


    def RMSE(self, estimator,X_train, Y_train, cv=5,n_jobs=4):
        cv_results = cross_val_score(estimator,X_train,Y_train,cv=cv,scoring="neg_mean_squared_error",n_jobs=n_jobs)
        return (np.sqrt(-cv_results)).mean()


    def objective_function(self, weights):
        pred = deepcopy(self.predictions)
        pred["Lasso"] = weights[0]*pred["Lasso"]
        pred["Ridge"] = weights[0] * pred["Ridge"]
        pred["LassoLars"] = weights[0] * pred["LassoLars"]
        pred["Elasticnetcv"] = weights[0] * pred["Elasticnetcv"]
        pred["Y_pred"] = pred.iloc[:, 0:4].sum(axis=1)
        fitness = -(sum((pred["Y_pred"] - pred["Y"])**2))**0.5
        return fitness,


if __name__ == "__main__":
    n=4
    model = CreateModels()
    simple_avg = model.objective_function([0.25, 0.25, 0.25, 0.25])
    print("Simple Average Result :", [0.25, 0.25, 0.25, 0.25], simple_avg)

    wo = WeightsOptimizer(n, model)
    optimized_weights = wo.ga()

    print("Optimized Average Result")
    print(optimized_weights)



