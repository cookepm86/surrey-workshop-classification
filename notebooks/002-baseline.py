#!/usr/bin/env python
# coding: utf-8

# # Baseline Model
# 
# To give any model metrics context, 
# we need a baseline to compare against.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from seaborn import boxplot
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

pd.set_option("max_columns", 100)
plt.style.use("/Users/miccoo/Desktop/kindred.mplstyle")
kcs = plt.rcParams["axes.prop_cycle"].by_key()["color"]
SEED = 42


# ## Load Data

df = pd.read_excel("../data/default of credit card clients.xls", header=1, index_col=0)
print(df.index.is_unique)
df.head()


# ## Basic Preprocessing

df.rename(
    columns={
        "default payment next month": "DEFAULT",
        "PAY_0": "PAY_1"
    }, 
    inplace=True
)
df["SEX"] -= 1
df["LOG_LIMIT_BAL"] = np.log(df["LIMIT_BAL"])


# ## Train/Test Split

X, y = df, df["DEFAULT"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)


# ## Model Pipeline
# 
# Let's build a simple pipeline which uses a logistic regression model with 2 features
# to predict the target variable (*DEFAULT*).
# These were found to be two of the most correlated variables in the EDA notebook.

def create_logistic_regression_pipeline(X_train, 
                                        y_train, 
                                        numerical_features, 
                                        categorical_features, 
                                        param_grid):
    """
    Create a logistic regression pipeline which
    performs basic preprocessing for numerical
    and categorical features.

    :param X_train: Training feature data
    :param y_train: Training target data
    :param numerical_features: List of numerical features
    :param categorical_features: List of categorical features
    :param param_grid: Hyperparameter grid to search over
    """
    # one-hot encode categorical variables
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", drop="first")
    # all features should be on a similar scale for a logistic regression model
    numerical_transformer = StandardScaler()
    column_preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    
    # combine preprocessing with logistic regression model
    clf = Pipeline(
        steps=[
            ("column_preprocessor", column_preprocessor),
            ("classifier", LogisticRegression(random_state=SEED, solver="liblinear"))
        ]
    )
    
    # perform hyperparameter tuning via cross-validation
    gs = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        scoring="roc_auc",
        n_jobs=5,
        cv=5
    )
    gs.fit(X_train, y_train)
    
    return gs


numerical_features = ["LOG_LIMIT_BAL"]
categorical_features = ["PAY_1"]

param_grid = {
    "classifier__C": [1e-4, 1e-3, 1e-2, 1e-1, 1],
    "classifier__penalty": ["l1", "l2"]
}

gs = create_logistic_regression_pipeline(
    X_train,
    y_train,
    numerical_features=numerical_features,
    categorical_features=categorical_features,
    param_grid=param_grid
)
validation_auc = gs.best_score_
test_auc = roc_auc_score(y_test, gs.predict_proba(X_test)[:, 1])

print(f"Validation ROC-AUC score: {round(validation_auc, 2)}")
print(f"Test ROC-AUC score: {round(test_auc, 2)}")
print(gs.best_params_)


# We can check the feature importances of the model.

# get model coefficients
pd.DataFrame(
    zip(
        ["bias"] + gs.best_estimator_["column_preprocessor"].get_feature_names_out().tolist(),
        gs.best_estimator_["classifier"].intercept_.reshape(-1).tolist()
        + gs.best_estimator_["classifier"].coef_.reshape(-1).tolist()
    ),
    columns=["feature", "coef_"])\
    .set_index("feature")\
    .sort_values("coef_")


# ## Model Bias
# 
# Let's check the residuals.
# Seems like we are missing some explanatory variables.

y_pred = gs.predict_proba(X_test)[:, 1]

test_df = df.loc[X_test.index].copy()
# get the residuals/squared deviance
test_df["RESIDUALS"] = -2*(y_test*np.log(y_pred) + (1 - y_test)*np.log(1 - y_pred))

fig, ax = plt.subplots()
ax.hist(test_df["RESIDUALS"], bins=np.arange(0, 1.01, 0.05))
plt.show()


# The residuals are slightly correlated with the predictions, 
# but we'll ignore this for our baseline.

test_df["DEFAULT_PRED_PROBA"] = y_pred
test_df["BINNED_PRED_DEFAULT"] =     np.array(pd.cut(test_df["DEFAULT_PRED_PROBA"], np.arange(0, 1.01, 0.2), labels=range(1, 6)))

fig, ax = plt.subplots(figsize=(20, 8))
boxplot(x="BINNED_PRED_DEFAULT", y="RESIDUALS", data=test_df, ax=ax)
plt.show()


test_df[["RESIDUALS", "DEFAULT_PRED_PROBA"]].corr()


# ## Save Model

with open("../models/baseline.pkl", "wb") as f:
    pickle.dump(gs.best_estimator_, f)


# # Conclusion
# 
# * We have a simple baseline result to give future results context
# * Some issues with the residuals, but doesn't need to be perfect as it's a baseline
# * Could play with features/different scaling/different categorical encoding
