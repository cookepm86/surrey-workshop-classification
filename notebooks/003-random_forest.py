#!/usr/bin/env python
# coding: utf-8

# # Random Forest Model
# 
# Very robust and successful non-parametric machine-learning model https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf
# Very few assumptions!

get_ipython().run_line_magic('load_ext', 'autotime')
from itertools import chain
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from pprint import pprint
from seaborn import boxplot
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import FeatureUnion, Pipeline

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
for n in range(1, 7):
    df[f"BILL_AMT_NORM{n}"] = df[f"BILL_AMT{n}"] / df["LIMIT_BAL"]
for n in range(1, 6):
    df[f"PAY_AMT_NORM{n}"] = (0.01 + df[f"PAY_AMT{n}"]) / (0.01 + df[f"BILL_AMT{n+1}"])
    df[f"LAST_PAY_DIFF{n}"] = df[f"BILL_AMT{n+1}"] - df[f"PAY_AMT{n}"]
df["PAY_AMT_NORM6"] = df["PAY_AMT6"] / df["LIMIT_BAL"]


# ## Train/Test Split

X, y = df, df["DEFAULT"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)


# ## Model Pipeline
# 
# Let's build a simple pipeline which uses a Random Forest model to predict *DEFAULT*.

class DataFrameSelector(BaseEstimator, TransformerMixin):
    """
    Select columns from pandas dataframe by specifying a list of column names
    """
    def __init__(self, attribute_names=None):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.loc[:, self.attribute_names]
    
    def get_feature_names(self):
        return self.attribute_names
    
    
def create_random_forest_pipeline(X_train, 
                                  y_train,
                                  param_grid):
    """
    Create a Random Forest pipeline which
    performs basic preprocessing for numerical
    and categorical features.

    :param X_train: Training feature data
    :param y_train: Training target data
    :param param_grid: Hyperparameter grid to search over
    """
    # build numerical feature pipeline
    num_feature_selector = DataFrameSelector()
    num_pipeline = Pipeline([("feature_selector", num_feature_selector)])
    
    # build categorical feature pipeline
    cat_feature_selector = DataFrameSelector()
    cat_one_hot_encoder = OneHotEncoder(handle_unknown="ignore", drop="first")
    cat_pipeline = Pipeline(
        [
            ("feature_selector", cat_feature_selector),
            ("one_hot_encoder", cat_one_hot_encoder)
        ]
    )
    
    # combine preprocessing pipelines
    union_pipeline = FeatureUnion(
        [
            ("num_pipeline", num_pipeline),
            ("cat_pipeline", cat_pipeline)
        ],
        n_jobs=1,
        transformer_weights=None
    )
    
    # combine preprocessing with logistic regression model
    clf = Pipeline(
        steps=[
            ("union", union_pipeline),
            ("classifier", RandomForestClassifier(random_state=SEED, n_jobs=1))
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


# ## Feature Selection
# 
# We don't have many features, so we can do a fair bit by hand.
# We start with the most important/correlated features, as identified in the EDA notebooks
# and we incrementally add features in order of (anti-)correlation.
# 
# We try *BILL_AMTN* AND *BILL_AMT_NORMN* as features to add.

param_grid = {
    "union__num_pipeline__feature_selector__attribute_names": [
        ["LIMIT_BAL"], 
        ["LIMIT_BAL"] + [f"PAY_AMT{n}" for n in range(1, 7)],
        ["LIMIT_BAL"] + [f"PAY_AMT_NORM{n}" for n in range(1, 6)],
        ["LIMIT_BAL"] + [f"BILL_AMT{n}" for n in range(1, 7)],
        ["LIMIT_BAL"] + [f"BILL_AMT_NORM{n}" for n in range(1, 7)],
    ],
    "union__cat_pipeline__feature_selector__attribute_names": [
        ["PAY_1"], 
        [f"PAY_{n}" for n in range(1, 7)],
    ],
    "classifier__n_estimators": [100],
    "classifier__max_depth": [5, 10],
    "classifier__max_features": ["auto"],
    "classifier__class_weight": [None]
}

gs = create_random_forest_pipeline(X_train, y_train, param_grid)
validation_auc = gs.best_score_
test_auc = roc_auc_score(y_test, gs.predict_proba(X_test)[:, 1])

print(f"Validation ROC-AUC score: {round(100*validation_auc, 2)}")
print(f"Test ROC-AUC score: {round(100*test_auc, 2)}\n")
pprint(gs.best_params_)


# Next, let's try adding *PAY_AMTN* OR *PAY_AMT_NORMN*.

param_grid = {
    "union__num_pipeline__feature_selector__attribute_names": [
        ["LIMIT_BAL"] + [f"BILL_AMT_NORM{n}" for n in range(1, 7)],
        ["LIMIT_BAL"] + [f"BILL_AMT_NORM{n}" for n in range(1, 7)] + [f"PAY_AMT{n}" for n in range(1, 6)],
        ["LIMIT_BAL"] + [f"BILL_AMT_NORM{n}" for n in range(1, 7)] + [f"PAY_AMT_NORM{n}" for n in range(1, 6)],
    ],
    "union__cat_pipeline__feature_selector__attribute_names": [
        [f"PAY_{n}" for n in range(1, 7)],
    ],
    "classifier__n_estimators": [100],
    "classifier__max_depth": [5, 10],
    "classifier__max_features": ["auto"],
    "classifier__class_weight": [None]
}

gs = create_random_forest_pipeline(X_train, y_train, param_grid)
validation_auc = gs.best_score_
test_auc = roc_auc_score(y_test, gs.predict_proba(X_test)[:, 1])

print(f"Validation ROC-AUC score: {round(100*validation_auc, 2)}")
print(f"Test ROC-AUC score: {round(100*test_auc, 2)}\n")
pprint(gs.best_params_)


# Try adding *PAY_AMT6* or *PAY_AMT_NORM6*.

param_grid = {
    "union__num_pipeline__feature_selector__attribute_names": [
        ["LIMIT_BAL"] 
        + [f"BILL_AMT_NORM{n}" for n in range(1, 7)] 
        + [f"PAY_AMT_NORM{n}" for n in range(1, 6)],
        ["LIMIT_BAL", "PAY_AMT6"] 
        + [f"BILL_AMT_NORM{n}" for n in range(1, 7)] 
        + [f"PAY_AMT_NORM{n}" for n in range(1, 6)],
        ["LIMIT_BAL", "PAY_AMT_NORM6"] 
        + [f"BILL_AMT_NORM{n}" for n in range(1, 7)] 
        + [f"PAY_AMT_NORM{n}" for n in range(1, 6)],
    ],
    "union__cat_pipeline__feature_selector__attribute_names": [
        [f"PAY_{n}" for n in range(1, 7)],
    ],
    "classifier__n_estimators": [100],
    "classifier__max_depth": [5, 10],
    "classifier__max_features": ["auto"],
    "classifier__class_weight": [None]
}

gs = create_random_forest_pipeline(X_train, y_train, param_grid)
validation_auc = gs.best_score_
test_auc = roc_auc_score(y_test, gs.predict_proba(X_test)[:, 1])

print(f"Validation ROC-AUC score: {round(100*validation_auc, 2)}")
print(f"Test ROC-AUC score: {round(100*test_auc, 2)}\n")
pprint(gs.best_params_)


# Try adding *LAST_DAY_DIFFN*.

param_grid = {
    "union__num_pipeline__feature_selector__attribute_names": [
        ["LIMIT_BAL", "PAY_AMT6"] 
        + [f"BILL_AMT_NORM{n}" for n in range(1, 7)] 
        + [f"PAY_AMT_NORM{n}" for n in range(1, 6)],
        ["LIMIT_BAL", "PAY_AMT6"] 
        + [f"BILL_AMT_NORM{n}" for n in range(1, 7)] 
        + list(chain(*[[f"PAY_AMT_NORM{n}", f"LAST_PAY_DIFF{n}"] for n in range(1, 6)])),
    ],
    "union__cat_pipeline__feature_selector__attribute_names": [
        [],
        [f"PAY_{n}" for n in range(1, 7)],
    ],
    "classifier__n_estimators": [100],
    "classifier__max_depth": [5, 10],
    "classifier__max_features": ["auto"],
    "classifier__class_weight": [None]
}

gs = create_random_forest_pipeline(X_train, y_train, param_grid)
validation_auc = gs.best_score_
test_auc = roc_auc_score(y_test, gs.predict_proba(X_test)[:, 1])

print(f"Validation ROC-AUC score: {round(100*validation_auc, 2)}")
print(f"Test ROC-AUC score: {round(100*test_auc, 2)}\n")
pprint(gs.best_params_)


# Try adding the demographic features.

param_grid = {
    "union__num_pipeline__feature_selector__attribute_names": [
        ["LIMIT_BAL", "PAY_AMT6"] 
        + [f"BILL_AMT_NORM{n}" for n in range(1, 7)] 
        + [f"PAY_AMT_NORM{n}" for n in range(1, 6)],
        ["LIMIT_BAL", "PAY_AMT6", "AGE"] 
        + [f"BILL_AMT_NORM{n}" for n in range(1, 7)] 
        + [f"PAY_AMT_NORM{n}" for n in range(1, 6)],
    ],
    "union__cat_pipeline__feature_selector__attribute_names": [
        [f"PAY_{n}" for n in range(1, 7)],
        ["EDUCATION"] + [f"PAY_{n}" for n in range(1, 7)],
        ["MARRIAGE"] + [f"PAY_{n}" for n in range(1, 7)],
        ["SEX"] + [f"PAY_{n}" for n in range(1, 7)],
        ["EDUCATION", "MARRIAGE"] + [f"PAY_{n}" for n in range(1, 7)],
        ["EDUCATION", "SEX"] + [f"PAY_{n}" for n in range(1, 7)],
        ["MARRIAGE", "SEX"] + [f"PAY_{n}" for n in range(1, 7)],
        ["EDUCATION", "MARRIAGE", "SEX"] + [f"PAY_{n}" for n in range(1, 7)],
    ],
    "classifier__n_estimators": [100],
    "classifier__max_depth": [5, 10],
    "classifier__max_features": ["auto"],
    "classifier__class_weight": [None]
}

gs = create_random_forest_pipeline(X_train, y_train, param_grid)
validation_auc = gs.best_score_
test_auc = roc_auc_score(y_test, gs.predict_proba(X_test)[:, 1])

print(f"Validation ROC-AUC score: {round(100*validation_auc, 2)}")
print(f"Test ROC-AUC score: {round(100*test_auc, 2)}\n")
pprint(gs.best_params_)


# We can examine the most important features in the model.

# get model coefficients
feature_importance_df =     pd.DataFrame(
    zip(
        gs.best_estimator_["union"].transformer_list[0][1]["feature_selector"].get_feature_names()
        + gs.best_estimator_["union"].transformer_list[1][1]["one_hot_encoder"].get_feature_names_out().tolist(),
        100*gs.best_estimator_["classifier"].feature_importances_
    ),
    columns=["feature", "coef_"])\
    .set_index("feature")\
    .sort_values("coef_", ascending=False)

feature_importance_df    .head(10)    .round(2)


# And the least important!
# We could look at removing some of the least important features, 
# or rolling them into another feature.

feature_importance_df    .tail(10)    .round(4)


# ## Hyperparameter Tuning
# 
# Now that we have performed feature selection 
# (with close to the default hyperparameters for Random Forest),
# we will tune the hyperparameters.

param_grid = {
    "union__num_pipeline__feature_selector__attribute_names": [
        ["LIMIT_BAL", "PAY_AMT6", "AGE"] 
        + [f"BILL_AMT_NORM{n}" for n in range(1, 7)] 
        + [f"PAY_AMT_NORM{n}" for n in range(1, 6)],
    ],
    "union__cat_pipeline__feature_selector__attribute_names": [
        ["EDUCATION", "MARRIAGE"] + [f"PAY_{n}" for n in range(1, 7)],
    ],
    "classifier__n_estimators": [50, 100, 150, 200, 250],
    "classifier__max_depth": [5, 10, 15],
    "classifier__max_features": [3, "auto"],
    "classifier__class_weight": [
        None, 
        {0: 1, 1: 1.5},
        {0: 1, 1: 2},
        {0: 1, 1: 2.5},
        {0: 1, 1: 3},
        "balanced"
    ],
}

gs = create_random_forest_pipeline(X_train, y_train, param_grid)
validation_auc = gs.best_score_
test_auc = roc_auc_score(y_test, gs.predict_proba(X_test)[:, 1])

print(f"Validation ROC-AUC score: {round(100*validation_auc, 2)}")
print(f"Test ROC-AUC score: {round(100*test_auc, 2)}\n")
pprint(gs.best_params_)


# get model coefficients
feature_importance_df =     pd.DataFrame(
    zip(
        gs.best_estimator_["union"].transformer_list[0][1]["feature_selector"].get_feature_names()
        + gs.best_estimator_["union"].transformer_list[1][1]["one_hot_encoder"].get_feature_names_out().tolist(),
        100*gs.best_estimator_["classifier"].feature_importances_
    ),
    columns=["feature", "coef_"])\
    .set_index("feature")\
    .sort_values("coef_", ascending=False)

feature_importance_df    .head(10)    .round(2)


# ## Evaluating the Model
# 
# Let's compare our model to the baseline model (and to results in the literature).

with open("../models/baseline.pkl", "rb") as f:
    baseline_model = pickle.load(f)


# Let's get the accuracy to compare to https://bradzzz.gitbooks.io/ga-dsi-seattle/content/dsi/dsi_05_classification_databases/2.1-lesson/assets/datasets/DefaultCreditCardClients_yeh_2009.pdf

y_pred = gs.best_estimator_.predict_proba(X_test)[:, 1]
y_baseline_pred = baseline_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred >= 0.5)
baseline_acc = accuracy_score(y_test, y_baseline_pred >= 0.5)

print(f"Model accuracy: {round(100*acc, 2)}")
print(f"Baseline model accuracy: {round(100*baseline_acc, 2)}")


# We can evaluate this over a range of thresholds.

thresholds = np.linspace(0, 1, 100)

accuracies = [accuracy_score(y_test, y_pred >= _t) for _t in thresholds]
baseline_accuracies = [accuracy_score(y_test, y_baseline_pred >= _t) for _t in thresholds]

print(f"Maximum model accuracy: {round(100*max(accuracies), 2)}")
print(f"Maximum baseline model accuracy: {round(100*max(baseline_accuracies), 2)}")

fig, ax = plt.subplots()
ax.plot(thresholds, accuracies, label="RF")
ax.plot(thresholds, baseline_accuracies, label="Baseline")
ax.set(title="Accuracy by Threshold", xlabel="Threshold", ylabel="Accuracy")
ax.legend()
plt.show()


# ### Recall Operator Characteristic Curve

print(f"Model ROC-AUC score: {round(100*roc_auc_score(y_test, y_pred), 2)}")
print(f"Baseline model ROC-AUC score: {round(100*roc_auc_score(y_test, y_baseline_pred), 2)}")

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
fpr_baseline, tpr_baseline, thresholds_baseline = roc_curve(y_test, y_baseline_pred)

fig, ax = plt.subplots()
ax.plot(fpr, tpr, label="RF")
ax.plot(fpr_baseline, tpr_baseline, label="Baseline")
ax.plot([0, 1], [0, 1], color="black", linestyle="--")
ax.set(title="ROC Curve", xlabel="FPR", ylabel="TPR")
ax.legend()
plt.show()


# ### Precision/Recall Curve

prec, rec, thresholds = precision_recall_curve(y_test, y_pred)
prec_baseline, rec_baseline, thresholds_baseline = precision_recall_curve(y_test, y_baseline_pred)

fig, ax = plt.subplots()
ax.plot(prec, rec, label="RF")
ax.plot(prec_baseline, rec_baseline, label="Baseline")
ax.set(title="Precision-Recall Curve", xlabel="Precision", ylabel="Recall")
ax.legend()
plt.show()


# Plot the same thing but with threshold explicit.

prec, rec, thresholds = precision_recall_curve(y_test, y_pred)
prec_baseline, rec_baseline, thresholds_baseline = precision_recall_curve(y_test, y_baseline_pred)

fig, ax = plt.subplots()
ax.plot(thresholds, prec[:-1], color=kcs[0], label="RF-Precision")
ax.plot(thresholds, rec[:-1], color=kcs[1], label="RF-Recall")
ax.plot(thresholds_baseline, prec_baseline[:-1], color=kcs[0], linestyle="--", label="Baseline-Precision")
ax.plot(thresholds_baseline, rec_baseline[:-1], color=kcs[1], linestyle="--", label="Baseline-Recall")
ax.set(title="Precision-Recall Curve", xlabel="Threshold", ylabel="Score")
ax.legend()
plt.show()


# ### Model Bias

test_df = df.loc[X_test.index].copy()
# get the residuals/squared deviance
test_df["RESIDUALS"] = -2*(y_test*np.log(y_pred) + (1 - y_test)*np.log(1 - y_pred))

fig, ax = plt.subplots()
ax.hist(test_df["RESIDUALS"], bins=np.arange(0, 1.01, 0.05))
plt.show()


test_df["DEFAULT_PRED_PROBA"] = y_pred
test_df["BINNED_PRED_DEFAULT"] =     np.array(pd.cut(test_df["DEFAULT_PRED_PROBA"], np.arange(0, 1.01, 0.2), labels=range(1, 6)))

fig, ax = plt.subplots(figsize=(20, 8))
boxplot(x="BINNED_PRED_DEFAULT", y="RESIDUALS", data=test_df, ax=ax)
plt.show()


# ## Save Model

with open("../models/random_forest.pkl", "wb") as f:
    pickle.dump(gs.best_estimator_, f)


# ## Compare with No Feature Engineering

param_grid = {
    "union__num_pipeline__feature_selector__attribute_names": [
        ["LIMIT_BAL", "PAY_AMT6", "AGE"] 
        + [f"BILL_AMT_NORM{n}" for n in range(1, 7)] 
        + [f"PAY_AMT_NORM{n}" for n in range(1, 6)],
    ],
    "union__cat_pipeline__feature_selector__attribute_names": [
        ["EDUCATION", "MARRIAGE"] + [f"PAY_{n}" for n in range(1, 7)],
    ],
    "classifier__n_estimators": [200],
    "classifier__max_depth": [10],
    "classifier__max_features": ["auto"],
    "classifier__class_weight": [
        {0: 1, 1: 1.5},
    ],
}

gs = create_random_forest_pipeline(X_train, y_train, param_grid)
validation_auc = gs.best_score_
test_auc = roc_auc_score(y_test, gs.predict_proba(X_test)[:, 1])

print(f"Validation ROC-AUC score: {round(100*validation_auc, 2)}")
print(f"Test ROC-AUC score: {round(100*test_auc, 2)}\n")
pprint(gs.best_params_)


param_grid = {
    "union__num_pipeline__feature_selector__attribute_names": [
        ["LIMIT_BAL", "PAY_AMT6", "AGE"] 
        + [f"BILL_AMT{n}" for n in range(1, 7)] 
        + [f"PAY_AMT{n}" for n in range(1, 6)],
    ],
    "union__cat_pipeline__feature_selector__attribute_names": [
        ["EDUCATION", "MARRIAGE"] + [f"PAY_{n}" for n in range(1, 7)],
    ],
    "classifier__n_estimators": [200],
    "classifier__max_depth": [10],
    "classifier__max_features": ["auto"],
    "classifier__class_weight": [
        {0: 1, 1: 1.5},
    ],
}

gs = create_random_forest_pipeline(X_train, y_train, param_grid)
validation_auc = gs.best_score_
test_auc = roc_auc_score(y_test, gs.predict_proba(X_test)[:, 1])

print(f"Validation ROC-AUC score: {round(100*validation_auc, 2)}")
print(f"Test ROC-AUC score: {round(100*test_auc, 2)}\n")
pprint(gs.best_params_)


# ## Conclusion
# 
# * Feature engineering important in improving model performance
# * Improve on baseline performance - incremental in terms of accuracy
# * Further investigations:
#     * Could do some sampling
#     * Remove unimportant features
#     * Merge redundant categories
