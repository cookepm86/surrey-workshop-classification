#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis (EDA)
# 
# Want to explore data before model building.
# In particular:
# * Check data quality
# * Understand feature/target distributions
# * Identify import features by correlation
# * Understand correlations between features
# 
# **Equivalent:** if we had to build a logistic regression model with a small number of features, how would I build this model?
# Check out https://www.statlearning.com/ for a great introduction to logistic regression and Random Forest model
# which we will use later.

from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import poisson, probplot
from seaborn import boxplot, heatmap
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from kindred_cmap import double as kindred_cmap

pd.set_option("max_columns", 100)
plt.style.use("/Users/miccoo/Desktop/kindred.mplstyle")
kcs = plt.rcParams["axes.prop_cycle"].by_key()["color"]
SEED = 42


# ## Load Data
# 
# Taken from https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients.

Image(url="legend.png")


df = pd.read_excel("../data/default of credit card clients.xls", header=1, index_col=0)
print(f"The client IDs are unique: {df.index.is_unique}")
df.head()


# ## Data Quality
# 
# Looks good initially, we don't have any nulls and all categorical features are encoded as integers.

df.info(verbose=True)


# Questions to the stakeholders:
# * Do we have a benchmark? **TODO:** move to other NB
# * Is there any additional data they think might be useful?
# * Is any of the data known to be unreliable?
# * Can they explain any fields which we're not sure of the meaning?

# ## Basic Preprocessing
# 
# Columns names with whitespace are a pain, and it looks like the 0 subscript for PAY is misleading.

df.rename(
    columns={
        "default payment next month": "DEFAULT",
        "PAY_0": "PAY_1"
    }, 
    inplace=True
)
df["SEX"] -= 1


# ## Univariate Distributions

# ### Target
# 
# The target distribution is quite **imbalanced**. 
# This will inform metric and model choice - 
# in particular, are false positives (FPs) more important than false negatives (FNs) or vice versa.

churn_val_counts = df["DEFAULT"].value_counts(normalize=True)
print(f"{round(100*churn_val_counts.loc[1], 2)}% of clients default.")

fig, ax = plt.subplots()
ax.bar(churn_val_counts.index, churn_val_counts.values)
ax.set(
    title="Default Distribution", xlabel="Default", ylabel="Fraction", 
    xticks=[0, 1], xticklabels=["False", "True"]
)
plt.show()


# Let's get a glimpse of correlations.
# We choose a Spearman correlation with the Random Forest in mind,
# where only ordering of features matters.
# 
# The *PAY_N* features are the most correlated with the target, 
# which is not surprising.
# *LIMIT_BAL* is anti-correlated with the target variable, 
# as are *PAY_AMTN*.

corr_df = df.corr("spearman")

corr_df["DEFAULT"].sort_values()


# ### Features

# #### Discrete

# ##### Sex

# We have 2 sexes - mostly female.

sex_dict = {0: "male", 1: "female"}

sex_val_counts = df["SEX"].value_counts(normalize=True)
print(f"{round(100*sex_val_counts.loc[1], 2)}% of clients are female.")

sex_default_matrix =     df    .groupby(["SEX", "DEFAULT"])    ["LIMIT_BAL"]    .count()    .rename("COUNT")    .reset_index()    .pivot_table(values="COUNT", index=["SEX"], columns=["DEFAULT"])
# normalize by counts
sex_default_matrix /= sex_default_matrix.sum(axis=1).values.reshape(-1, 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
ax1.bar(sex_val_counts.index, sex_val_counts.values)
ax1.set(
    title="Sex Distribution", xlabel="Sex", ylabel="Fraction", 
    xticks=[0, 1], xticklabels=["Male", "Female"]
)

heatmap(
    sex_default_matrix[[1]], 
    linewidth=5, 
    annot=True, 
    annot_kws={"size": 24}, 
    cmap=kindred_cmap
)
ax2.set(
    title="Default Rate by Sex", xlabel="Default Rate", ylabel="Sex", 
    xticks=[0.5], xticklabels=[],
    yticks=[0.5, 1.5], yticklabels=["Male", "Female"]
)

plt.show()


# There are no particularly strong correlations.

corr_df["SEX"].sort_values()


# ##### Education
# 
# Most customers have university or graduate school education.
# Some unknown categories - these could be lumped together.
# There are also values 0 (only 14) which are undefined by the legend.

# from the legend
education_dict = {
    0: "Missing",
    1: "Graduate School", 
    2: "University", 
    3: "High School",
    4: "Others",
    5: "Unknown",
    6: "Unknown"
}

education_val_counts = df["EDUCATION"].value_counts(normalize=True)
for n in range(7):
    print(f"{round(100*education_val_counts.loc[n], 2)}% of clients have {education_dict[n]} education.")
    
education_default_matrix =     df    .groupby(["EDUCATION", "DEFAULT"])    ["LIMIT_BAL"]    .count()    .rename("COUNT")    .reset_index()    .pivot_table(values="COUNT", index=["EDUCATION"], columns=["DEFAULT"])    .fillna(0)
education_default_matrix /= education_default_matrix.sum(axis=1).values.reshape(-1, 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
ax1.bar(education_val_counts.index, education_val_counts.values)
ax1.set(
    title="Education Distribution", xlabel="Education", ylabel="Fraction",
    xticks=range(7), xticklabels=[education_dict[n] for n in range(7)]
)

heatmap(
    education_default_matrix[[1]], 
    linewidth=5, 
    vmin=0.0, 
    vmax=0.25, 
    annot=True, 
    annot_kws={"size": 24},
    cmap=kindred_cmap
)
ax2.set(
    title="Default Rate by Education", xlabel="Default Rate", ylabel="Education", 
    xticks=[0.5], xticklabels=[],
    yticks=[n + 0.5 for n in range(7)]
)
ax2.set_yticklabels([education_dict[n] for n in range(7)], rotation=0)

plt.show()


# Education is quite correlated with *LIMIT_BAL*.

corr_df["EDUCATION"].sort_values()


education_df = df.groupby("EDUCATION").mean()
education_df = (education_df - education_df.min()) / (education_df.max() - education_df.min())

fig, ax = plt.subplots(figsize=(20, 5))
heatmap(education_df, cmap=kindred_cmap, vmax=1, vmin=0, ax=ax, linewidth=1)
ax.set_yticks(np.array(range(7)) + 0.5)
ax.set_yticklabels([education_dict[n] for n in range(7)], rotation=0)
plt.show()


# ##### Marriage
# 
# Slightly more married customers.

marriage_dict = {
    0: "Missing",
    1: "Married", 
    2: "Single", 
    3: "Others"
}

marriage_val_counts = df["MARRIAGE"].value_counts(normalize=True)
for n in range(4):
    print(f"{round(100*marriage_val_counts.loc[n], 2)}% of clients have {marriage_dict[n]} marriage.")
        
marriage_default_matrix =     df    .groupby(["MARRIAGE", "DEFAULT"])    ["LIMIT_BAL"]    .count()    .rename("COUNT")    .reset_index()    .pivot_table(values="COUNT", index=["MARRIAGE"], columns=["DEFAULT"])    .fillna(0)
marriage_default_matrix /= marriage_default_matrix.sum(axis=1).values.reshape(-1, 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
ax1.bar(marriage_val_counts.index, marriage_val_counts.values)
ax1.set(
    title="Marriage Distribution", xlabel="Marriage", ylabel="Fraction",
    xticks=range(4), xticklabels=[marriage_dict[n] for n in range(4)]
)

heatmap(
    marriage_default_matrix[[1]], 
    linewidth=5, 
    vmin=0.0, 
    vmax=0.3, 
    annot=True, 
    annot_kws={"size": 24},
    cmap=kindred_cmap
)
ax2.set(
    title="Default Rate by Marriage", xlabel="Default Rate", ylabel="Marriage", 
    xticks=[0.5], xticklabels=[],
    yticks=[n + 0.5 for n in range(4)]
)
ax2.set_yticklabels([marriage_dict[n] for n in range(4)], rotation=0)

plt.show()


# Strong correlation with age.

corr_df["MARRIAGE"].sort_values()


marriage_df = df.groupby("MARRIAGE").mean()
marriage_df = (marriage_df - marriage_df.min()) / (marriage_df.max() - marriage_df.min())

fig, ax = plt.subplots(figsize=(20, 3))
heatmap(marriage_df, cmap=kindred_cmap, vmax=1, vmin=0, ax=ax, linewidth=1)
ax.set_yticks(np.array(range(4)) + 0.5)
ax.set_yticklabels([marriage_dict[n] for n in range(4)], rotation=0)
plt.show()


# #### Repayment Statuses
# 
# What does the -2 value represent? Typically go to the stakeholders to find out.

pay_df =     pd.concat([df[f"PAY_{n}"].value_counts() for n in range(1, 7)], axis=1)    .fillna(0)    .astype(int)    .T
pay_df


fig, ax = plt.subplots()

for n in range(1, 7):
    ax.plot(df.groupby(f"PAY_{n}")["DEFAULT"].mean(), label=f"PAY_{n}")
ax.set(title="Default Rate by Repayment Status", ylim=[0, 1], xlabel="Status", ylabel="Rate")
ax.legend()

plt.show()


# There is a strong correlation with *BILL_AMTN* and anti-correlation with *LIMIT_BAL*.

fig, ax = plt.subplots(figsize=(20, 4))
heatmap(
    corr_df[[f"PAY_{n}" for n in range(1, 7)]].T, 
    vmax=1, 
    vmin=-1, 
    cmap=kindred_cmap, 
    linewidth=1, 
    ax=ax
)
plt.show()


# #### Continuous

# ##### LIMIT_BAL
# 
# All positive values, which we expect.

(df["LIMIT_BAL"] <= 0).sum()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle("LIMIT_BAL", size=26)

ax1.hist(df["LIMIT_BAL"])
ax1.set(title="Linear Scale", xlabel="LIMIT_BAL", ylabel="Count")

ax2.hist(np.log(df["LIMIT_BAL"]))
ax2.set(title="Log Scale", xlabel="Log(LIMIT_BAL)", ylabel="Count")

plt.show()


# There don't appear to be outliers.

df["LIMIT_BAL"].describe()


# Not quite log-normal.

fig, ax = plt.subplots()
probplot(np.log(df["LIMIT_BAL"]), plot=ax)
plt.show()


corr_df["LIMIT_BAL"].sort_values()


fig, ax = plt.subplots(figsize=(20, 8))

df["LIMIT_BAL_Q"] = pd.qcut(df["LIMIT_BAL"], 10)

df.groupby("LIMIT_BAL_Q")["DEFAULT"].mean().plot(ax=ax, marker="x")
ax.set(title="Default Rate by LIMIT_BAL", ylabel="Rate")

df.drop("LIMIT_BAL_Q", axis=1, inplace=True)
plt.show()


# ##### BILL_AMT
# 
# We don't appear to have outliers, but we do have negative bill amounts - maybe clients overpaid?

df[[f"BILL_AMT{n}" for n in range(1, 7)]].describe()


(df[[f"BILL_AMT{n}" for n in range(1, 7)]] < 0).mean()


(df[[f"BILL_AMT{n}" for n in range(1, 7)]] == 0).mean()


fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(20, 16))
fig.suptitle("Non-negative BILL_AMT Distribution", fontsize=24)

ax1.hist(np.log(1 + df[df["BILL_AMT1"] >= 0]["BILL_AMT1"]))
ax1.set(xlabel="Log(1 + BILL_AMT1)", ylabel="Count")

ax2.hist(np.log(1 + df[df["BILL_AMT2"] >= 0]["BILL_AMT2"]))
ax2.set(xlabel="Log(1 + BILL_AMT2)", ylabel="Count")

ax3.hist(np.log(1 + df[df["BILL_AMT3"] >= 0]["BILL_AMT3"]))
ax3.set(xlabel="Log(1 + BILL_AMT3)", ylabel="Count")

ax4.hist(np.log(1 + df[df["BILL_AMT4"] >= 0]["BILL_AMT4"]))
ax4.set(xlabel="Log(1 + BILL_AMT4)", ylabel="Count")

ax5.hist(np.log(1 + df[df["BILL_AMT5"] >= 0]["BILL_AMT5"]))
ax5.set(xlabel="Log(1 + BILL_AMT5)", ylabel="Count")

ax6.hist(np.log(1 + df[df["BILL_AMT6"] >= 0]["BILL_AMT6"]))
ax6.set(xlabel="Log(1 + BILL_AMT6)", ylabel="Count")

plt.show()


# Interestingly, it is clients with mid-range bill amounts which have the highest default rates.

fig, ax = plt.subplots(figsize=(20, 8))

for n in range(1, 7):
    df[f"BILL_AMT{n}_Q"] = pd.qcut(df[f"BILL_AMT{n}"], 5, labels=[f"Q{m}" for m in range(1, 6)])

for m in range(1, 6):
    ax.plot(
        range(1, 7), 
        [df[df[f"BILL_AMT{n}_Q"] == f"Q{m}"]["DEFAULT"].mean() for n in range(1, 7)], 
        marker="x",
        label=f"Q{m}"
    )
ax.set(title="Default Rate by BILL_AMT Quintile", xlabel="Month", ylabel="Rate")
ax.legend()

for n in range(1, 7):
    df.drop(f"BILL_AMT{n}_Q", axis=1, inplace=True)

plt.show()


# Quite correlated across time and, maybe surprisingly, not with *LIMIT_BAL*.

fig, ax = plt.subplots(figsize=(20, 4))
heatmap(
    corr_df[[f"BILL_AMT{n}" for n in range(1, 7)]].T, 
    vmax=1, 
    vmin=-1, 
    cmap=kindred_cmap, 
    linewidth=1, 
    ax=ax
)
plt.show()


# ##### PAY_AMT
# 
# We don't appear to have outliers.

df[[f"PAY_AMT{n}" for n in range(1, 7)]].describe()


fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(20, 16))
fig.suptitle("Non-negative PAY_AMT Distribution", fontsize=24)

ax1.hist(np.log(1 + df[df["PAY_AMT1"] >= 0]["PAY_AMT1"]))
ax1.set(xlabel="Log(1 + PAY_AMT1)", ylabel="Count")

ax2.hist(np.log(1 + df[df["PAY_AMT2"] >= 0]["PAY_AMT2"]))
ax2.set(xlabel="Log(1 + PAY_AMT2)", ylabel="Count")

ax3.hist(np.log(1 + df[df["PAY_AMT3"] >= 0]["PAY_AMT3"]))
ax3.set(xlabel="Log(1 + PAY_AMT3)", ylabel="Count")

ax4.hist(np.log(1 + df[df["PAY_AMT4"] >= 0]["PAY_AMT4"]))
ax4.set(xlabel="Log(1 + PAY_AMT4)", ylabel="Count")

ax5.hist(np.log(1 + df[df["PAY_AMT5"] >= 0]["PAY_AMT5"]))
ax5.set(xlabel="Log(1 + PAY_AMT5)", ylabel="Count")

ax6.hist(np.log(1 + df[df["PAY_AMT6"] >= 0]["PAY_AMT6"]))
ax6.set(xlabel="Log(1 + PAY_AMT6)", ylabel="Count")

plt.show()


# Interestingly, it is clients with mid-range bill amounts which have the highest default rates.

fig, ax = plt.subplots(figsize=(20, 8))

for n in range(1, 7):
    df[f"PAY_AMT{n}_Q"] = pd.qcut(df[f"PAY_AMT{n}"], 3, labels=[f"Q{m}" for m in range(1, 4)])

for m in range(1, 4):
    ax.plot(
        range(1, 7), 
        [df[df[f"PAY_AMT{n}_Q"] == f"Q{m}"]["DEFAULT"].mean() for n in range(1, 7)], 
        marker="x",
        label=f"Q{m}"
    )
ax.set(title="Default Rate by PAY_AMT Tercile", xlabel="Month", ylabel="Rate")
ax.legend()

for n in range(1, 7):
    df.drop(f"PAY_AMT{n}_Q", axis=1, inplace=True)

plt.show()


fig, ax = plt.subplots(figsize=(20, 4))
heatmap(
    corr_df[[f"PAY_AMT{n}" for n in range(1, 7)]].T, 
    vmax=1, 
    vmin=-1, 
    cmap=kindred_cmap, 
    linewidth=1, 
    ax=ax
)
plt.show()


# Seems that *PAY_AMTN* lags *BILL_AMTN* by 1.

corr_df.loc[[f"BILL_AMT{n}" for n in range(1, 7)], [f"PAY_AMT{n}" for n in range(1, 7)]].round(2)


# ## Can we reverse engineer PAY_N?
# 
# I just had a play around with the max_depth here.

X, y = df.drop(["DEFAULT"] + [f"PAY_{n}" for n in range(1, 7)], axis=1).values, df["PAY_1"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

dc = DecisionTreeClassifier(random_state=SEED, max_depth=7)
dc.fit(X_train, y_train)

np.mean(dc.predict(X_train) == y_train), np.mean(dc.predict(X_test) == y_test)


# ## Normalising
# 
# There is some correlation between *PAY_AMTN*, *BILL_AMTN* and *LIMIT_BAL*.
# Maybe we can normalize to engineer less correlated features.

corr_df.loc[
    ["LIMIT_BAL"] + [f"BILL_AMT{n}" for n in range(1, 7)], 
    ["LIMIT_BAL"] + [f"PAY_AMT{n}" for n in range(1, 7)]
].round(2)


for n in range(1, 7):
    df[f"BILL_AMT_NORM{n}"] = df[f"BILL_AMT{n}"] / df["LIMIT_BAL"]
for n in range(1, 6):
    df[f"PAY_AMT_NORM{n}"] = (1 + df[f"PAY_AMT{n}"]) / (1 + df[f"BILL_AMT{n+1}"])
    df[f"LAST_PAY_DIFF{n}"] = df[f"BILL_AMT{n+1}"] - df[f"PAY_AMT{n}"]


new_corr_df = df.corr("spearman")


# This actually looks worse for *BILL_AMT_NORMN*, but better for *PAY_AMTN*.

new_corr_df.loc[
    ["LIMIT_BAL"] + [f"BILL_AMT_NORM{n}" for n in range(1, 7)], 
    ["LIMIT_BAL"] + [f"PAY_AMT_NORM{n}" for n in range(1, 6)]
].round(2)


# # Conclusions

# * Target is imbalanced - choose appropriate metric/model.
# * Some of the categorical features have missing data.
# * Previous repayment statuses are the most correlated features with the target,
# but the meaning of the statuses is unclear. 
# We can't re-engineer them.
# * The amount of credit is anti-correlated with the target
# * Due to correlation between *PAY_AMTN* AND *BILL_AMTN* it may be worth normalizing *PAY_AMTN* by *BILL_AMTN*.
