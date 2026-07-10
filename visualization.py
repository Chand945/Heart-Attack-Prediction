#===========================================
# HEART ATTACK PREDICTION
# Data Visualization
#===========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("data/dataset.csv")

# Style
sns.set_style("whitegrid")

# -----------------------------------------
# 1. Heart Disease Count
# -----------------------------------------

plt.figure(figsize=(6,5))
sns.countplot(x="target", data=df)
plt.title("Heart Disease Count")
plt.xlabel("Target (0 = No Disease, 1 = Disease)")
plt.ylabel("Count")
plt.show()

# -----------------------------------------
# 2. Age Distribution
# -----------------------------------------

plt.figure(figsize=(8,5))
sns.histplot(df["age"], bins=15, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# -----------------------------------------
# 3. Gender Distribution
# -----------------------------------------

plt.figure(figsize=(6,5))
sns.countplot(x="sex", data=df)
plt.title("Gender Distribution")
plt.xlabel("Sex (0 = Female, 1 = Male)")
plt.ylabel("Count")
plt.show()

# -----------------------------------------
# 4. Chest Pain Type
# -----------------------------------------

plt.figure(figsize=(7,5))
sns.countplot(x="cp", data=df)
plt.title("Chest Pain Types")
plt.xlabel("Chest Pain Type")
plt.ylabel("Count")
plt.show()

# -----------------------------------------
# 5. Correlation Heatmap
# -----------------------------------------

plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()