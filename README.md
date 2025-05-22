 Exploratory Data Analysis (EDA): topic
#code
# --- 2. Import Required Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Settings
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# --- 3. Load Dataset ---
file_path = "/mnt/data/healthcare_dataset.csv"
df = pd.read_csv(file_path)

# --- 4. Dataset Overview ---
print("✅ First 5 rows of the dataset:")
print(df.head())

print("\n📄 Dataset Info:")
df.info()

print("\n📊 Statistical Summary:")
print(df.describe(include='all'))

# --- 5. Missing Values ---
print("\n❓ Missing Values:")
print(df.isnull().sum())

# Visualize missing values
sns.heatmap(df.isnull(), cbar=False, cmap="magma")
plt.title("Missing Values Heatmap")
plt.show()

# --- 6. Data Types Breakdown ---
print("\n🔍 Data Types:")
print(df.dtypes.value_counts())

# --- 7. Numerical Features Analysis ---
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
if not numeric_cols.empty:
    print(f"\n📈 Numeric Features: {list(numeric_cols)}")
    df[numeric_cols].hist(bins=15, figsize=(15, 10), layout=(len(numeric_cols) // 3 + 1, 3))
    plt.suptitle("Distributions of Numerical Features", fontsize=16)
    plt.show()

    # Box plots for outlier detection
    for col in numeric_cols:
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot - {col}')
        plt.show()

    # Correlation matrix
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()
else:
    print("❌ No numeric columns found.")

# --- 8. Categorical Features Analysis ---
cat_cols = df.select_dtypes(include='object').columns
if not cat_cols.empty:
    print(f"\n📊 Categorical Features: {list(cat_cols)}")
    for col in cat_cols:
        if df[col].nunique() <= 20:
            sns.countplot(y=col, data=df, order=df[col].value_counts().index)
            plt.title(f"Count Plot - {col}")
            plt.tight_layout()
            plt.show()
else:
    print("❌ No categorical columns found.")

# --- 9. Unique Values Check ---
print("\n🔎 Unique values in categorical columns:")
for col in cat_cols:
    print(f"{col}: {df[col].nunique()} unique values")

# --- 10. Class Distribution (if classification) ---
if 'target' in df.columns:
    sns.countplot(x='target', data=df)
    plt.title('Target Variable Distribution')
    plt.show() 
