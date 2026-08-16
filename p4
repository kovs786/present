import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned SpaceX dataset
df = pd.read_csv("spacex_cleaned_data.csv")

# Display basic information
print("Dataset shape:", df.shape)
display(df.head())

# Dataset information
print("Dataset Information:")
df.info()

# Summary statistics
print("\nSummary Statistics:")
display(df.describe())

# Check missing values
print("\nMissing Values:")
display(df.isnull().sum())

# Convert date column to datetime
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Extract year
df["Year"] = df["Date"].dt.year

# Count launches by year
launches_by_year = df.groupby("Year").size()

# Plot
plt.figure(figsize=(10, 5))
plt.plot(launches_by_year.index, launches_by_year.values, marker="o")
plt.title("SpaceX Launches by Year")
plt.xlabel("Year")
plt.ylabel("Number of Launches")
plt.xticks(launches_by_year.index, rotation=45)
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))

launch_site_counts = df["Launch Site"].value_counts()

sns.barplot(
    x=launch_site_counts.values,
    y=launch_site_counts.index
)

plt.title("Number of SpaceX Launches by Launch Site")
plt.xlabel("Number of Launches")
plt.ylabel("Launch Site")
plt.show()

success_counts = df["Class"].value_counts()

labels = ["Successful", "Unsuccessful"]

plt.figure(figsize=(7, 7))
plt.pie(
    success_counts.values,
    labels=labels[:len(success_counts)],
    autopct="%1.1f%%",
    startangle=90
)

plt.title("SpaceX Launch Success Rate")
plt.show()

plt.figure(figsize=(10, 6))

plt.hist(
    df["Payload Mass (kg)"].dropna(),
    bins=20
)

plt.title("Distribution of Payload Mass")
plt.xlabel("Payload Mass (kg)")
plt.ylabel("Number of Launches")
plt.show()

plt.figure(figsize=(8, 6))

sns.boxplot(
    x="Class",
    y="Payload Mass (kg)",
    data=df
)

plt.title("Payload Mass by Launch Outcome")
plt.xlabel("Launch Outcome")
plt.ylabel("Payload Mass (kg)")
plt.xticks([0, 1], ["Unsuccessful", "Successful"])
plt.show()

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Payload Mass (kg)",
    y="Flight Number",
    hue="Class"
)

plt.title("Payload Mass and SpaceX Launch Outcome")
plt.xlabel("Payload Mass (kg)")
plt.ylabel("Flight Number")
plt.show()

launch_outcome = pd.crosstab(
    df["Launch Site"],
    df["Class"]
)

launch_outcome.plot(
    kind="bar",
    stacked=True,
    figsize=(12, 6)
)

plt.title("Launch Outcomes by Launch Site")
plt.xlabel("Launch Site")
plt.ylabel("Number of Launches")
plt.xticks(rotation=45)
plt.legend(["Unsuccessful", "Successful"], title="Outcome")
plt.show()

# Select numerical variables
numeric_df = df.select_dtypes(include=["number"])

# Calculate correlations
correlation_matrix = numeric_df.corr()

# Plot heatmap
plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap of SpaceX Numerical Variables")
plt.show()

