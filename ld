import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned SpaceX dataset
df = pd.read_csv("../data/spacex_cleaned_data.csv")

# Create the scatter plot
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Flight Number",
    y="Launch Site",
    hue="Class",
    s=100
)

plt.title("Flight Number vs. Launch Site")
plt.xlabel("Flight Number")
plt.ylabel("Launch Site")
plt.legend(
    title="Launch Outcome",
    labels=["Unsuccessful", "Successful"]
)

plt.tight_layout()
plt.show()
