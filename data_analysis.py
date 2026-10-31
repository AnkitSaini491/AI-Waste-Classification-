
import os
import pandas as pd
import matplotlib.pyplot as plt

# Dataset path
dataset_path = "dataset/train"

# Waste categories
categories = [
    "cardboard",
    "glass",
    "metal",
    "organic",
    "paper",
    "plastic"
]

# Count images in each category
data = []

for category in categories:
    folder = os.path.join(dataset_path, category)

    if os.path.exists(folder):
        count = len([
            file for file in os.listdir(folder)
            if file.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
    else:
        count = 0

    data.append({
        "Waste Category": category.capitalize(),
        "Image Count": count
    })

# Create DataFrame
df = pd.DataFrame(data)

# Display analysis
print("\nWaste Dataset Analysis")
print(df)

print("\nTotal Images:", df["Image Count"].sum())

print("\nCategory Statistics:")
print(df.describe())

# Bar Chart
plt.figure(figsize=(9, 5))
plt.bar(df["Waste Category"], df["Image Count"])
plt.title("Waste Category Distribution")
plt.xlabel("Waste Categories")
plt.ylabel("Number of Images")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# Pie Chart
plt.figure(figsize=(7, 7))
plt.pie(
    df["Image Count"],
    labels=df["Waste Category"],
    autopct="%1.1f%%"
)
plt.title("Waste Dataset Percentage")
plt.show()

# Save analysis
df.to_csv("waste_analysis.csv", index=False)

print("\nAnalysis saved successfully!")

