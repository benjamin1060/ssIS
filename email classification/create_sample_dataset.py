

import pandas as pd

df = pd.read_csv("https://media.githubusercontent.com/media/PTIT-Assignment-Projects/ai-svm-email-spam/refs/heads/main/dataset/combined_data.csv")
print(df.shape)

target_column = 'label'  # Replace with the actual name of your target column
proportions = df[target_column].value_counts(normalize=True)

# Calculate the number of samples for each class
a = [i for i in range(1000, 82000, 5000)]
for i in a:
  num_samples = i
  sample_counts = (proportions * num_samples).round().astype(int)

  # Sample the dataset
  sampled_df = pd.concat([
      df[df[target_column] == label].sample(n=count, random_state=42)
      for label, count in sample_counts.items()
  ])

  # Reset index if needed
  sampled_df.reset_index(drop=True, inplace=True)
  sampled_df.to_csv(f"dataset/sampled_dataset{i}.csv", index=False)
