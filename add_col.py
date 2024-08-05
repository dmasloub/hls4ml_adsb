import pandas as pd

# Load the CSV data
file_path = 'updated_anomaly_circle_with_ecef.csv'  # Change this to your file path if needed
df = pd.read_csv(file_path)

if 'vertical_rate' not in df.columns:
    df['vertical_rate'] = None

# Fill missing values in the 'anomaly' column with 1
df['vertical_rate'].fillna(0, inplace=True)

# Save the updated DataFrame back to a CSV file
output_file_path = 'test_circle.csv'
df.to_csv(output_file_path, index=False)

print(f"Updated data saved to {output_file_path}")
