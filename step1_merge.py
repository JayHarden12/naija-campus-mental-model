import pandas as pd

print("--- Phase 1: Loading Data ---")

# 1. Load the CSV file
df_csv = pd.read_csv('Dataset.csv')
# Standardize the column names so they are lowercase
df_csv = df_csv.rename(columns={'Context': 'context', 'Response': 'response'})
# Keep only the two columns we care about
df_csv = df_csv[['context', 'response']]
print(f"Successfully loaded {len(df_csv)} rows from Dataset.csv")

# 2. Load the JSON file 
# (Using lines=True because your JSON is formatted line-by-line)
df_json = pd.read_json('combined_dataset.json', lines=True)
# Standardize the column names here too
df_json = df_json.rename(columns={'Context': 'context', 'Response': 'response'})
df_json = df_json[['context', 'response']]
print(f"Successfully loaded {len(df_json)} rows from combined_dataset.json")

print("\n--- Phase 2: Merging and Cleaning ---")

# 3. Stack the two datasets on top of each other
df_master = pd.concat([df_csv, df_json], ignore_index=True)
initial_count = len(df_master)

# 4. Remove exact duplicate rows (keeping rows with same context but different response)
df_master = df_master.drop_duplicates(subset=['context', 'response'])
final_count = len(df_master)

print(f"Combined total: {initial_count} rows.")
print(f"Removed {initial_count - final_count} duplicate rows.")
print(f"Final clean dataset size: {final_count} unique rows.")

print("\n--- Phase 3: Preparing for AI Translation ---")

# 5. Create empty columns where the AI will eventually paste the translations
df_master['naija_pidgin'] = ""
df_master['naija_english'] = ""

# 6. Save this completed file to your workspace
df_master.to_csv('Master_Dataset.csv', index=False)
print("SUCCESS! Your data is merged and saved as 'Master_Dataset.csv'")