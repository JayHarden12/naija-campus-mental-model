import pandas as pd
import re

print("Starting CSV Cleanup...")

# 1. Load the CSV
df = pd.read_csv('response_bank.csv')

# Strip any whitespace from column names
df.columns = df.columns.str.strip()

# 2. Fix the Structural Corruption (Drop Unnamed columns)
cols_to_keep = ['intent', 'naija_hybrid_response', 'standard_english_response']
df = df[cols_to_keep]

# Strip whitespace from intents
df['intent'] = df['intent'].str.strip()

# 3. Clean the "user_frustration" Bucket
# Move Sapa/Money responses to financial_stress
sapa_keywords = ['sapa', 'broke']
sapa_mask = df['intent'] == 'user_frustration'
for idx, row in df[sapa_mask].iterrows():
    if any(keyword in str(row['naija_hybrid_response']).lower() or keyword in str(row['standard_english_response']).lower() for keyword in sapa_keywords):
        df.at[idx, 'intent'] = 'financial_stress'

# Move Interpersonal Drama to friendship_conflict
drama_keywords = ['drama', 'people']
drama_mask = df['intent'] == 'user_frustration'
for idx, row in df[drama_mask].iterrows():
    if any(keyword in str(row['naija_hybrid_response']).lower() or keyword in str(row['standard_english_response']).lower() for keyword in drama_keywords):
        df.at[idx, 'intent'] = 'friendship_conflict'

# 4. Standardize the MANI Hotline for Suicide Ideation
mani_number = "0809 111 6264"
robotic_phrase = r"Abeg call the suicide hotline for \d+ right now\."

def fix_suicide_responses(text):
    if pd.isna(text): return text
    text = str(text)
    
    # Remove the robotic appended phrase
    text = re.sub(robotic_phrase, "", text).strip()
    
    # Replace the incorrect NSPI number (08092106493) with the correct MANI number
    text = text.replace("08092106493", mani_number)
    
    # If a suicide response somehow doesn't have the MANI number, gracefully add it
    if "crisis_suicide_ideation" in df['intent'].values and mani_number.replace(" ", "") not in text.replace(" ", "") and "112" not in text:
        text = text + f" Please reach out to MANI at {mani_number} for immediate support."
        
    return text

suicide_mask = df['intent'] == 'crisis_suicide_ideation'
df.loc[suicide_mask, 'naija_hybrid_response'] = df.loc[suicide_mask, 'naija_hybrid_response'].apply(fix_suicide_responses)
df.loc[suicide_mask, 'standard_english_response'] = df.loc[suicide_mask, 'standard_english_response'].apply(fix_suicide_responses)

# 5. Save the cleaned dataset
df.to_csv('response_bank_cleaned.csv', index=False)
print("✅ CSV cleaned and saved as 'response_bank_cleaned.csv'")