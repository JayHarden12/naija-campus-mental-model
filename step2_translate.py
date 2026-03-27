import pandas as pd
import google.generativeai as genai
import time
import os

# 1. Setup the AI
genai.configure(api_key="AIzaSyD3Q3vYUyRDdASsPGLHf2FpuoK9EorwL8w")
# Switching to flash for a much higher free-tier daily quota (1500 RPM vs 50 for Pro)
#model = genai.GenerativeModel('gemini-2.5-flash')
model = genai.GenerativeModel('gemini-3.1-pro-preview')
# model = genai.GenerativeModel('gemini-3-pro-preview')
# model = genai.GenerativeModel('gemini-2.5-pro')
# model = genai.GenerativeModel('gemini-pro-latest')

# 2. Load the Master Dataset
file_name = 'Master_Dataset.csv'
df = pd.read_csv(file_name)

# 3. Define the Zero-Shot translation function
def translate_text(english_text):
    prompt = f"""
    You are an expert in Nigerian linguistics and Lagos youth culture. 
    Take this clinical English text about a mental health struggle and provide TWO localized translations.
    
    1. A version in heavy, authentic Nigerian Pidgin (Naija). Do NOT use formal BBC Pidgin. Use street-level vocabulary and somatic expressions where appropriate.
    2. A version in casual Nigerian texting English (how a university student types to a friend).
    
    Original Text: "{english_text}"
    
    Format EXACTLY like this with no extra words:
    Pidgin: [Your Pidgin translation]
    English: [Your Nigerian English translation]
    """
    
    # Do not catch the exception here so that the automation loop can handle the retry/wait logic.
    response = model.generate_content(prompt)
    text = response.text.strip()
    
    # Split the text into the two parts
    pidgin_part = text.split('English:')[0].replace('Pidgin:', '').strip()
    english_part = text.split('English:')[1].strip()
    
    return pidgin_part, english_part

print(f"Starting unrestricted translation process for {len(df)} rows...")

# 4. The Full Automation Loop (With smart retries)
for index, row in df.iterrows():
    
    # Resume feature: Only translate if the row is empty
    if pd.isna(row['naija_pidgin']) or str(row['naija_pidgin']).strip() == "":
        print(f"Translating row {index + 1} of {len(df)}...")
        
        success = False
        retries = 0
        while not success and retries < 5:
            try:
                pidgin, english = translate_text(row['context'])
                
                if pidgin and english:
                    df.at[index, 'naija_pidgin'] = pidgin
                    df.at[index, 'naija_english'] = english
                    
                    # Save progress immediately
                    df.to_csv(file_name, index=False)
                    success = True
                    # A small sleep to prevent hitting any per-minute limits rapidly
                    time.sleep(4.2)
            except Exception as e:
                error_msg = str(e)
                print(f"  -> AI Error: {error_msg}")
                if "429" in error_msg or "quota" in error_msg.lower():
                    print("  -> Rate limit (429) hit. Sleeping for 45 seconds before retrying...")
                    time.sleep(45)
                    retries += 1
                elif "504" in error_msg or "deadline" in error_msg.lower():
                    print("  -> Timeout (504). Sleeping for 15 seconds before retrying...")
                    time.sleep(15)
                    retries += 1
                else:
                    # For other types of errors, back off and retry slightly or exit if severe
                    print("  -> Unhandled error. Sleeping for 5s then retrying...")
                    time.sleep(5)
                    retries += 1

print("\nFull dataset translation is complete!")