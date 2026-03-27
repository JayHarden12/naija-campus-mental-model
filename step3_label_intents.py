import pandas as pd
import google.generativeai as genai
import time
import os

# 1. Setup your API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyD3Q3vYUyRDdASsPGLHf2FpuoK9EorwL8w"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
# Using gemini-3.1-pro-preview for advanced reasoning
model = genai.GenerativeModel('gemini-3.1-pro-preview')

# 2. Your exact 45-Intent Taxonomy
INTENTS = [
    "casual_greeting", "casual_goodbye", "bot_identity", "bot_capabilities", 
    "user_appreciation", "user_frustration", "seeking_coping_advice", 
    "academic_workload", "academic_failure", "exam_anxiety", 
    "project_thesis_stress", "lecturer_conflict", "concentration_focus", 
    "financial_stress", "food_insecurity", "housing_accommodation", 
    "relationship_heartbreak", "family_pressure", "family_conflict", 
    "family_loss_grief", "friendship_conflict", "loneliness_isolation", 
    "peer_pressure", "bullying_harassment", "general_anxiety", 
    "panic_attack", "sadness_depression", "lack_of_motivation", 
    "low_self_esteem", "anger_irritability", "fatigue_burnout", 
    "sleep_insomnia", "appetite_changes", "paranoia_trust_issues", 
    "crisis_suicide_ideation", "crisis_self_harm", "crisis_abuse_trauma", 
    "substance_abuse_addiction", "unplanned_pregnancy_scare", 
    "health_medical_anxiety", "future_career_anxiety", 
    "social_media_phone_addiction", "neurodivergence_adhd", 
    "spiritual_religious_guilt", "body_image_insecurities"
]

def classify_multiple_intents(text):
    """Asks Gemini to pick up to 3 intents, comma-separated."""
    prompt = f"""
    You are an expert clinical psychologist and data scientist.
    Read the following text from a Nigerian university student. 
    Classify the text into UP TO THREE of the most relevant intents from the list below.
    Order them from the most prominent issue to the least prominent issue.
    
    If NONE of the valid intents perfectly describe the text, you may generate ONE new, highly concise intent (using snake_case, max 3 words) that best fits the text. If you generate a new intent, YOU MUST prepend it with 'new:' (e.g., new:therapy_process_inquiry, general_anxiety).

    Valid Intents: {', '.join(INTENTS)}
    
    Text: "{text}"
    
    Respond strictly with a comma-separated list of the intent names (maximum 3). 
    Do not add any other words, explanations, or quotes.
    Example output: financial_stress, academic_failure, new:therapy_inquiry
    """
    try:
        response = model.generate_content(prompt)
        # Clean the output and split by commas
        predicted_string = response.text.strip().lower()
        
        # Split into a list and clean up whitespace
        predicted_list = [intent.strip() for intent in predicted_string.split(',')]
        
        valid_predictions = []
        for i in predicted_list:
            if i in INTENTS:
                valid_predictions.append(i)
            elif i.startswith("new:"):
                # Clean it up and add it
                new_intent = i.replace("new:", "").strip()
                if new_intent and new_intent not in valid_predictions:
                    valid_predictions.append(new_intent)
                    if new_intent not in INTENTS:
                        INTENTS.append(new_intent)
                        print(f"*** Discovered NEW Intent: {new_intent} ***")

        if valid_predictions:
            return ", ".join(valid_predictions[:3])
        else:
            return "bot_capabilities"
            
    except Exception as e:
        print(f"API Error: {e}")
        time.sleep(5) 
        return "bot_capabilities"

# ==========================================
# --- RUNNING THE MULTI-LABEL CLASSIFICATION ---
# ==========================================
if __name__ == "__main__":
    print("Loading Master Dataset...")
    if os.path.exists('Master_Dataset_Labeled.csv'):
        df = pd.read_csv('Master_Dataset_Labeled.csv')
        print("Resuming from Master_Dataset_Labeled.csv")
    else:
        df = pd.read_csv('Master_Dataset.csv')
    
    if 'intents' not in df.columns:
        df['intents'] = None

    print(f"Starting multi-label classification for {len(df)} rows...")
    
    for index, row in df.iterrows():
        if pd.notna(row['intents']):
            continue
            
        user_text = str(row['context'])
        
        if user_text.strip() == "" or user_text.lower() == "nan":
            continue
            
        predicted = classify_multiple_intents(user_text)
        df.at[index, 'intents'] = predicted
        
        print(f"Row {index+1} -> {predicted}")
        
        # Save immediately after every prediction so progress is never lost
        df.to_csv('Master_Dataset_Labeled.csv', index=False)
            
    print("\n✅ Multi-Labeling Complete! Ready for complex DistilBERT training.")