import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import pickle
import random
import pandas as pd
import re
import difflib

# ==========================================
# --- 1. INITIALIZE THE AI BRAIN & DATA ---
# ==========================================
print("Waking up the DistilBERT AI...")
MODEL_PATH = './naija_campus_model'

try:
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    with open(f'{MODEL_PATH}/mlb_classes.pkl', 'rb') as f:
        mlb_classes = pickle.load(f)
    print("✅ AI Brain loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

try:
    # Load the response bank once at startup to save memory
    df_bank = pd.read_csv('response_bank.csv')
    df_bank.columns = df_bank.columns.str.strip()
    df_bank['intent'] = df_bank['intent'].str.strip()
    print("✅ Response Bank loaded successfully!")
except FileNotFoundError:
    print("❌ Warning: response_bank.csv not found.")
    df_bank = pd.DataFrame(columns=['intent', 'standard_english_response', 'naija_hybrid_response'])

# ==========================================
# --- 2. THE TONE DETECTOR (The Mouth) ---
# ==========================================
def detect_user_tone(user_message):
    """Scans the user's message to automatically determine the appropriate tone."""
    message_lower = user_message.lower()
    naija_markers = [
        r'\bdey\b', r'\bna\b', r'\bomo\b', r'\bwetin\b', r'\babi\b', 
        r'\bwahala\b', r'\bshege\b', r'\bfit\b', r'\babeg\b', r'\bsef\b', 
        r'\bsha\b', r'\bwan\b', r'\bsapa\b', r'\bcast\b', r'\bginger\b',
        r'how far', r'\btire\b', r'\bshey\b', r'\bmake\b', r'\bno\b', 
        r'\be\b', r'\bsay\b', r'\bpikin\b', r'\bsabi\b', r'\byarn\b', 
        r'\bchoke\b', r'\bgbas\b', r'\bgbos\b', r'\bkuku\b', r'\bjapa\b', 
        r'\bgbege\b', r'\bpalava\b', r'\bwaka\b', r'\bdem\b', r'\buna\b', 
        r'\bwey\b', r'\bkain\b', r'\btaya\b', r'\bno be\b', r'e don', 
        r'na so', r'no dey', r'\bmymind\b', r'\bbody\b', r'\bguy\b', 
        r'\booh\b', r'\bahn\b', r'\bchai\b', r'\beish\b', r'\bgat\b', 
        r'\bam\b', r'\bcomot\b', r'\bchop\b', r'\bmatter\b', r'\bfall\b',
        r'wan die', r'my head', r'\bgist\b', r'\bnow\b', r'\blamba\b', 
        r'\byawa\b', r'\bhafa\b'
    ]
    
    for marker in naija_markers:
        if re.search(marker, message_lower):
            return "naija_hybrid"
            
    return "standard_english"

# ==========================================
# --- 3. THE TYPO-TOLERANT ROUTER ---
# ==========================================
def is_fuzzy_match(word, word_list, cutoff=0.85):
    """Checks if a user's word is a typo of our keywords (e.g., 'hwfa' matches 'hafa')"""
    matches = difflib.get_close_matches(word, word_list, n=1, cutoff=cutoff)
    return len(matches) > 0

# ==========================================
# --- 4. THE DISTILBERT PREDICTOR (The Brain) ---
# ==========================================
def predict_intents(text, threshold=0.5):
    
    # --- THE ROUTER (The Bouncer & The Paramedic) ---
    clean_text = re.sub(r'[^\w\s]', '', text.lower().strip())
    words = clean_text.split() # Break sentence into individual words
    
    greetings = ['hi', 'hello', 'hafa', 'how far', 'good morning', 'good afternoon', 'good evening', 'good day', 'hey', 'sup', 'xup', 'how bodi']
    goodbyes = ['bye', 'goodbye', 'later', 'goodnight', 'we go see']
    affirmation_words = ['sharp', 'ok', 'okay', 'alright', 'cool', 'nice', 'thanks', 'sure', 'normal', 'mad', 'observe', 'nothing', 'gist', 'vent']
    affirmation_phrases = ['i dey', 'i dey o', 'no shaking', 'just dey', 'we dey', 'thank you', 'thank you o', 'you do well', 'you do well o']

    # 1. PARAMEDIC
    strict_crisis = ['kill myself', 'die', 'suicide', 'sniper', 'take my life', 'tired of living', 'my life']
    ambiguous_crisis = ['end am', 'end it']
    safe_contexts = ['chat', 'game', 'conversation', 'relationship', 'call', 'class', 'school', 'movie']

    is_crisis = any(risk in clean_text for risk in strict_crisis)
    
    if not is_crisis:
        for amb in ambiguous_crisis:
            if amb in clean_text:
                if not any(safe in clean_text for safe in safe_contexts):
                    return ["clarify_ambiguous_crisis"] 
                    
    if is_crisis:
        return ["crisis_suicide_ideation"]

   # 2. FINANCIAL ROUTER 
    financial_keywords = ['job', 'money', 'sapa', 'broke', 'school fees', 'cash', 'work', 'hustle']
    if any(word in words for word in financial_keywords) or "find job" in clean_text:
        return ["financial_stress"]

   # 3. FUZZY MATCH FOR GREETINGS & SMALL TALK (Catches typos!)
    if len(words) <= 7:  
        if any(is_fuzzy_match(w, greetings) for w in words) or "how far" in clean_text or "how bodi" in clean_text:
            return ["casual_greeting"]
        if any(is_fuzzy_match(w, goodbyes) for w in words) or "we go see" in clean_text or ("chat" in clean_text and len(words) <= 3):
            return ["casual_goodbye"]
        if any(is_fuzzy_match(w, affirmation_words) for w in words) or any(phrase == clean_text for phrase in affirmation_phrases):
            return ["conversational_affirmation"]

    # --- WAKE UP DISTILBERT ---
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits).squeeze().numpy()
    
    import numpy as np
    if probs.ndim == 0:
        probs = np.expand_dims(probs, axis=0)
        
    predicted_indices = np.where(probs > threshold)[0]
    if len(predicted_indices) == 0:
        return [] 
        
    predicted_intents = [mlb_classes[i] for i in predicted_indices]
    predicted_intents = sorted(predicted_intents, key=lambda i: probs[list(mlb_classes).index(i)], reverse=True)
    return predicted_intents

# ==========================================
# --- 5. THE PERSONALIZED RESPONSE GENERATOR ---
# ==========================================
def get_chatbot_response(primary_intent, user_message, user_name="My G", course_code=None):
    detected_tone = detect_user_tone(user_message)
    
    if primary_intent == "conversational_affirmation":
        if detected_tone == "naija_hybrid":
            return f"We thank God. So {user_name}, wetin dey happen? Any school stress or life wahala wey you wan offload today?"
        return f"That's good to hear, {user_name}. What's on your mind today? I'm here if you want to talk."

    if primary_intent == "out_of_domain":
        if detected_tone == "naija_hybrid":
            return f"{user_name}, my brain mainly dey focus on school stress and mental wahala. Wetin dey your mind today?"
        return f"I'm primarily trained to help with academic stress, {user_name}. Is there anything on your mind you'd like to share?"

    if primary_intent == "clarify_ambiguous_crisis":
        if detected_tone == "naija_hybrid":
            return f"Abeg {user_name}, I want make sure I understand you well. When you say 'end am', you mean the chat, or things don heavy for your chest?"
        return f"{user_name}, I want to make sure I understand. Do you mean ending this conversation, or are you having thoughts of ending your life?"

    matching_rows = df_bank[df_bank['intent'] == primary_intent]
    if matching_rows.empty:
        return f"I can tell you are dealing with {primary_intent}, but I'm still learning how to respond. Can you tell me more?"
        
    responses = matching_rows['naija_hybrid_response'].tolist() if detected_tone == "naija_hybrid" else matching_rows['standard_english_response'].tolist()
    clean_responses = [str(r) for r in responses if pd.notna(r)]
    
    reply = random.choice(clean_responses) if clean_responses else "I am here for you."
    
    # 🌟 MAGIC 1: INJECT THE USER'S NAME
    reply = reply.replace("My G,", f"{user_name},").replace("My friend,", f"{user_name},").replace("My guy,", f"{user_name},").replace("my guy", user_name)
    
    # 🌟 MAGIC 2: INJECT THE COURSE CODE (Named Entity Recognition)
    if course_code:
        if detected_tone == "naija_hybrid":
            reply = f"Ah, this {course_code} matter! " + reply
        else:
            reply = f"I see you are stressed about {course_code}. " + reply

    return reply

# ==========================================
# --- 6. LIVE INTERACTIVE TERMINAL CHAT ---
# ==========================================
if __name__ == "__main__":
    print("\n=======================================================")
    print("  NAIJA CAMPUS BOT - AI TERMINAL (PRESS CTRL+C TO QUIT) ")
    print("=======================================================\n")
    
    # 🌟 MAGIC 3: USER ONBOARDING (The Personalization Gap)
    print("Bot: How far? Welcome to Naija Campus Bot. Wetin I fit call you? (Enter your name or nickname)")
    user_name = input("You: ").strip()
    if not user_name: user_name = "Boss" # Fallback if they hit enter
    print(f"\nBot: Sharp! Nice to meet you, {user_name}. If you wan vent or just gist, I dey listen.\n")
    
    current_conversation_intent = None
    risk_counter = 0  # 🌟 MAGIC 4: THE RISK TRACKER
    high_risk_intents = ['sadness_depression', 'general_anxiety', 'panic_attack', 'loneliness_isolation', 'fatigue_burnout']
    
    while True:
        try:
            user_input = input("You: ")
            
            if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                print(f"\nBot: Alright {user_name}, we go talk later. Make you take care of yourself out there.")
                break
                
            # EXTRACT COURSE CODE (e.g., CSC401, MTH 101)
            course_match = re.search(r'\b[a-zA-Z]{3}\s?\d{3}\b', user_input)
            course_code = course_match.group(0).upper() if course_match else None
                
            detected_intents = predict_intents(user_input)
            
            if detected_intents:
                primary_intent = detected_intents[0]
                ignore_states = ["out_of_domain", "casual_greeting", "casual_goodbye", "clarify_ambiguous_crisis", "conversational_affirmation"]
                if primary_intent in ignore_states:
                    current_conversation_intent = None  
                else:
                    current_conversation_intent = primary_intent 
            else:
                primary_intent = current_conversation_intent or "out_of_domain"

            print(f"  [DEBUG - Tone: {detect_user_tone(user_input)} | Intent: {primary_intent} | Course: {course_code}]")
            
            # RISK ESCALATION LOGIC
            if primary_intent in high_risk_intents:
                risk_counter += 1
            
            if risk_counter >= 3:
                print(f"Bot: {user_name}, I don notice say this matter dey weigh you down heavily. As an AI, I really think you should talk to a human counselor on campus. You deserve to feel better. 💙\n")
                risk_counter = 0 # Reset tracker after escalating
            else:
                bot_reply = get_chatbot_response(primary_intent, user_input, user_name, course_code)
                print(f"Bot: {bot_reply}\n")
            
        except KeyboardInterrupt:
            print(f"\n\nBot: Catch you later, {user_name}. Remember to drink water and take am easy.")
            break