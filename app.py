import json
import random
from fuzzywuzzy import process
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import streamlit as st

# Load intents JSON
with open("trip_planning_dataset.json", "r") as file:
    intents = json.load(file)["intents"]

# Initialize the transformer model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(intents))

# Create a classification pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Map intents to indices for training
intent_to_label = {intent["tag"]: i for i, intent in enumerate(intents)}
label_to_intent = {i: intent["tag"] for i, intent in enumerate(intents)}

# Intent detection function
def detect_intent(user_input):
    try:
        # Use transformers model for classification
        result = classifier(user_input)[0]
        confidence = result["score"]
        intent_tag = result["label"]

        # Check confidence threshold
        if confidence > 0.5:  # Adjust threshold as needed
            return intent_tag
    except:
        pass

    # Fallback to fuzzy matching
    intent_scores = []
    for intent in intents:
        for pattern in intent["patterns"]:
            score = process.extractOne(user_input, [pattern])[1]
            intent_scores.append((intent["tag"], score))

    # Get the intent with the highest score
    best_match = max(intent_scores, key=lambda x: x[1])
    if best_match[1] > 50:  # Set a threshold for confidence
        return best_match[0]
    return "unknown"  # Default fallback

# Get response based on intent
def get_response(intent_tag):
    for intent in intents:
        if intent["tag"] == intent_tag:
            return random.choice(intent["responses"])  # Randomize responses if multiple are available
    return "Sorry, I don't have an answer for that."

# Chatbot response
def chatbot_response(user_input):
    detected_intent = detect_intent(user_input)
    response = get_response(detected_intent)
    return response

# Streamlit app
st.title("Tourist Guide ChatBot")
st.write("Hi! I'm here to help with your trip planning. Ask me anything!")

# Input text from the user
user_input = st.text_input("You:", placeholder="Type your question here and press Enter")

# Display the response
if user_input:
    if user_input.lower() in ["exit", "quit", "bye", "thank you"]:
        st.write("Chatbot: Goodbye! Have a great trip!")
    else:
        response = chatbot_response(user_input)
        st.text_area("Chatbot:", value=response, height=120, max_chars=None)
