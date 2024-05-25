
# pip install pytesseract pillow

import pytesseract
from PIL import Image

def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

# Code to extract text from an image using Tesseract OCR:
import re

def extract_dates(text):
    date_pattern = r'\b\d{2}/\d{2}/\d{4}\b'  # Example date pattern (MM/DD/YYYY)
    dates = re.findall(date_pattern, text)
    return dates

def extract_information(text):
    dates = extract_dates(text)
    # Add more extraction logic as needed
    return {
        'dates': dates,
        # Add more extracted data here
    }

# Data Extraction: Process Extracted Text
#Assume we need to extract specific information, such as dates, names, or other entities
import re

def extract_dates(text):
    date_pattern = r'\b\d{2}/\d{2}/\d{4}\b'  # Example date pattern (MM/DD/YYYY)
    dates = re.findall(date_pattern, text)
    return dates

def extract_information(text):
    dates = extract_dates(text)
    # Add more extraction logic as needed
    return {
        'dates': dates,
        # Add more extracted data here
    }

# Machine Learning: Analyze Extracted Data
# Assume we use a pre-trained model to predict something based on the extracted data 
import joblib

# Load a pre-trained model (for example, a simple classifier)
model = joblib.load('path_to_model.pkl')

def analyze_data(extracted_data):
    # Prepare the data for the model
    # This step depends on your specific model and feature engineering
    input_data = prepare_input_data(extracted_data)
    prediction = model.predict(input_data)
    return prediction

def prepare_input_data(extracted_data):
    # Example feature preparation
    features = [len(extracted_data['dates'])]  # Simple example feature
    return [features]

# ChatGPT API: Interact with User

# pip install openai

import openai

openai.api_key = 'your_openai_api_key'

def chat_with_gpt(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",  # or any other engine
        prompt=prompt,
        max_tokens=100
    )
    return response.choices[0].text.strip()

# Example usage
user_prompt = "What can you tell me about the extracted data?"
gpt_response = chat_with_gpt(user_prompt)
print(gpt_response)

#Integrating Everything

def main(image_path):
    # Step 1: OCR
    text = extract_text_from_image(image_path)
    print("Extracted Text:", text)
    
    # Step 2: Data Extraction
    extracted_data = extract_information(text)
    print("Extracted Information:", extracted_data)
    
    # Step 3: ML Analysis
    prediction = analyze_data(extracted_data)
    print("ML Prediction:", prediction)
    
    # Step 4: ChatGPT Interaction
    gpt_prompt = f"The extracted data is: {extracted_data}. The prediction is: {prediction}. Can you provide more insights?"
    gpt_response = chat_with_gpt(gpt_prompt)
    print("ChatGPT Response:", gpt_response)

# Example usage
image_path = 'path_to_your_image.png'
main(image_path)




