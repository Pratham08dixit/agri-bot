import os
import streamlit as st
import speech_recognition as sr
import google.generativeai as genai
from langdetect import detect
from deep_translator import GoogleTranslator
from gtts import gTTS
from io import BytesIO
from dotenv import load_dotenv, find_dotenv
import requests
import json
import re
from PIL import Image

# Load .env variables
load_dotenv(find_dotenv())

# Get API keys from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Make sure it's set in your .env file.")

# Configure Gemini API Key
genai.configure(api_key=GOOGLE_API_KEY)

# List of Indian languages and dialects
INDIAN_LANGUAGES = [
    "as", "bn", "bho", "brx", "doi", "en", "gu", "hi", "kn", "ks", "kok", 
    "mai", "ml", "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur"
]

LANGUAGE_NAMES = {
    "as": "Assamese",
    "bn": "Bengali",
    "bho": "Bhojpuri",
    "brx": "Bodo",
    "doi": "Dogri",
    "en": "English",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ks": "Kashmiri",
    "kok": "Konkani",
    "mai": "Maithili",
    "ml": "Malayalam",
    "mni": "Manipuri",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Odia",
    "pa": "Punjabi",
    "sa": "Sanskrit",
    "sat": "Santali",
    "sd": "Sindhi",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu"
}

# Mapping for Google Speech Recognition language codes
GOOGLE_SPEECH_LANG_MAP = {
    "as": "as-IN",
    "bn": "bn-IN",
    "bho": "bho-IN",
    "brx": "brx-IN",
    "doi": "doi-IN",
    "en": "en-US",
    "gu": "gu-IN",
    "hi": "hi-IN",
    "kn": "kn-IN",
    "ks": "ks-IN",
    "kok": "kok-IN",
    "mai": "mai-IN",
    "ml": "ml-IN",
    "mni": "mni-IN",
    "mr": "mr-IN",
    "ne": "ne-IN",
    "or": "or-IN",
    "pa": "pa-IN",
    "sa": "sa-IN",
    "sat": "sat-IN",
    "sd": "sd-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "ur": "ur-IN"
}

# Initialize chat history in session_state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize uploaded image in session_state (do not auto-analyze)
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

# ---------------------
# Utility Functions
# ---------------------

def translate_text(text, target_lang):
    try:
        detected_lang = detect(text)
        if detected_lang == target_lang:
            return text, detected_lang  # No translation needed
        translated_text = GoogleTranslator(source=detected_lang, target=target_lang).translate(text)
        return translated_text, detected_lang
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text, "unknown"

def text_to_speech(text, lang):
    try:
        tts = gTTS(text=text, lang=lang)
        audio_bytes = BytesIO()
        tts.write_to_fp(audio_bytes)
        return audio_bytes
    except Exception as e:
        st.error(f"TTS error: {e}")
        return None

# Modified get_speech_input to accept the selected language from the sidebar
def get_speech_input(selected_lang_code="en"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now.")
        try:
            audio = recognizer.listen(source, timeout=5)
            google_lang = GOOGLE_SPEECH_LANG_MAP.get(selected_lang_code, "en-US")
            query = recognizer.recognize_google(audio, language=google_lang)
            return query
        except sr.UnknownValueError:
            return "Sorry, could not understand your speech."
        except sr.RequestError:
            return "Could not request results. Please check your internet connection."

# Expanded list of agriculture-related keywords
FARMING_KEYWORDS = [
    "crop", "crops", "vegetables", "fruits", "farm", "farming", "cultivation", "farmer", "agriculture", "soil", "irrigation",
    "harvest", "seeds", "seed", "fertilizer", "pesticides", "pesticide", "weather", "yield", "drought", "insecticide", "insecticides",
    "livestock", "organic", "horticulture", "greenhouse", "germination", "plant", "plants", "mulching", "herb", "herbs", "shurb", "shurbs",
    "sowing", "planting", "harvesting", "plowing", "disease", "treatment", "fertilizers", "agro", "fruit", "vegetable", "season", "drought", "rain", "weather", "flora"
]

def is_agriculture_query(query):
    """
    Determines whether a query is agriculture-related using a hybrid approach.
    First, it checks for known agriculture-related keywords.
    If none are found, it uses Gemini (LLM) to classify the query.
    """
    query_lower = query.lower()
    # First check: using keywords
    if any(keyword in query_lower for keyword in FARMING_KEYWORDS):
        return True

    # Second check: LLM-based classification
    classification_prompt = (
        "Answer with only YES or NO. Is the following query related to farming, crops, soil, "
        "plant diseases, harvesting, fertilizers, or related agricultural topics? "
        f"Query: \"{query}\""
    )
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(classification_prompt)
    if response and "YES" in response.text.upper():
        return True
    return False

def get_gemini_response(query, chat_history):
    # Use LLM-based filtering to check if the query is agriculture-related
    if not is_agriculture_query(query):
        return "This bot is designed only for farmers and agriculture-related queries."
    
    # Build conversation context prompt with explicit instruction for a detailed answer.
    conversation_context = ""
    for user_query, bot_response in chat_history:
        conversation_context += f"User: {user_query}\nBot: {bot_response}\n"
    conversation_context += (
        f"User: {query}\n"
        "Bot (please provide a detailed and comprehensive explanation): "
    )
    
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(conversation_context)
    return response.text if response else "Sorry, I couldn't generate a response."

def clean_json_response(text):
    # Remove markdown code fences (e.g., ```json ... ```)
    cleaned = re.sub(r'^```json\s*', '', text)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    return cleaned.strip()

def analyze_plant_with_gemini(uploaded_image):
    """
    Uses Gemini Vision API to analyze the plant image and return a JSON response with keys:
    'species', 'disease', and 'treatment'.
    """
    try:
        # Load the image data into PIL.Image format
        image = Image.open(uploaded_image)

        # Gemini Vision model instance (using a flash model here)
        model = genai.GenerativeModel("gemini-2.0-flash")

        # Construct a prompt that instructs the API to output only valid JSON.
        prompt = (
            "Analyze this plant image and provide a JSON response with the following keys: "
            "'species' (plant species), 'disease' (any visible disease), and 'treatment' (suggested treatment). "
            "Return only valid JSON without any extra commentary."
        )

        # Send image to Gemini API for analysis
        response = model.generate_content([image, prompt])
        # Clean the response text by removing markdown formatting if present
        cleaned_text = clean_json_response(response.text)
        # Parse the cleaned text as JSON
        analysis_result = json.loads(cleaned_text)
        return analysis_result
    
    except json.JSONDecodeError as json_err:
        st.error(f"JSON parsing error: {json_err}. Raw response: {response.text}")
        return None

    except Exception as e:
        st.error(f"Error analyzing plant image with Gemini: {e}")
        return None

# ---------------------
# Streamlit UI for Chat
# ---------------------

st.title("🌿 Agri Bot (Multilingual) 🌿")
st.subheader("Ask your farming-related questions via voice or text.")

# Sidebar for Language Selection
st.sidebar.title("Language Selection")
selected_lang = st.sidebar.selectbox("Choose your preferred language:",
                                     options=INDIAN_LANGUAGES,
                                     format_func=lambda x: LANGUAGE_NAMES[x])

# User Input Mode for Chat
input_mode = st.radio("Choose Input Mode:", ["Voice", "Text"])

query = None
if input_mode == "Voice":
    if st.button("Record & Ask"):
        # Pass the selected language to get_speech_input
        query = get_speech_input(selected_lang)
elif input_mode == "Text":
    query = st.text_input("Type your question:")

# ---------------------
# File Uploader for Plant/Crop Image
# ---------------------
st.header("Upload Plant/Crop Image (Optional)")
uploaded_file = st.file_uploader("Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"], key="plant_image_uploader")
if uploaded_file is not None:
    st.session_state.uploaded_image = uploaded_file
    st.success("Image uploaded successfully")

# ---------------------
# Process User Query
# ---------------------
if query:
    # Display the original query in the language the user asked
    st.write(f"**You asked:** {query}")
    
    # Detect and translate query to English for processing
    translated_query, detected_lang = translate_text(query, "en")
    st.write(f"**Detected Language:** {LANGUAGE_NAMES.get(detected_lang, detected_lang).upper()}")
    
    # Determine if the query is about the uploaded image using image-related keywords.
    IMAGE_QUERY_KEYWORDS = ["plant", "disease", "treatment", "analyze", "image", "leaf", "stem", "fruit"]
    is_image_query = False
    if st.session_state.uploaded_image is not None:
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in IMAGE_QUERY_KEYWORDS):
            is_image_query = True

    # Process the query accordingly.
    if is_image_query:
        with st.spinner("Analyzing the uploaded image for query context..."):
            analysis_result = analyze_plant_with_gemini(st.session_state.uploaded_image)
        if analysis_result is None:
            response_text = "Image analysis failed. Please try again or ask a general query."
        else:
            # Build a prompt with the image analysis context and the user's query, asking for a detailed response.
            context_info = json.dumps(analysis_result, indent=2)
            prompt = (
                f"Based on the following plant image analysis:\n{context_info}\n"
                f"Answer the following query in a detailed and comprehensive manner: {translated_query}\n"
                "Provide a thorough explanation."
            )
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(prompt)
            response_text = response.text if response else "Sorry, I couldn't generate a response."
            # Optionally clear the uploaded image so subsequent queries are not influenced
            st.session_state.uploaded_image = None
    else:
        # Process as a general agricultural query.
        response_text = get_gemini_response(translated_query, st.session_state.chat_history)
    
    # Translate response back to user's selected language
    final_response, _ = translate_text(response_text, selected_lang)
    
    # Save conversation to chat history
    st.session_state.chat_history.append((query, final_response))
    
    st.write("### Response:")
    st.write(final_response)
    
    # Convert response to speech
    audio_bytes = text_to_speech(final_response, selected_lang)
    if audio_bytes:
        st.audio(audio_bytes.getvalue(), format="audio/mp3")
    else:
        st.error("Could not generate speech response.")

# Display chat history in the sidebar
st.sidebar.title("Chat History")
for user_query, bot_response in reversed(st.session_state.chat_history):
    st.sidebar.write(f"**You:** {user_query}")
    st.sidebar.write(f"**Bot:** {bot_response}")
    st.sidebar.write("---")
