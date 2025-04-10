import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import optuna
import re
import os
import fitz  # PyMuPDF for PDF processing
import requests
import json

# Set page configuration
st.set_page_config(
    page_title="Legal Summarization System",
    page_icon="🎓",
    layout="wide"
)

# ========== LEGAL DOCUMENT SUMMARIZATION FUNCTIONS ==========

def sanitize_text(text):
    """Removes illegal characters from text."""
    return re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)

def extract_text_from_pdf(pdf_input):
    """
    Extracts and processes text from a PDF file.
    Accepts either a file path or a file-like object from Streamlit uploader.
    """
    try:
        if isinstance(pdf_input, str) and os.path.exists(pdf_input):
            doc = fitz.open(pdf_input)
        else:
            pdf_bytes = pdf_input.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        doc.close()
        return sanitize_text(text.strip())
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

@st.cache(allow_output_mutation=True)
def load_model_and_tokenizer(model_name):
    """Loads tokenizer and model efficiently."""
    st.write("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    if torch.cuda.is_available():
        model.to("cuda")
    st.write("Model loaded successfully.")
    return model, tokenizer

def generate_advanced_legal_prompt(case_type):
    """Generates structured prompts for concise summarization."""
    prompts = {
        "contract_dispute": "Summarize this contract dispute concisely: 1. Parties involved 2. Nature of the contract and alleged breach 3. Key legal arguments from both sides 4. Case status or resolution",
        "employment_law": "Summarize this employment law case concisely: 1. Employee/employer details 2. Nature of the dispute 3. Key claims by the employee 4. Key defenses by the employer 5. Case status",
        "criminal_law": "Summarize this criminal case concisely: 1. Defendant details and charges 2. Key evidence presented 3. Prosecution's main arguments 4. Defense counterarguments 5. Case status",
        "intellectual_property": "Summarize this intellectual property case concisely: 1. Parties involved 2. Nature of alleged infringement 3. Key legal arguments from both sides 4. Case status or ruling",
        "default_prompt": "Summarize this legal case concisely: 1. Key parties involved 2. Nature of the dispute 3. Main arguments from both sides 4. Current status or resolution"
    }
    return prompts.get(case_type.lower(), prompts["default_prompt"])

def generate_concise_summary(input_text, model, tokenizer, hyperparams, case_type):
    """Generates a structured, concise summary with optimized hyperparameters."""
    prompt = generate_advanced_legal_prompt(case_type)
    input_str = f"{prompt}\n\n### Document:\n{input_text[:4096]}\n\n### Summary:\n"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_inputs = tokenizer(
        input_str, return_tensors="pt", padding=True, truncation=True, max_length=4096
    ).to(device)
    
    with torch.no_grad():
        summary_output = model.generate(
            model_inputs.input_ids,
            max_new_tokens=hyperparams.get("max_new_tokens", 150),
            num_beams=hyperparams.get("num_beams", 5),
            temperature=hyperparams.get("temperature", 0.2),
            do_sample=hyperparams.get("do_sample", True),
            top_p=hyperparams.get("top_p", 0.85),
            no_repeat_ngram_size=3
        )
    full_output = tokenizer.decode(summary_output[0], skip_special_tokens=True)
    return sanitize_text(full_output.split("### Summary:")[-1].strip())

def objective(trial, input_text, model, tokenizer, case_type):
    """Objective function for hyperparameter tuning using Optuna."""
    summary = generate_concise_summary(input_text, model, tokenizer, {
        "max_new_tokens": trial.suggest_int("max_new_tokens", 100, 200),
        "num_beams": trial.suggest_int("num_beams", 4, 6),
        "temperature": trial.suggest_float("temperature", 0.1, 0.3),
        "do_sample": trial.suggest_categorical("do_sample", [True]),
        "top_p": trial.suggest_float("top_p", 0.8, 0.95)
    }, case_type)
    return len(summary)

# ========== LEGAL BOT FUNCTIONS ==========

API_KEY = "replace_with_your_api_key"  # Replace with your actual API key
API_URL = f"https://api.openai.com/v1/chat/completions"

def get_legal_bot_response(prompt):
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            response_json = response.json()
            if "candidates" in response_json:
                return response_json["candidates"][0]["content"]["parts"][0]["text"]
            else:
                return "No response from the Legal Bot."
        else:
            return "Failed to fetch response: " + response.text
    except Exception as e:
        return f"Error: {str(e)}"

# ========== MAIN APP ==========

st.title("Legal Summarization System")

# --- Legal Document Summarization Section ---
st.header("Document Summarization")

# Load model and tokenizer
model_name = "coderop12/Empowering_Legal_Summarization"
with st.spinner("Loading model and tokenizer..."):
    model, tokenizer = load_model_and_tokenizer(model_name)

# Select case type
case_types = {
    "Contract Dispute": "contract_dispute",
    "Employment Law": "employment_law",
    "Criminal Law": "criminal_law",
    "Intellectual Property": "intellectual_property",
    "Default": "default_prompt"  # Ensure the default prompt is selectable
}
selected_case = st.selectbox("Select Summarization Type", list(case_types.keys()), index=len(case_types)-1)  # Default prompt is the last option
case_type = case_types[selected_case]

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
    with st.spinner("Extracting text from PDF..."):
        input_text = extract_text_from_pdf(uploaded_file)
        st.session_state.input_text = input_text  # Store extracted text in session state
    if input_text:
        st.success("PDF text successfully extracted.")
        if 'study' not in st.session_state or 'input_text' not in st.session_state:
            st.write("Optimizing summarization parameters with 15 trials for conciseness...")
            with st.spinner("Optimizing parameters, please wait..."):
                study = optuna.create_study(direction="maximize")
                study.optimize(lambda trial: objective(trial, input_text, model, tokenizer, case_type), n_trials=15)
                st.session_state.study = study  # Store study in session state
        with st.spinner("Generating concise summary..."):
            concise_summary = generate_concise_summary(input_text, model, tokenizer, st.session_state.study.best_trial.params, case_type)
            st.session_state.concise_summary = concise_summary  # Store summary in session state
        st.subheader("Final Concise Summary")
        st.write(concise_summary)
    else:
        st.error("No text extracted from the PDF. Please check the file.")
else:
    if 'concise_summary' in st.session_state:
        st.subheader("Final Concise Summary")
        st.write(st.session_state.concise_summary)

st.markdown("---")

# --- Legal Bot Section ---
st.header("Legal Bot!")

# Manage chatbot state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history using Streamlit's chat components
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input for chat
user_input = st.chat_input("Type your message...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get Legal Bot response and display it
    response = get_legal_bot_response(user_input)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
