import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import re
import os
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Dark Trace AI - Message Analysis",
    page_icon="🔍",
    layout="wide"
)

# Load model and resources
MODEL_PATH = Path('artifacts/improved_pipeline.joblib')
EMOJI_DICT_PATH = Path('artifacts/emoji_dict.csv')
SLANG_DICT_PATH = Path('artifacts/slang_dict.csv')

@st.cache_resource
def load_resources():
    model = joblib.load(MODEL_PATH)
    emoji_dict = pd.read_csv(EMOJI_DICT_PATH)
    slang_dict = pd.read_csv(SLANG_DICT_PATH)
    return model, emoji_dict, slang_dict

class EnhancedFeatureExtractorV2(BaseEstimator, TransformerMixin):
    def __init__(self, text_col='message_text'):
        self.text_col = text_col
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.Series):
            texts = X
        else:
            texts = X[self.text_col]
            
        n_samples = len(texts)
        
        # Initialize feature arrays
        emoji_features = np.zeros((n_samples, 5))
        context_features = np.zeros((n_samples, 8))
        behavioral_features = np.zeros((n_samples, 6))
        slang_features = np.zeros((n_samples, 5))
        style_features = np.zeros((n_samples, 5))
        
        # Load dictionaries
        _, emoji_dict, slang_dict = load_resources()
        
        emoji_pattern = re.compile(r'[\U0001F300-\U0001F9FF]')
        drug_emojis = {'🍁', '💊', '💉', '🔥', '💨', '🌿', '🌱', '🪴', '🍄'}
        money_emojis = {'💰', '💵', '💸', '🤑', '💲'}
        
        for i, text in enumerate(texts):
            text_lower = text.lower()
            words = text_lower.split()
            
            # Emoji Analysis
            emojis = emoji_pattern.findall(text)
            emoji_features[i, 0] = len(emojis)  # total emojis
            emoji_features[i, 1] = len(set(emojis))  # unique emojis
            
            # Suspicious combinations
            drug_emoji_count = sum(1 for e in emojis if e in drug_emojis)
            money_emoji_count = sum(1 for e in emojis if e in money_emojis)
            emoji_features[i, 2] = drug_emoji_count > 0 and money_emoji_count > 0
            
            # Emoji sequence patterns
            if len(emojis) >= 2:
                emoji_pairs = list(zip(emojis, emojis[1:]))
                suspicious_pairs = sum(1 for e1, e2 in emoji_pairs 
                                    if (e1 in drug_emojis and e2 in money_emojis) or 
                                       (e1 in money_emojis and e2 in drug_emojis))
                emoji_features[i, 3] = suspicious_pairs > 0
            
            # Emoji density
            emoji_features[i, 4] = len(emojis) / len(words) if words else 0
            
            # Context Analysis
            word_pairs = list(zip(words, words[1:])) if len(words) > 1 else []
            
            # Time patterns
            time_words = {'asap', 'quick', 'fast', 'rush', 'hurry', 'now', 'tonight', 'soon'}
            time_bigrams = {'right now', 'real quick', 'hit me', 'hmu', 'dm me'}
            context_features[i, 0] = any(w in time_words for w in words) or \
                                   any(f"{w1} {w2}" in time_bigrams for w1, w2 in word_pairs)
            
            # Location patterns
            location_indicators = {'spot', 'location', 'place', 'meet', 'behind', 'alley', 'corner', 'parking'}
            discreet_locations = {'back', 'behind', 'alley', 'low', 'key', 'quiet', 'private'}
            context_features[i, 2] = any(w in location_indicators for w in words)
            context_features[i, 3] = any(w in discreet_locations for w in words)
            
            # Price/Quantity patterns
            price_pattern = re.compile(r'(?:(?:\$\d+(?:k)?)|(?:\d+(?:k)?\s*(?:dollars?|bucks?))|(?:\d+(?:\.\d{2})?))') 
            quantity_pattern = re.compile(r'\d+\s*(?:g|gram|oz|ounce|lb|pound|k|kilo|piece|pack)')
            context_features[i, 4] = bool(price_pattern.search(text_lower))
            context_features[i, 5] = bool(quantity_pattern.search(text_lower))
            
            # Message length
            context_features[i, 7] = len(text) / 100.0  # normalized message length
            
            # Behavioral Analysis
            urgency_words = {'need', 'want', 'looking', 'asap', 'urgent', 'today', 'quick', 'fast'}
            discretion_words = {'quiet', 'private', 'secret', 'discrete', 'careful', 'safe', 'clean'}
            quality_words = {'good', 'pure', 'clean', 'best', 'quality', 'fire', 'premium', 'top'}
            transaction_words = {'have', 'got', 'available', 'supply', 'plug', 'connect', 'hook'}
            trust_words = {'legit', 'trusted', 'reliable', 'safe', 'guaranteed', 'genuine'}
            exclusive_words = {'limited', 'exclusive', 'special', 'only', 'few', 'left'}
            
            behavioral_features[i, 0] = sum(w in urgency_words for w in words) / len(words)
            behavioral_features[i, 1] = any(w in discretion_words for w in words)
            behavioral_features[i, 2] = any(w in quality_words for w in words)
            behavioral_features[i, 3] = any(w in transaction_words for w in words)
            behavioral_features[i, 4] = any(w in trust_words for w in words)
            behavioral_features[i, 5] = any(w in exclusive_words for w in words)
            
            # Slang Analysis
            slang_words = set(w.lower() for w in slang_dict['slang_term'])
            text_slang = [w for w in words if w in slang_words]
            
            slang_features[i, 0] = len(text_slang)  # total slang
            slang_features[i, 1] = len(set(text_slang))  # unique slang
            
            # Slang proximity
            if len(text_slang) > 1:
                word_positions = [j for j, w in enumerate(words) if w in slang_words]
                min_distance = min(word_positions[j+1] - word_positions[j] 
                                for j in range(len(word_positions)-1))
                slang_features[i, 2] = 1.0 / (1.0 + min_distance)
            
            # Slang categories
            if text_slang:
                categories = set(slang_dict[slang_dict['slang_term'].str.lower().isin(text_slang)]['drug_type'])
                slang_features[i, 3] = len(categories)
            
            # Slang density
            slang_features[i, 4] = len(text_slang) / len(words) if words else 0
            
            # Style Analysis
            style_features[i, 0] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
            style_features[i, 1] = text.count('?') / len(text) if text else 0
            style_features[i, 2] = text.count('!') / len(text) if text else 0
            style_features[i, 3] = sum(c.isdigit() for c in text) / len(text) if text else 0
            special_chars = set('@#$%*')
            style_features[i, 4] = sum(c in special_chars for c in text) / len(text) if text else 0
        
        return np.column_stack([
            emoji_features,
            context_features,
            behavioral_features,
            slang_features,
            style_features
        ])

def get_feature_details(text, feature_extractor):
    """Get detailed feature analysis for a message"""
    features = feature_extractor.transform(pd.Series([text]))
    
    details = {
        "Emoji Analysis": {
            "Total Emojis": features[0, 0],
            "Unique Emojis": features[0, 1],
            "Suspicious Combinations": "Yes" if features[0, 2] > 0 else "No",
            "Suspicious Sequences": "Yes" if features[0, 3] > 0 else "No",
            "Emoji Density": f"{features[0, 4]:.2%}"
        },
        "Context Analysis": {
            "Urgency Indicators": "Yes" if features[0, 5] > 0 else "No",
            "Location Indicators": "Yes" if features[0, 7] > 0 else "No",
            "Price Patterns": "Yes" if features[0, 9] > 0 else "No",
            "Quantity Patterns": "Yes" if features[0, 10] > 0 else "No"
        },
        "Behavioral Analysis": {
            "Urgency Score": f"{features[0, 13]:.2%}",
            "Discretion Indicators": "Yes" if features[0, 14] > 0 else "No",
            "Quality Terms": "Yes" if features[0, 15] > 0 else "No",
            "Transaction Patterns": "Yes" if features[0, 16] > 0 else "No"
        },
        "Slang Analysis": {
            "Total Slang Words": int(features[0, 19]),
            "Unique Slang Words": int(features[0, 20]),
            "Slang Density": f"{features[0, 23]:.2%}"
        }
    }
    
    return details

# App title and description
st.title("🔍 Dark Trace AI - Message Analysis")
st.markdown("""
This tool analyzes messages for potential suspicious content using advanced AI.
Enter a message below to get real-time analysis and risk assessment.
""")

# Sidebar with options
st.sidebar.title("Analysis Options")
threshold = st.sidebar.slider(
    "Risk Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Adjust the sensitivity of the risk detection"
)

show_details = st.sidebar.checkbox(
    "Show Detailed Analysis",
    value=True,
    help="Display detailed feature analysis"
)

# --- Model Version Display ---


def get_model_version_info():
    try:
        stat = os.stat(MODEL_PATH)
        mod_time = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        model, _, _ = load_resources()
        version = getattr(model, 'version', None)
        return mod_time, version
    except Exception:
        return None, None

mod_time, model_version = get_model_version_info()
st.sidebar.markdown('---')
st.sidebar.markdown('**Model Info:**')
if model_version:
    st.sidebar.markdown(f"- Version: `{model_version}`")
if mod_time:
    st.sidebar.markdown(f"- Last updated: `{mod_time}`")
# --- End Model Version Display ---

# Input area
message = st.text_area(
    "Enter message to analyze:",
    height=100,
    placeholder="Type or paste message here..."
)

# Initialize feature extractor and load model
feature_extractor = EnhancedFeatureExtractorV2()
model, _, _ = load_resources()

if message:
    # Create features
    features = feature_extractor.transform(pd.Series([message]))
    
    # Get prediction
    prediction = model.predict_proba(features)[0, 1]
    
    # Display result
    st.markdown("### Analysis Results")
    
    # Risk meter
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.progress(prediction)
        risk_level = "High" if prediction >= 0.7 else "Medium" if prediction >= 0.4 else "Low"
        st.markdown(f"**Risk Score:** {prediction:.2%} ({risk_level} Risk)")
    
    if show_details:
        st.markdown("### Detailed Analysis")
        details = get_feature_details(message, feature_extractor)
        
        # Display feature details in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Content Analysis")
            st.json(details["Context Analysis"])
            st.markdown("#### 🔄 Behavioral Patterns")
            st.json(details["Behavioral Analysis"])
            
        with col2:
            st.markdown("#### 😀 Emoji Analysis")
            st.json(details["Emoji Analysis"])
            st.markdown("#### 🗣 Slang Analysis")
            st.json(details["Slang Analysis"])
    
    # Warning for high-risk messages
    if prediction >= 0.7:
        st.warning("⚠️ This message shows strong indicators of suspicious content.")
    elif prediction >= 0.4:
        st.warning("⚠️ This message shows some indicators of suspicious content.")

# Footer
st.markdown("---")
st.markdown("*Note: This tool is for demonstration purposes only. Always verify results manually.*")
