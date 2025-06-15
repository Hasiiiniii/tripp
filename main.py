import streamlit as st
import requests
import json
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd
import time
import os

# Configuration
st.set_page_config(
    page_title="TripCraft AI - Personalized Travel Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for travel-themed styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        border-left: 4px solid #ff6b9d;
        margin-left: 2rem;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-left: 4px solid #00d4aa;
        margin-right: 2rem;
    }
    
    .itinerary-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .budget-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #ff8c42;
    }
    
    .demographic-tag {
        background: #667eea;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
    
    .day-header {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        padding: 0.8rem;
        border-radius: 8px;
        font-weight: bold;
        margin: 1rem 0 0.5rem 0;
    }
    
    .activity-item {
        background: #f8f9ff;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border-left: 3px solid #667eea;
    }
    
    .model-info {
        background: #f0f2f6;
        padding: 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
        margin: 0.5rem 0;
    }
    
    .error-message {
        background: #ffe6e6;
        color: #d63384;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #d63384;
        margin: 1rem 0;
    }
    
    .success-message {
        background: #e6ffe6;
        color: #198754;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #198754;
        margin: 1rem 0;
    }
    
    .model-status {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-size: 0.8rem;
    }
    
    .model-available {
        background: #e6ffe6;
        color: #198754;
        border-left: 3px solid #198754;
    }
    
    .model-loading {
        background: #fff3cd;
        color: #664d03;
        border-left: 3px solid #ffca2c;
    }
    
    .model-error {
        background: #ffe6e6;
        color: #d63384;
        border-left: 3px solid #d63384;
    }
</style>
""", unsafe_allow_html=True)

# Updated Qwen models - using only verified working models
QWEN_TRAVEL_MODELS = {
    "Qwen2.5-7B-Instruct": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "description": "Powerful instruction-following model for complex travel planning",
        "features": ["Deep Reasoning", "Complex Planning", "Cultural Insights"],
        "recommended": True,
        "max_tokens": 800,
        "timeout": 60
    },
    "Qwen2.5-3B-Instruct": {
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "description": "Balanced model for efficient travel planning",
        "features": ["Fast", "Efficient", "Good Balance"],
        "recommended": True,
        "max_tokens": 600,
        "timeout": 45
    },
    "Qwen2.5-1.5B-Instruct": {
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "description": "Lightweight model for quick travel suggestions",
        "features": ["Very Fast", "Basic Planning", "Resource Efficient"],
        "recommended": False,
        "max_tokens": 400,
        "timeout": 30
    },
    "Qwen2.5-0.5B-Instruct": {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "description": "Ultra-fast for quick travel tips",
        "features": ["Ultra Fast", "Quick Tips", "Minimal Resources"],
        "recommended": False,
        "max_tokens": 300,
        "timeout": 20
    },
    "Qwen2.5-Coder-7B-Instruct": {
        "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "description": "Code-focused model adapted for structured travel planning",
        "features": ["Structured Output", "Detailed Planning", "Logical Flow"],
        "recommended": False,
        "max_tokens": 700,
        "timeout": 50
    }
}

# Demographic categories with specific preferences
DEMOGRAPHICS = {
    "Youth (18-25)": {
        "preferences": ["adventure", "nightlife", "budget-friendly", "social", "backpacking", "hostels"],
        "budget_range": (500, 2000),
        "activity_style": "active_social"
    },
    "Young Professionals (26-35)": {
        "preferences": ["cultural", "food", "moderate_luxury", "networking", "boutique_hotels", "experiences"],
        "budget_range": (1000, 4000),
        "activity_style": "balanced_exploration"
    },
    "Families": {
        "preferences": ["kid-friendly", "educational", "safe", "comfortable", "family_resorts", "activities"],
        "budget_range": (1500, 5000),
        "activity_style": "family_oriented"
    },
    "Couples": {
        "preferences": ["romantic", "intimate", "scenic", "luxury", "fine_dining", "spas"],
        "budget_range": (1200, 6000),
        "activity_style": "romantic_relaxed"
    },
    "Seniors (55+)": {
        "preferences": ["comfortable", "cultural", "guided_tours", "accessible", "heritage", "cruise"],
        "budget_range": (2000, 8000),
        "activity_style": "comfortable_cultural"
    },
    "Adventure Seekers": {
        "preferences": ["extreme_sports", "outdoor", "wilderness", "challenging", "unique", "eco_lodges"],
        "budget_range": (800, 4000),
        "activity_style": "high_adventure"
    },
    "Digital Nomads": {
        "preferences": ["wifi", "coworking", "long_stay", "affordable", "community", "flexible"],
        "budget_range": (800, 3000),
        "activity_style": "work_travel_balance"
    }
}

def get_hf_token():
    """Get Hugging Face token from secrets or environment"""
    try:
        # Try Streamlit secrets first
        if hasattr(st, 'secrets') and 'HUGGINGFACE_TOKEN' in st.secrets:
            return st.secrets['HUGGINGFACE_TOKEN']
        # Fall back to environment variable
        elif 'HUGGINGFACE_TOKEN' in os.environ:
            return os.environ['HUGGINGFACE_TOKEN']
        else:
            return None
    except Exception as e:
        st.error(f"Error accessing secrets: {e}")
        return None

def check_model_availability(model_id: str, hf_token: str) -> Dict[str, any]:
    """Check if a model is available and ready"""
    headers = {"Authorization": f"Bearer {hf_token}"}
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    
    try:
        # Simple test query
        payload = {
            "inputs": "Hello, are you ready?",
            "parameters": {
                "max_new_tokens": 10,
                "temperature": 0.7
            }
        }
        
        response = requests.post(api_url, headers=headers, json=payload, timeout=15)
        
        if response.status_code == 200:
            return {"status": "available", "message": "Model ready", "response_time": "< 15s"}
        elif response.status_code == 503:
            try:
                error_data = response.json()
                if "loading" in str(error_data).lower():
                    estimated_time = error_data.get("estimated_time", 20)
                    return {"status": "loading", "message": f"Model loading (ETA: {estimated_time}s)", "estimated_time": estimated_time}
                else:
                    return {"status": "error", "message": "Service temporarily unavailable"}
            except:
                return {"status": "loading", "message": "Model is loading, please wait"}
        elif response.status_code == 401:
            return {"status": "error", "message": "Invalid Hugging Face token"}
        elif response.status_code == 404:
            return {"status": "error", "message": "Model not found"}
        elif response.status_code == 429:
            return {"status": "error", "message": "Rate limit exceeded"}
        else:
            return {"status": "error", "message": f"HTTP {response.status_code}"}
            
    except requests.exceptions.Timeout:
        return {"status": "error", "message": "Connection timeout (model may be slow)"}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"Connection error: {str(e)[:50]}"}
    except Exception as e:
        return {"status": "error", "message": f"Unexpected error: {str(e)[:50]}"}

def init_session_state():
    """Initialize session state for TripCraft AI"""
    defaults = {
        "messages": [],
        "current_trip": None,
        "user_profile": {},
        "itinerary": [],
        "budget_breakdown": {},
        "travel_preferences": [],
        "demographic": "Youth (18-25)",
        "selected_model": "Qwen2.5-3B-Instruct",
        "model_status": {},
        "hf_token_valid": False,
        "last_model_check": {}
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def create_qwen_travel_prompt(user_input: str, demographic: str, budget: int, days: int, destination: str) -> str:
    """Create a specialized travel planning prompt optimized for Qwen models"""
    demo_info = DEMOGRAPHICS.get(demographic, {})
    preferences = demo_info.get("preferences", [])
    activity_style = demo_info.get("activity_style", "balanced")
    
    system_prompt = f"""You are TripCraft AI, an expert travel planning assistant. Create detailed, personalized travel itineraries based on demographics, budgets, and preferences.

Planning Context:
- Demographic: {demographic}
- Destination: {destination}
- Budget: ${budget} USD
- Duration: {days} days
- Preferences: {', '.join(preferences)}
- Activity Style: {activity_style}

Your expertise includes:
✈️ Demographic-specific recommendations
💰 Smart budget planning with detailed breakdowns
📅 Day-by-day itinerary creation
🏨 Accommodation matching preferences
🍽️ Local dining recommendations
🎯 Activities based on interests and capabilities
🚗 Transportation and logistics planning
🌍 Cultural insights and local tips

Instructions:
1. Provide practical, detailed, and personalized travel advice
2. Always consider demographic preferences and budget constraints
3. Include specific recommendations with estimated costs
4. Suggest accommodations suitable for the demographic
5. Recommend activities that match the activity style
6. Provide local insights and cultural tips
7. Include budget-saving tips specific to the demographic
8. Format response with clear sections and helpful details
9. Be specific about locations, timings, and costs
10. Consider seasonal factors and local events

Format your response with clear sections like:
- Overview
- Daily Itinerary
- Budget Breakdown
- Accommodation Suggestions
- Local Tips
- Transportation Guide"""

    return f"{system_prompt}\n\nUser Request: {user_input}\n\nTripCraft AI Response:"

def query_qwen_model(prompt: str, model_id: str, hf_token: str) -> str:
    """Query Qwen model with optimized parameters and robust error handling"""
    headers = {"Authorization": f"Bearer {hf_token}"}
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    
    # Get model-specific parameters
    model_name = next((name for name, info in QWEN_TRAVEL_MODELS.items() if info["model_id"] == model_id), "unknown")
    model_config = QWEN_TRAVEL_MODELS.get(model_name, QWEN_TRAVEL_MODELS["Qwen2.5-3B-Instruct"])
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": model_config["max_tokens"],
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "return_full_text": False,
            "repetition_penalty": 1.1,
            "stop": ["User Request:", "System:", "Human:", "Assistant:", "\n\nUser:", "\n\nHuman:"]
        },
        "options": {
            "wait_for_model": True,
            "use_cache": False
        }
    }
    
    max_retries = 3
    base_delay = 5
    timeout = model_config["timeout"]
    
    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
            
            # Handle different response codes
            if response.status_code == 200:
                result = response.json()
                text = extract_response_text(result)
                
                if text and len(text.strip()) > 10:
                    return clean_response_text(text)
                else:
                    return "I'm ready to help you plan your trip! Could you provide more details about your destination, preferences, or specific questions?"
            
            elif response.status_code == 503:
                # Model loading
                try:
                    error_data = response.json()
                    if "loading" in str(error_data).lower():
                        if attempt == 0:
                            estimated_time = error_data.get("estimated_time", 30)
                            st.warning(f"🤖 Model is loading... Estimated time: {estimated_time} seconds")
                        
                        wait_time = min(20 + (attempt * 15), 60)
                        time.sleep(wait_time)
                        continue
                    else:
                        return "The model service is temporarily busy. Please try again in a few moments."
                except:
                    if attempt < max_retries - 1:
                        time.sleep(base_delay * (attempt + 1))
                        continue
                    return "Model is currently loading. Please try again in a minute."
            
            elif response.status_code == 429:
                # Rate limiting
                if attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** attempt)
                    st.warning(f"Rate limit reached. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    return "I'm experiencing high demand right now. Please try again in a few minutes."
            
            elif response.status_code == 401:
                return "Authentication error. Please check your Hugging Face token configuration."
            
            elif response.status_code == 400:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", "Invalid request")
                    return f"Request error: {error_msg}. Please try rephrasing your question."
                except:
                    return "There was an issue with your request. Please try rephrasing your question."
            
            else:
                if attempt < max_retries - 1:
                    time.sleep(base_delay)
                    continue
                return f"Service temporarily unavailable (HTTP {response.status_code}). Please try again later."
                
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                st.warning(f"Request timeout. Retrying... (Attempt {attempt + 2}/{max_retries})")
                time.sleep(base_delay * (attempt + 1))
                continue
            else:
                return "The request timed out. The model might be busy. Please try again with a shorter query."
                
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(base_delay)
                continue
            else:
                return f"Connection error. Please check your internet connection and try again."
                
        except Exception as e:
            return f"An unexpected error occurred. Please try again or contact support if the issue persists."
    
    return "I'm unable to process your request at the moment. Please try again later."

def extract_response_text(result):
    """Extract text from various response formats"""
    if isinstance(result, list) and result:
        if isinstance(result[0], dict):
            return result[0].get('generated_text', '')
        else:
            return str(result[0])
    elif isinstance(result, dict):
        return result.get('generated_text', result.get('text', ''))
    else:
        return str(result)

def clean_response_text(text: str) -> str:
    """Clean up response text from artifacts"""
    if not text or not text.strip():
        return "I'm here to help you plan your trip! What would you like to know?"
    
    # Remove common artifacts
    cleanup_markers = [
        "TripCraft AI Response:",
        "Response:",
        "Assistant:",
        "TripCraft AI:",
        "Human:",
        "User Request:",
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>"
    ]
    
    cleaned_text = text
    for marker in cleanup_markers:
        if marker in cleaned_text:
            parts = cleaned_text.split(marker)
            if len(parts) > 1:
                cleaned_text = parts[-1].strip()
    
    # Remove excessive whitespace
    cleaned_text = ' '.join(cleaned_text.split())
    
    # Ensure minimum length
    if len(cleaned_text.strip()) < 20:
        return "I'd be happy to help you plan your trip! Could you share more details about your destination preferences, activities you enjoy, or specific questions about your travel plans?"
    
    return cleaned_text.strip()

def create_itinerary_display(destination: str, days: int, demographic: str, budget: int):
    """Create a visual itinerary display"""
    demo_info = DEMOGRAPHICS.get(demographic, {})
    
    # Sample activities based on demographic
    activities_by_demo = {
        "Youth (18-25)": [
            "Visit local hostels and meet fellow travelers",
            "Explore street food markets and local bars",
            "Take free walking tours of historic districts",
            "Visit museums on free/discounted days",
            "Join group adventure activities and tours"
        ],
        "Young Professionals (26-35)": [
            "Professional networking events and coworking spaces",
            "Boutique hotel experiences and rooftop bars",
            "Cultural workshops and cooking classes",
            "Business district tours and local startups",
            "Wine tasting and foodie experiences"
        ],
        "Families": [
            "Visit family-friendly attractions and theme parks",
            "Educational museum visits with interactive exhibits",
            "Safe playground and recreation areas",
            "Family dining at kid-friendly restaurants",
            "Interactive cultural experiences and workshops"
        ],
        "Couples": [
            "Romantic sunset viewing spots and scenic walks",
            "Couples spa and wellness treatments",
            "Fine dining experiences and wine tastings",
            "Private tours and intimate cultural sites",
            "Photography sessions at picturesque locations"
        ],
        "Seniors (55+)": [
            "Guided historical tours with comfortable transportation",
            "Cultural centers and heritage sites",
            "Scenic drives and accessible viewpoints",
            "Traditional craft workshops and demonstrations",
            "Comfortable dining with local specialties"
        ],
        "Adventure Seekers": [
            "Extreme sports and outdoor adventures",
            "Wilderness hiking and camping experiences",
            "Rock climbing and water sports",
            "Wildlife safaris and eco-tours",
            "Unique adventure accommodations"
        ],
        "Digital Nomads": [
            "Coworking spaces and networking events",
            "Long-term accommodation research",
            "Local SIM card and internet setup",
            "Cafe hopping for work-friendly spots",
            "Community meetups and events"
        ]
    }
    
    st.markdown("### 📅 Your Personalized Itinerary")
    
    activities = activities_by_demo.get(demographic, activities_by_demo["Youth (18-25)"])
    daily_budget = budget / days if days > 0 else budget
    
    for day in range(1, min(days + 1, 8)):  # Show up to 7 days
        st.markdown(f'<div class="day-header">Day {day} - {destination}</div>', unsafe_allow_html=True)
        
        # Morning activity
        morning_activity = activities[(day-1) % len(activities)]
        st.markdown(f'''
        <div class="activity-item">
            <strong>🌅 Morning (9:00 AM - 12:00 PM)</strong><br>
            {morning_activity}<br>
            <em>Estimated cost: ${daily_budget * 0.3:.0f}</em>
        </div>
        ''', unsafe_allow_html=True)
        
        # Afternoon activity
        afternoon_activity = activities[day % len(activities)]
        st.markdown(f'''
        <div class="activity-item">
            <strong>☀️ Afternoon (1:00 PM - 5:00 PM)</strong><br>
            {afternoon_activity}<br>
            <em>Estimated cost: ${daily_budget * 0.4:.0f}</em>
        </div>
        ''', unsafe_allow_html=True)
        
        # Evening activity
        evening_activity = activities[(day+1) % len(activities)]
        st.markdown(f'''
        <div class="activity-item">
            <strong>🌆 Evening (6:00 PM - 10:00 PM)</strong><br>
            {evening_activity}<br>
            <em>Estimated cost: ${daily_budget * 0.3:.0f}</em>
        </div>
        ''', unsafe_allow_html=True)

def create_budget_breakdown(total_budget: int, days: int, demographic: str):
    """Create detailed budget breakdown"""
    st.markdown("### 💰 Budget Breakdown")
    
    # Budget allocation based on demographic
    allocations = {
        "Youth (18-25)": {"accommodation": 0.25, "food": 0.30, "activities": 0.25, "transport": 0.15, "misc": 0.05},
        "Young Professionals (26-35)": {"accommodation": 0.30, "food": 0.25, "activities": 0.25, "transport": 0.15, "misc": 0.05},
        "Families": {"accommodation": 0.35, "food": 0.25, "activities": 0.20, "transport": 0.15, "misc": 0.05},
        "Couples": {"accommodation": 0.30, "food": 0.25, "activities": 0.25, "transport": 0.15, "misc": 0.05},
        "Seniors (55+)": {"accommodation": 0.40, "food": 0.20, "activities": 0.20, "transport": 0.15, "misc": 0.05},
        "Adventure Seekers": {"accommodation": 0.20, "food": 0.25, "activities": 0.35, "transport": 0.15, "misc": 0.05},
        "Digital Nomads": {"accommodation": 0.40, "food": 0.25, "activities": 0.15, "transport": 0.10, "misc": 0.10}
    }
    
    allocation = allocations.get(demographic, allocations["Youth (18-25)"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Total Budget Allocation")
        for category, percentage in allocation.items():
            amount = total_budget * percentage
            st.markdown(f'''
            <div class="budget-card">
                <strong>{category.title()}:</strong> ${amount:.0f} ({percentage*100:.0f}%)<br>
                <em>Daily: ${amount/days:.0f}</em>
            </div>
            ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 💡 Budget Tips")
        tips = {
            "Youth (18-25)": [
                "Book hostels in advance for better rates",
                "Use public transportation and walk when possible",
                "Look for free walking tours and city passes",
                "Cook some meals if accommodation has kitchen facilities"
            ],
            "Young Professionals (26-35)": [
                "Mix boutique hotels with quality mid-range options",
                "Book experiences in advance for better deals",
                "Use business lounges and co-working spaces",
                "Consider travel rewards credit cards"
            ],
            "Families": [
                "Look for family packages and group discounts",
                "Book accommodations with kitchenette",
                "Check for children's discounts at attractions",
                "Plan some free activities like parks and beaches"
            ],
            "Couples": [
                "Look for romantic package deals",
                "Book dinner reservations in advance",
                "Consider couples spa packages",
                "Split costs on shared experiences"
            ],
            "Seniors (55+)": [
                "Look for senior discounts at attractions",
                "Book guided tours for convenience and safety",
                "Choose accessible accommodations",
                "Consider comprehensive travel insurance"
            ],
            "Adventure Seekers": [
                "Book adventure activities in advance",
                "Consider camping or eco-lodges",
                "Look for multi-activity packages",
                "Invest in proper gear to avoid rental costs"
            ],
            "Digital Nomads": [
                "Look for monthly accommodation discounts",
                "Invest in reliable internet connectivity",
                "Join co-working spaces for networking",
                "Plan for work equipment and setup costs"
            ]
        }
        
        for tip in tips.get(demographic, tips["Youth (18-25)"]):
            st.markdown(f"• {tip}")

def format_travel_message(role: str, content: str):
    """Format travel-specific chat messages"""
    if role == "user":
        st.markdown(f'''
        <div class="chat-message user-message">
            <div style="font-weight: bold; margin-bottom: 0.5rem;">🧳 You</div>
            <div>{content}</div>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="chat-message assistant-message">
            <div style="font-weight: bold; margin-bottom: 0.5rem;">✈️ TripCraft AI</div>
            <div>{content}</div>
        </div>
        ''', unsafe_allow_html=True)

def main():
    init_session_state()
    
    # Load configuration
    hf_token = get_hf_token()
    
    if not hf_token:
        st.error("🚨 **Configuration Error**: HUGGINGFACE_TOKEN not found.")
        st.markdown("""
        **Setup Instructions:**
        1. Get your Hugging Face token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
        2. Add it to your Streamlit Cloud secrets as `HUGGINGFACE_TOKEN`
        3. Refresh this page
        
        **For local development:** Set the environment variable `HUGGINGFACE_TOKEN`
        """)
        return
    
    # Validate token format
    if not hf_token.startswith('hf_'):
        st.error("🚨 **Invalid Token**: Hugging Face tokens should start with 'hf_'")
        return
    
    st.session_state.hf_token_valid = True
    
    # Header
    st.markdown('''
    <div class="main-header">
        <h1>✈️ TripCraft AI</h1>
        <p>Powered by Qwen Language Models</p>
        <p><em>Advanced AI • Demographic-Based • Budget-Conscious • Personalized Planning</em></p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🎯 Trip Planning Setup")
        
        # Qwen Model selection
        st.subheader("🤖 Qwen AI Model")
        
       # Show recommended models first
        recommended_models = [name for name, info in QWEN_TRAVEL_MODELS.items() if info.get("recommended", False)]
        other_models = [name for name, info in QWEN_TRAVEL_MODELS.items() if not info.get("recommended", False)]
        
        model_options = recommended_models + ["---"] + other_models
        
        selected_model = st.selectbox(
            "Choose AI Model:",
            model_options,
            index=0,
            help="Recommended models are optimized for travel planning"
        )
        
        if selected_model == "---":
            st.warning("Please select a valid model")
            return
        
        st.session_state.selected_model = selected_model
        
        # Show model info
        model_info = QWEN_TRAVEL_MODELS[selected_model]
        st.markdown("**Model Features:**")
        for feature in model_info["features"]:
            st.markdown(f"• {feature}")
        
        # Model status check
        if st.button("🔍 Check Model Status"):
            with st.spinner("Checking model availability..."):
                status = check_model_availability(model_info["model_id"], hf_token)
                st.session_state.model_status[selected_model] = status
                st.session_state.last_model_check[selected_model] = datetime.now()
        
        # Display model status
        if selected_model in st.session_state.model_status:
            status = st.session_state.model_status[selected_model]
            last_check = st.session_state.last_model_check.get(selected_model)
            
            if status["status"] == "available":
                st.markdown(f'<div class="model-status model-available">✅ {status["message"]}</div>', unsafe_allow_html=True)
            elif status["status"] == "loading":
                st.markdown(f'<div class="model-status model-loading">⏳ {status["message"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="model-status model-error">❌ {status["message"]}</div>', unsafe_allow_html=True)
            
            if last_check:
                time_diff = datetime.now() - last_check
                st.caption(f"Last checked: {time_diff.seconds}s ago")
        
        st.divider()
        
        # User demographics
        st.subheader("👥 Travel Profile")
        demographic = st.selectbox(
            "Traveler Type:",
            list(DEMOGRAPHICS.keys()),
            index=0,
            help="This helps customize recommendations to your travel style"
        )
        st.session_state.demographic = demographic
        
        # Show demographic info
        demo_info = DEMOGRAPHICS[demographic]
        st.markdown("**Your Travel Style:**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Activity Style:** {demo_info['activity_style'].replace('_', ' ').title()}")
            st.markdown(f"**Budget Range:** ${demo_info['budget_range'][0]:,} - ${demo_info['budget_range'][1]:,}")
        
        with col2:
            st.markdown("**Preferences:**")
            for pref in demo_info['preferences'][:3]:
                st.markdown(f"<span class='demographic-tag'>{pref.replace('_', ' ').title()}</span>", unsafe_allow_html=True)
        
        st.divider()
        
        # Trip details
        st.subheader("🌍 Trip Details")
        
        destination = st.text_input(
            "Destination:",
            placeholder="e.g., Paris, Tokyo, New York, Bali",
            help="Enter the city or country you want to visit"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            days = st.number_input(
                "Duration (days):",
                min_value=1,
                max_value=30,
                value=7,
                help="Number of days for your trip"
            )
        
        with col2:
            budget = st.number_input(
                "Budget (USD):",
                min_value=100,
                max_value=50000,
                value=demo_info['budget_range'][0],
                step=100,
                help="Total budget for your entire trip"
            )
        
        # Travel dates
        start_date = st.date_input(
            "Travel Start Date:",
            value=datetime.now() + timedelta(days=30),
            min_value=datetime.now().date(),
            help="When do you plan to start your trip?"
        )
        
        # Additional preferences
        st.subheader("🎯 Additional Preferences")
        
        travel_style = st.multiselect(
            "Travel Interests:",
            ["Culture & History", "Food & Dining", "Adventure & Sports", "Nature & Wildlife", 
             "Nightlife & Entertainment", "Shopping", "Relaxation & Wellness", "Photography",
             "Local Experiences", "Museums & Arts", "Festivals & Events", "Architecture"],
            help="Select your main interests for this trip"
        )
        
        accommodation_type = st.selectbox(
            "Preferred Accommodation:",
            ["Hotels", "Hostels", "Vacation Rentals", "Boutique Hotels", "Resorts", "Camping", "Mixed"],
            help="What type of accommodation do you prefer?"
        )
        
        transportation = st.selectbox(
            "Transportation Preference:",
            ["Public Transport", "Rental Car", "Walking + Public", "Taxi/Rideshare", "Tours", "Mixed"],
            help="How do you prefer to get around?"
        )
        
        # Save preferences
        st.session_state.user_profile = {
            "demographic": demographic,
            "destination": destination,
            "days": days,
            "budget": budget,
            "start_date": start_date,
            "travel_style": travel_style,
            "accommodation": accommodation_type,
            "transportation": transportation
        }
        
        # Quick trip planner
        st.divider()
        st.subheader("⚡ Quick Planner")
        
        if destination and st.button("🎯 Generate Quick Itinerary", type="secondary"):
            create_itinerary_display(destination, days, demographic, budget)
            create_budget_breakdown(budget, days, demographic)
    
    # Main chat interface
    st.subheader("💬 Chat with TripCraft AI")
    
    # Display chat history
    for message in st.session_state.messages:
        format_travel_message(message["role"], message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about your travel plans... (e.g., 'Plan a 5-day trip to Tokyo for $2000')"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        format_travel_message("user", prompt)
        
        # Prepare context
        user_profile = st.session_state.user_profile
        demographic = user_profile.get("demographic", "Youth (18-25)")
        destination = user_profile.get("destination", "")
        days = user_profile.get("days", 7)
        budget = user_profile.get("budget", 1500)
        
        # Create enhanced prompt
        travel_prompt = create_qwen_travel_prompt(prompt, demographic, budget, days, destination)
        
        # Show thinking indicator
        with st.spinner(f"🤖 TripCraft AI ({st.session_state.selected_model}) is planning your trip..."):
            model_info = QWEN_TRAVEL_MODELS[st.session_state.selected_model]
            response = query_qwen_model(travel_prompt, model_info["model_id"], hf_token)
        
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})
        format_travel_message("assistant", response)
        
        # Show model info
        st.markdown(f'<div class="model-info">Response generated by: {st.session_state.selected_model}</div>', unsafe_allow_html=True)
    
    # Action buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🗺️ Get Itinerary"):
            if st.session_state.user_profile.get("destination"):
                create_itinerary_display(
                    st.session_state.user_profile["destination"],
                    st.session_state.user_profile["days"],
                    st.session_state.user_profile["demographic"],
                    st.session_state.user_profile["budget"]
                )
            else:
                st.warning("Please set your destination in the sidebar first!")
    
    with col2:
        if st.button("💰 Budget Plan"):
            if st.session_state.user_profile.get("budget"):
                create_budget_breakdown(
                    st.session_state.user_profile["budget"],
                    st.session_state.user_profile["days"],
                    st.session_state.user_profile["demographic"]
                )
            else:
                st.warning("Please set your budget in the sidebar first!")
    
    with col3:
        if st.button("🎯 Smart Suggestions"):
            demo = st.session_state.user_profile.get("demographic", "Youth (18-25)")
            dest = st.session_state.user_profile.get("destination", "your destination")
            
            suggestions_prompt = f"Give me 5 smart travel tips and hidden gems for {demo} travelers visiting {dest}"
            
            with st.spinner("Getting personalized suggestions..."):
                model_info = QWEN_TRAVEL_MODELS[st.session_state.selected_model]
                travel_prompt = create_qwen_travel_prompt(suggestions_prompt, demo, 
                    st.session_state.user_profile.get("budget", 1500),
                    st.session_state.user_profile.get("days", 7), dest)
                response = query_qwen_model(travel_prompt, model_info["model_id"], hf_token)
            
            st.session_state.messages.append({"role": "user", "content": suggestions_prompt})
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
    
    with col4:
        if st.button("🧹 Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Footer with tips
    st.divider()
    st.markdown("### 💡 Travel Planning Tips")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🎯 Be Specific**
        - Include your destination
        - Mention your budget range
        - Specify travel dates
        - Share your interests
        """)
    
    with col2:
        st.markdown("""
        **💬 Example Questions**
        - "Plan a 7-day trip to Japan for $3000"
        - "Best family activities in London"
        - "Budget backpacking in Southeast Asia"
        - "Romantic getaway ideas under $2000"
        """)
    
    with col3:
        st.markdown("""
        **🤖 AI Features**
        - Demographic-based recommendations
        - Real-time budget planning
        - Cultural insights & local tips
        - Personalized itinerary creation
        """)
    
    # About section
    with st.expander("ℹ️ About TripCraft AI"):
        st.markdown("""
        **TripCraft AI** is powered by advanced Qwen language models to provide personalized travel planning assistance. 
        
        **Key Features:**
        - 🎯 **Demographic-Based Planning**: Tailored recommendations for different traveler types
        - 💰 **Smart Budget Management**: Detailed cost breakdowns and money-saving tips
        - 🌍 **Cultural Insights**: Local knowledge and authentic experiences
        - 📱 **Interactive Planning**: Real-time chat with AI travel expert
        - 🏨 **Comprehensive Coverage**: Accommodations, activities, dining, and transportation
        
        **Supported Demographics:**
        - Youth (18-25): Adventure and budget-focused
        - Young Professionals (26-35): Balanced luxury and experiences
        - Families: Kid-friendly and educational activities
        - Couples: Romantic and intimate experiences
        - Seniors (55+): Comfortable and cultural experiences
        - Adventure Seekers: Extreme sports and unique adventures
        - Digital Nomads: Work-travel balance and community
        
        **Powered by Qwen Models**: State-of-the-art language models optimized for detailed, helpful, and culturally-aware travel planning.
        """)

if __name__ == "__main__":
    main()
