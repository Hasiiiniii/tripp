import streamlit as st
import requests
import json
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd
import time

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
    
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# FIXED: Available models that actually work with Hugging Face Inference API
WORKING_TRAVEL_MODELS = {
    "Qwen2.5-7B-Instruct (Recommended)": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "description": "Best balance of performance and speed",
        "features": ["Excellent Planning", "Detailed Responses", "Multilingual"],
        "status": "Available"
    },
    "Qwen2.5-14B-Instruct (Advanced)": {
        "model_id": "Qwen/Qwen2.5-14B-Instruct", 
        "description": "Most capable model for complex planning",
        "features": ["Deep Analysis", "Complex Reasoning", "Cultural Insights"],
        "status": "Available"
    },
    "Qwen2.5-72B-Instruct (Premium)": {
        "model_id": "Qwen/Qwen2.5-72B-Instruct",
        "description": "Top-tier model with exceptional capabilities",
        "features": ["Best Quality", "Comprehensive Planning", "Expert Level"],
        "status": "May be slow"
    },
    "Qwen2-7B-Instruct (Stable)": {
        "model_id": "Qwen/Qwen2-7B-Instruct",
        "description": "Reliable and well-tested model",
        "features": ["Stable", "Fast Response", "Good Quality"],
        "status": "Available"
    },
    "Microsoft DialoGPT (Alternative)": {
        "model_id": "microsoft/DialoGPT-large",
        "description": "Alternative conversational model",
        "features": ["Fast", "Conversational", "Backup Option"],
        "status": "Available"
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
    }
}

def init_session_state():
    """Initialize session state for TripCraft AI"""
    defaults = {
        "messages": [],
        "current_trip": None,
        "user_profile": {},
        "itinerary": [],
        "budget_breakdown": {},
        "travel_preferences": [],
        "demographic": None,
        "selected_model": "Qwen2.5-7B-Instruct (Recommended)"
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def create_travel_prompt(user_input: str, demographic: str, budget: int, days: int, destination: str, model_name: str) -> str:
    """Create a travel planning prompt optimized for available models"""
    demo_info = DEMOGRAPHICS.get(demographic, {})
    preferences = demo_info.get("preferences", [])
    activity_style = demo_info.get("activity_style", "balanced")
    
    # Optimized prompt for Qwen models
    system_prompt = f"""You are TripCraft AI, an expert travel planning assistant. Create personalized, detailed travel itineraries based on user demographics, preferences, and budget constraints.

TRAVEL CONTEXT:
- Traveler Type: {demographic}
- Destination: {destination}
- Budget: ${budget:,} for {days} days
- Preferred Activities: {', '.join(preferences)}
- Travel Style: {activity_style.replace('_', ' ').title()}

YOUR EXPERTISE:
✈️ Demographic-specific recommendations
💰 Detailed budget planning and cost breakdowns
📅 Day-by-day itinerary creation
🏨 Accommodation suggestions matching preferences
🍽️ Local dining and cuisine recommendations
🎯 Activities tailored to interests and capabilities
🚗 Transportation and logistics planning
🌍 Cultural insights and local tips

RESPONSE GUIDELINES:
- Provide practical, actionable travel advice
- Consider the traveler's demographic preferences
- Stay within budget constraints
- Include specific costs when possible
- Suggest alternatives for different price points
- Mention seasonal considerations
- Include safety and cultural tips

Focus on creating a memorable experience that matches the {demographic.lower()} travel style while staying within the ${budget:,} budget."""

    return f"{system_prompt}\n\nUser Question: {user_input}\n\nTripCraft AI Response:"

def query_model_with_retry(prompt: str, model_id: str, hf_token: str, max_retries: int = 3) -> str:
    """Query model with improved error handling and retry logic"""
    headers = {"Authorization": f"Bearer {hf_token}"}
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    
    # Optimized parameters for different model types
    if "Qwen" in model_id:
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 500,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "return_full_text": False,
                "repetition_penalty": 1.1,
                "stop": ["<|im_end|>", "<|endoftext|>"]
            }
        }
    else:  # DialoGPT or other models
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 400,
                "temperature": 0.8,
                "top_p": 0.9,
                "do_sample": True,
                "return_full_text": False
            }
        }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            
            # Handle different response status codes
            if response.status_code == 503:
                error_data = response.json() if response.content else {}
                estimated_time = error_data.get("estimated_time", 30)
                
                if attempt < max_retries - 1:
                    st.warning(f"🤖 Model {model_id.split('/')[-1]} is loading... Estimated time: {estimated_time}s. Retrying in {estimated_time}s...")
                    time.sleep(min(estimated_time, 60))  # Cap wait time at 60 seconds
                    continue
                else:
                    return f"Model is currently loading. Please try again in a few minutes. You can also try a different model from the dropdown."
            
            elif response.status_code == 429:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10
                    st.warning(f"Rate limit reached. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    return "Rate limit exceeded. Please wait a moment and try again, or switch to a different model."
            
            elif response.status_code == 401:
                return "Authentication failed. Please check your Hugging Face token in the secrets."
            
            elif response.status_code == 404:
                return f"Model {model_id} not found. Please select a different model."
            
            response.raise_for_status()
            result = response.json()
            
            # Extract response text
            if isinstance(result, list) and result:
                text = result[0].get('generated_text', '')
            elif isinstance(result, dict):
                text = result.get('generated_text', result.get('text', str(result)))
            else:
                text = str(result)
            
            # Clean up response
            cleanup_markers = [
                "TripCraft AI Response:",
                "Assistant:",
                "Response:",
                "<|im_start|>assistant",
                "<|im_end|>"
            ]
            
            for marker in cleanup_markers:
                if marker in text:
                    text = text.split(marker)[-1].strip()
            
            # Ensure we have a meaningful response
            if not text.strip() or len(text.strip()) < 10:
                if attempt < max_retries - 1:
                    continue
                else:
                    return "I'm having trouble generating a response. Could you please rephrase your question or try a different model?"
            
            return text.strip()
            
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                st.warning(f"Request timeout. Retrying... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(5)
                continue
            else:
                return "Request timed out. Please try again with a shorter question or different model."
                
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            else:
                return f"Connection error: {str(e)}. Please check your internet connection and try again."
        
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            else:
                return f"An unexpected error occurred: {str(e)}. Please try a different model."
    
    return "Failed to get response after multiple attempts. Please try again later."

def test_model_availability(model_id: str, hf_token: str) -> bool:
    """Test if a model is available and responsive"""
    headers = {"Authorization": f"Bearer {hf_token}"}
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    
    test_payload = {
        "inputs": "Hello, can you help me plan a trip?",
        "parameters": {
            "max_new_tokens": 50,
            "temperature": 0.7
        }
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=test_payload, timeout=10)
        return response.status_code in [200, 503]  # 503 means loading but available
    except:
        return False

def create_itinerary_display(destination: str, days: int, demographic: str, budget: int):
    """Create a visual itinerary display"""
    demo_info = DEMOGRAPHICS.get(demographic, {})
    preferences = demo_info.get("preferences", [])
    
    # Sample activities based on demographic
    activities_by_demo = {
        "Youth (18-25)": [
            "Visit local hostels and meet fellow travelers",
            "Explore street food markets and local bars",
            "Take free walking tours",
            "Visit museums on free days",
            "Join group adventure activities"
        ],
        "Families": [
            "Visit family-friendly attractions and parks",
            "Educational museum visits",
            "Safe playground and recreation areas",
            "Family dining at kid-friendly restaurants",
            "Interactive cultural experiences"
        ],
        "Couples": [
            "Romantic sunset viewing spots",
            "Couples spa and wellness treatments",
            "Fine dining experiences",
            "Scenic walks and photography",
            "Wine tasting or cultural tours"
        ],
        "Seniors (55+)": [
            "Guided historical tours",
            "Comfortable cultural experiences",
            "Scenic drives and viewpoints",
            "Traditional craft workshops",
            "Heritage site visits"
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
        "Families": {"accommodation": 0.35, "food": 0.25, "activities": 0.20, "transport": 0.15, "misc": 0.05},
        "Couples": {"accommodation": 0.30, "food": 0.25, "activities": 0.25, "transport": 0.15, "misc": 0.05},
        "Seniors (55+)": {"accommodation": 0.40, "food": 0.20, "activities": 0.20, "transport": 0.15, "misc": 0.05}
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
                "Use public transportation",
                "Look for free walking tours",
                "Cook some meals if possible"
            ],
            "Families": [
                "Look for family packages",
                "Book accommodations with kitchenette",
                "Check for children's discounts",
                "Plan some free activities"
            ],
            "Couples": [
                "Look for romantic package deals",
                "Book dinner reservations in advance",
                "Consider spa packages",
                "Split costs on experiences"
            ],
            "Seniors (55+)": [
                "Look for senior discounts",
                "Book guided tours for convenience",
                "Choose accessible accommodations",
                "Consider travel insurance"
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
    try:
        hf_token = st.secrets.get("HUGGINGFACE_TOKEN", "")
        if not hf_token:
            st.error("🚨 **Configuration Error**: HUGGINGFACE_TOKEN not found in secrets.")
            st.info("Please add your Hugging Face token to Streamlit secrets to use the AI models.")
            st.code('''
            # Add this to your .streamlit/secrets.toml file:
            HUGGINGFACE_TOKEN = "your_token_here"
            ''')
            return
    except:
        st.error("🚨 **Configuration Error**: Please add your Hugging Face token to secrets.")
        return
    
    # Header
    st.markdown('''
    <div class="main-header">
        <h1>✈️ TripCraft AI</h1>
        <p>Powered by Qwen & Advanced Language Models</p>
        <p><em>Fixed Version • Working Models • Personalized Planning</em></p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🎯 Trip Planning Setup")
        
        # Model selection with status indicators
        st.subheader("🤖 AI Model Selection")
        selected_model_name = st.selectbox(
            "Choose your travel advisor:",
            list(WORKING_TRAVEL_MODELS.keys()),
            key="model_selector"
        )
        
        model_info = WORKING_TRAVEL_MODELS[selected_model_name]
        model_id = model_info["model_id"]
        
        # Display model information
        status_color = "#28a745" if model_info["status"] == "Available" else "#ffc107"
        st.markdown(f'''
        <div class="model-info">
            <strong>{selected_model_name}</strong><br>
            {model_info["description"]}<br>
            Features: {", ".join(model_info["features"])}<br>
            <span style="color: {status_color};">● {model_info["status"]}</span>
        </div>
        ''', unsafe_allow_html=True)
        
        # Model testing
        if st.button("🧪 Test Model", help="Test if the selected model is working"):
            with st.spinner("Testing model..."):
                is_available = test_model_availability(model_id, hf_token)
                if is_available:
                    st.success(f"✅ {selected_model_name} is available!")
                else:
                    st.error(f"❌ {selected_model_name} is not responding. Try another model.")
        
        st.session_state.selected_model = selected_model_name
        
        st.markdown("---")
        
        # Demographic selection
        st.subheader("👥 Travel Demographics")
        demographic = st.selectbox(
            "What describes your travel group?",
            list(DEMOGRAPHICS.keys())
        )
        
        # Display demographic info
        if demographic:
            demo_info = DEMOGRAPHICS[demographic]
            st.markdown(f'<div class="demographic-tag">{demographic}</div>', unsafe_allow_html=True)
            st.caption(f"Budget range: ${demo_info['budget_range'][0]:,} - ${demo_info['budget_range'][1]:,}")
            st.caption(f"Style: {demo_info['activity_style'].replace('_', ' ').title()}")
        
        st.markdown("---")
        
        # Travel details
        st.subheader("🗺️ Trip Details")
        destination = st.text_input("Destination", placeholder="e.g., Paris, France")
        
        col1, col2 = st.columns(2)
        with col1:
            days = st.number_input("Days", min_value=1, max_value=30, value=7)
        with col2:
            budget = st.number_input("Budget ($)", min_value=100, max_value=50000, value=2000, step=100)
        
        # Travel dates
        start_date = st.date_input("Start Date", datetime.now() + timedelta(days=30))
        
        # Save to session state
        st.session_state.user_profile = {
            "demographic": demographic,
            "destination": destination,
            "days": days,
            "budget": budget,
            "start_date": start_date
        }
        
        st.markdown("---")
        
        # Quick actions
        if st.button("🎨 Generate Sample Itinerary", use_container_width=True):
            if destination:
                create_itinerary_display(destination, days, demographic, budget)
        
        if st.button("💰 Show Budget Breakdown", use_container_width=True):
            create_budget_breakdown(budget, days, demographic)
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Model status warning
    if "May be slow" in model_info["status"]:
        st.markdown('''
        <div class="warning-box">
            ⚠️ <strong>Note:</strong> The selected model (72B) is very large and may take longer to respond. 
            Consider using Qwen2.5-7B-Instruct for faster responses.
        </div>
        ''', unsafe_allow_html=True)
    
    # Main chat interface
    st.subheader(f"💬 Chat with TripCraft AI ({selected_model_name})")
    
    # Display chat history
    for message in st.session_state.messages:
        format_travel_message(message["role"], message["content"])
    
    # Chat input
    user_input = st.chat_input("Ask about destinations, activities, budgets, or anything travel-related...")
    
    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        format_travel_message("user", user_input)
        
        # Generate AI response
        profile = st.session_state.user_profile
        travel_prompt = create_travel_prompt(
            user_input, 
            profile.get("demographic", "Youth (18-25)"),
            profile.get("budget", 2000),
            profile.get("days", 7),
            profile.get("destination", ""),
            selected_model_name
        )
        
        with st.spinner(f"🌍 {selected_model_name} is crafting your perfect travel plan..."):
            response = query_model_with_retry(travel_prompt, model_id, hf_token)
        
        # Add and display AI response
        st.session_state.messages.append({"role": "assistant", "content": response})
        format_travel_message("assistant", response)
        
        st.rerun()
    
    # Travel insights section
    if not st.session_state.messages:
        st.markdown("### 🌟 What TripCraft AI Can Do For You")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🧠 Advanced AI Planning**
            - Qwen model capabilities
            - Complex itinerary optimization
            - Cultural context understanding
            - Multi-language support
            - Intelligent recommendations
            """)
        
        with col2:
            st.markdown("""
            **💰 Smart Budget Management**
            - Detailed cost breakdowns
            - Demographic-based allocation
            - Money-saving strategies
            - Value optimization
            - Emergency fund planning
            """)
        
        with col3:
            st.markdown("""
            **📅 Personalized Scheduling**
            - Day-by-day itineraries
            - Activity timing optimization
            - Weather considerations
            - Local event integration
            - Flexible planning
            """)
        
        st.markdown("### 💡 Try asking TripCraft AI:")
        suggestions = [
            "Plan a 5-day budget trip to Tokyo for young professionals",
            "What are the best family activities in London with kids under 10?",
            "
