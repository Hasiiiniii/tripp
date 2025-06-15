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
    
    .error-message {
        background: #ffebee;
        border: 1px solid #ef5350;
        padding: 1rem;
        border-radius: 8px;
        color: #c62828;
        margin: 1rem 0;
    }
    
    .loading-message {
        background: #e3f2fd;
        border: 1px solid #2196f3;
        padding: 1rem;
        border-radius: 8px;
        color: #1565c0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Updated Qwen models with correct and available model IDs
QWEN_TRAVEL_MODELS = {
    "Qwen2.5-7B-Instruct": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "description": "Powerful model for complex travel planning",
        "features": ["Deep Reasoning", "Complex Planning", "Cultural Insights"]
    },
    "Qwen2.5-3B-Instruct": {
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "description": "Balanced model for travel recommendations",
        "features": ["Good Balance", "Detailed Planning", "Reliable"]
    },
    "Qwen2.5-1.5B-Instruct": {
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "description": "Efficient model for quick responses",
        "features": ["Fast", "Efficient", "Basic Planning"]
    },
    "Qwen2.5-0.5B-Instruct": {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "description": "Lightweight model for quick travel tips",
        "features": ["Very Fast", "Quick Tips", "Minimal Resources"]
    },
    "Qwen2-7B-Instruct": {
        "model_id": "Qwen/Qwen2-7B-Instruct",
        "description": "Reliable Qwen2 model for travel planning",
        "features": ["Stable", "Comprehensive", "Well-tested"]
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
        "selected_model": "Qwen2.5-3B-Instruct"
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def create_qwen_travel_prompt(user_input: str, demographic: str, budget: int, days: int, destination: str) -> str:
    """Create a specialized travel planning prompt optimized for Qwen models"""
    demo_info = DEMOGRAPHICS.get(demographic, {})
    preferences = demo_info.get("preferences", [])
    activity_style = demo_info.get("activity_style", "balanced")
    
    # Simplified prompt that works better with Qwen models
    system_prompt = f"""You are TripCraft AI, an expert travel planning assistant. Create personalized travel recommendations.

TRAVEL PLANNING CONTEXT:
- Traveler Type: {demographic}
- Destination: {destination}
- Budget: ${budget} USD
- Duration: {days} days
- Preferences: {', '.join(preferences)}
- Activity Style: {activity_style}

INSTRUCTIONS:
1. Provide practical travel advice based on the demographic and budget
2. Suggest specific activities, accommodations, and dining options
3. Include estimated costs when possible
4. Consider the traveler's age group and interests
5. Be specific and actionable in your recommendations

USER QUESTION: {user_input}

RESPONSE:"""
    
    return system_prompt

def test_model_availability(model_id: str, hf_token: str) -> bool:
    """Test if a model is available and responsive"""
    headers = {"Authorization": f"Bearer {hf_token}"}
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    
    test_payload = {
        "inputs": "Hello, can you help with travel planning?",
        "parameters": {
            "max_new_tokens": 50,
            "temperature": 0.7,
            "return_full_text": False
        }
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=test_payload, timeout=10)
        if response.status_code == 200:
            return True
        elif response.status_code == 503:
            # Model is loading but available
            return True
        else:
            return False
    except:
        return False

def query_qwen_model(prompt: str, model_id: str, hf_token: str, max_retries: int = 3) -> str:
    """Query Qwen model with improved error handling and retry logic"""
    headers = {"Authorization": f"Bearer {hf_token}"}
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    
    # Optimized parameters for better responses
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "return_full_text": False,
            "repetition_penalty": 1.1,
            "stop_sequences": ["USER:", "ASSISTANT:", "Human:", "AI:"]
        }
    }
    
    for attempt in range(max_retries):
        try:
            st.write(f"🔄 Attempt {attempt + 1}: Querying {model_id.split('/')[-1]}...")
            
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            
            # Handle different response codes
            if response.status_code == 503:
                error_data = response.json() if response.content else {}
                estimated_time = error_data.get('estimated_time', 30)
                
                st.warning(f"🤖 Model {model_id.split('/')[-1]} is loading... Estimated time: {estimated_time}s")
                
                # Wait and retry
                time.sleep(min(estimated_time + 5, 60))
                continue
                
            elif response.status_code == 429:
                st.warning("⏳ Rate limited. Waiting before retry...")
                time.sleep(10)
                continue
                
            elif response.status_code != 200:
                st.error(f"❌ HTTP {response.status_code}: {response.text}")
                if attempt == max_retries - 1:
                    return f"I'm experiencing technical difficulties with the {model_id.split('/')[-1]} model. The service returned an error (HTTP {response.status_code}). Please try again later or select a different model."
                continue
            
            # Parse successful response
            result = response.json()
            
            # Extract text from response
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict):
                    text = result[0].get('generated_text', '')
                else:
                    text = str(result[0])
            elif isinstance(result, dict):
                text = result.get('generated_text', result.get('text', ''))
            else:
                text = str(result)
            
            # Clean up the response
            if text:
                # Remove common artifacts
                cleanup_patterns = [
                    "RESPONSE:",
                    "ASSISTANT:",
                    "AI:",
                    "<|im_start|>assistant",
                    "<|im_end|>",
                    "<|endoftext|>",
                    "Human:",
                    "USER:"
                ]
                
                for pattern in cleanup_patterns:
                    if pattern in text:
                        parts = text.split(pattern)
                        text = parts[-1].strip()
                
                # Remove leading/trailing whitespace and empty lines
                text = text.strip()
                
                if len(text) > 10:  # Reasonable response length
                    st.success(f"✅ Response received from {model_id.split('/')[-1]}")
                    return text
                else:
                    st.warning(f"⚠️ Short response from {model_id.split('/')[-1]}: {text}")
                    if attempt == max_retries - 1:
                        return "I received a very short response. Could you please rephrase your question or provide more specific details about your travel plans?"
            else:
                st.warning(f"⚠️ Empty response from {model_id.split('/')[-1]}")
                if attempt == max_retries - 1:
                    return "I didn't receive a proper response. Please try rephrasing your question or select a different model."
        
        except requests.exceptions.Timeout:
            st.warning(f"⏱️ Timeout on attempt {attempt + 1}")
            if attempt == max_retries - 1:
                return "The request timed out. The model might be under heavy load. Please try again later."
        
        except requests.exceptions.RequestException as e:
            st.error(f"🌐 Network error: {str(e)}")
            if attempt == max_retries - 1:
                return "I'm experiencing network connectivity issues. Please check your internet connection and try again."
        
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            if attempt == max_retries - 1:
                return "An unexpected error occurred. Please try again or contact support if the problem persists."
        
        # Wait before retry
        if attempt < max_retries - 1:
            time.sleep(5)
    
    return "I was unable to process your request after multiple attempts. Please try again later."

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
        hf_token = st.secrets.get("HUGGINGFACE_TOKEN", "").strip()
        if not hf_token:
            st.error("🚨 **Configuration Error**: HUGGINGFACE_TOKEN not found in secrets.")
            st.info("Please add your Hugging Face token to Streamlit secrets to use the Qwen models.")
            st.code("""
# Add this to your Streamlit secrets (.streamlit/secrets.toml):
[secrets]
HUGGINGFACE_TOKEN = "hf_your_token_here"
            """)
            return
    except Exception as e:
        st.error(f"🚨 **Configuration Error**: {str(e)}")
        return
    
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
        selected_model_name = st.selectbox(
            "Choose your Qwen travel advisor:",
            list(QWEN_TRAVEL_MODELS.keys()),
            key="model_selector"
        )
        
        model_info = QWEN_TRAVEL_MODELS[selected_model_name]
        model_id = model_info["model_id"]
        
        # Display model information
        st.markdown(f'''
        <div class="model-info">
            <strong>{selected_model_name}</strong><br>
            {model_info["description"]}<br>
            Features: {", ".join(model_info["features"])}
        </div>
        ''', unsafe_allow_html=True)
        
        # Test model availability
        if st.button("🔍 Test Model Availability"):
            with st.spinner("Testing model..."):
                if test_model_availability(model_id, hf_token):
                    st.success(f"✅ {selected_model_name} is available!")
                else:
                    st.error(f"❌ {selected_model_name} is not available. Try another model.")
        
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
    
    # Main chat interface
    st.subheader(f"💬 Chat with TripCraft AI ({selected_model_name})")
    
    # Display token status
    if hf_token:
        st.info(f"🔑 Using Hugging Face token: {hf_token[:10]}...")
    
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
        travel_prompt = create_qwen_travel_prompt(
            user_input, 
            profile.get("demographic", "Youth (18-25)"),
            profile.get("budget", 2000),
            profile.get("days", 7),
            profile.get("destination", "")
        )
        
        with st.spinner(f"🌍 {selected_model_name} is planning your perfect trip..."):
            response = query_qwen_model(travel_prompt, model_id, hf_token)
        
        # Add and display AI response
        st.session_state.messages.append({"role": "assistant", "content": response})
        format_travel_message("assistant", response)
        
        st.rerun()
    
    # Travel insights section
    if not st.session_state.messages:
        st.markdown("### 🌟 What TripCraft AI with Qwen Can Do For You")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🧠 Advanced AI Planning**
            - Qwen model intelligence
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
            "Create a romantic weekend getaway itinerary for couples in Santorini",
            "What's the ideal budget breakdown for a senior trip to Egypt?",
            "Suggest adventure activities in New Zealand for thrill-seekers",
            "Help me plan a 10-day cultural tour of India within $3000"
        ]
        
        for suggestion in suggestions:
            if st.button(f"💭 {suggestion}", key=suggestion):
                st.session_state.messages.append({"role": "user", "content": suggestion})
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #666;">
        <p>✈️ <strong>TripCraft AI</strong> - Powered by {selected_model_name} | 
        🤖 Qwen Language Models | 🎯 Personalized Travel Planning | 💡 Built for Smart Adventures</p>
        <p><small>Model: {model_id}</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
