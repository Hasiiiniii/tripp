# 🌍 TripCraft AI - Personalized Travel Itinerary Assistant

![TripCraft AI](https://img.shields.io/badge/TripCraft-AI-blue?style=for-the-badge&logo=travel)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)
![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-FFD21E?style=for-the-badge)

**TripCraft AI** is an intelligent travel planning assistant designed to help users create personalized travel itineraries through advanced AI algorithms and demographic-based recommendations.

Developed as part of the **Summer of AI internship** at **Viswam.ai**, in collaboration with **Meta**, **Swecha Telangana**, and **IIIT Hyderabad**.

## ✨ Features

### 🔍 **Demographic-Based Recommendations**
- Tailored suggestions for youth, seniors, couples, friends, and families
- Personalized travel experiences based on traveler profiles

### 💰 **Budget-Conscious Planning**
- Detailed cost breakdowns to help users stay within their budget
- Smart recommendations based on budget constraints

### 📅 **Date-Specific Itineraries**
- Comprehensive day-by-day schedules
- Considers user preferences and travel dates

### 💬 **Real-Time Interaction**
- AI assistant that understands travel needs deeply
- Provides timely and relevant recommendations
- Conversational interface for natural interaction

### 🔧 **Advanced AI Integration**
- Multiple AI model support (DialoGPT, BlenderBot, Flan-T5, Mistral 7B)
- Intelligent model selection and fallback mechanisms
- Real-time model availability testing

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Hugging Face account and API token
- Git (for cloning the repository)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/tripcraft-ai.git
   cd tripcraft-ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Hugging Face API Token**
   
   Create a `.streamlit/secrets.toml` file in your project directory:
   ```toml
   HUGGINGFACE_TOKEN = "hf_your_token_here"
   DEFAULT_MODEL = "microsoft/DialoGPT-large"
   DEFAULT_ENDPOINT_TYPE = "Inference API"
   MAX_RESPONSE_LENGTH = 200
   TEMPERATURE = 0.7
   TOP_P = 0.9
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:8501` to start using TripCraft AI!

## 🛠️ Configuration

### Hugging Face Setup

1. Create a free account at [Hugging Face](https://huggingface.co/)
2. Generate an API token from your [settings page](https://huggingface.co/settings/tokens)
3. Add the token to your `secrets.toml` file

### Available Models

TripCraft AI supports multiple AI models:

- **Microsoft DialoGPT Large** - Conversational AI optimized for dialogue
- **Facebook BlenderBot** - Advanced conversational model
- **Google Flan-T5 Large** - Text-to-text generation model
- **Mistral 7B Instruct** - Instruction-following language model
- **Code Llama Instruct** - Code-aware conversational model
- **Qwen 2.5** - Multilingual large language model

## 📱 Usage

1. **Start a Conversation**: Type your travel-related query in the chat input
2. **Specify Your Needs**: Mention your destination, budget, travel dates, and group type
3. **Get Recommendations**: Receive personalized itineraries and suggestions
4. **Refine Your Plans**: Continue the conversation to adjust and perfect your travel plans

### Example Queries

- "Plan a 5-day budget trip to Paris for a couple under $2000"
- "Create a family-friendly itinerary for Tokyo with kids aged 8-12"
- "Suggest senior-friendly activities in Rome for a 7-day trip"
- "Plan a backpacker's guide to Southeast Asia for 3 weeks"

## 🎯 Features Overview

### User Interface
- Clean, responsive chat interface
- Real-time message formatting
- Conversation history tracking
- Advanced settings panel

### AI Capabilities
- Multi-model support with intelligent switching
- Context-aware responses
- Error handling and recovery
- Performance optimization

### Travel Planning
- Demographic-specific recommendations
- Budget optimization
- Date-sensitive planning
- Cultural and local insights

## 🔧 Technical Architecture

```
TripCraft AI
├── Frontend (Streamlit)
│   ├── Chat Interface
│   ├── Model Selection
│   └── Settings Panel
├── Backend (Python)
│   ├── API Integration
│   ├── Response Processing
│   └── Error Handling
└── AI Models (Hugging Face)
    ├── DialoGPT
    ├── BlenderBot
    ├── Flan-T5
    └── Mistral 7B
```

## 📊 Advanced Settings

- **Max Response Length**: Control the length of AI responses (50-500 tokens)
- **Temperature**: Adjust response creativity (0.1-2.0)
- **Top P**: Fine-tune response diversity (0.1-1.0)
- **Model Testing**: Verify model availability before use

## 🤝 Contributing

We welcome contributions to TripCraft AI! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution

- Additional AI model integrations
- Enhanced travel recommendation algorithms
- UI/UX improvements
- Multi-language support
- Integration with travel APIs (hotels, flights, attractions)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Viswam.ai** - For providing the internship opportunity and technical guidance
- **Meta** - For AI research collaboration and resources
- **Swecha Telangana** - For community support and local insights
- **IIIT Hyderabad** - For academic partnership and technical expertise
- **Hugging Face** - For providing the AI model infrastructure
- **Streamlit** - For the amazing web app framework

## 🐛 Issues & Bug Reports

If you encounter any issues or bugs, please:

1. Check the [Issues](https://github.com/yourusername/tripcraft-ai/issues) page
2. Create a new issue with detailed description
3. Include error messages and steps to reproduce

## 🚀 Future Roadmap

- [ ] Integration with real-time travel APIs
- [ ] Multi-language support
- [ ] Mobile app development
- [ ] Advanced personalization features
- [ ] Social sharing capabilities
- [ ] Offline mode support
- [ ] Travel expense tracking
- [ ] Group trip planning features

---

**Made with ❤️ by the TripCraft AI Team**

*Transforming travel planning through intelligent AI assistance*
