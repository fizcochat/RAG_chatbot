# Multilingual Support Documentation

## Overview

Fiscozen Chatbot now supports both Italian and English, making it accessible to international clients while maintaining its specialized Italian tax knowledge. This document explains how the multilingual functionality works and how to use it effectively.

## How It Works

The multilingual system employs a translation layer that preserves the chatbot's specialized Italian tax expertise while providing a seamless English interface:

1. **Under the Hood**: The core knowledge base and FastText topic filter remain in Italian, preserving domain expertise
2. **Translation Layer**: Queries and responses are translated in real-time using GPT-4
3. **Seamless Experience**: Users interact naturally in their preferred language without any indication of translation

## Technical Implementation

### Translation Mechanism

The translation is powered by OpenAI's GPT-4 model with specific system prompts to maintain tone and context:

```python
# English to Italian translation for queries
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a professional translator. Translate the following text from English to Italian, maintaining the same tone and style."},
        {"role": "user", "content": text}
    ],
    temperature=0.3
)
```

### Data Flow

When a user interacts in English:

1. User submits an English query
2. Query is translated to Italian
3. The Italian query is processed by FastText for topic relevance
4. If relevant, the query is sent to the RAG system
5. The Italian response is translated back to English
6. The English response is presented to the user

### Language Selection

- Language selection is managed through a dropdown in the sidebar
- The selected language is stored in the Streamlit session state: `st.session_state.language`
- All UI elements adapt to the selected language

## User Experience

### Language Switching

Users can switch languages at any time using the dropdown in the sidebar:

1. Open the sidebar on the left
2. Select your preferred language from the dropdown
3. The UI will immediately update to the selected language
4. All future interactions will be in the selected language

### UI Adaptations

The following elements adapt to the selected language:

- Welcome messages and descriptions
- Button labels and UI text
- Suggested questions
- Input field placeholders
- Error messages

## Design Decisions

### Why This Approach?

We chose to translate rather than build separate knowledge bases for several reasons:

1. **Knowledge Integrity**: The specialized Italian tax information remains intact
2. **Maintenance Efficiency**: Only one knowledge base to maintain and update
3. **Consistency**: Users get identical information regardless of language
4. **Scalability**: Makes it easier to add additional languages in the future

### Performance Considerations

- Translations add a slight delay to processing (typically <1 second)
- GPT-4 ensures high-quality translations that preserve meaning and context
- The system is optimized to minimize API calls

## Limitations and Edge Cases

- **Highly Technical Terms**: Some specialized Italian tax terms may not have direct English equivalents
- **Cultural Context**: Some Italian-specific tax concepts may require additional explanation in English
- **Multiple Languages in One Query**: The system works best when a single language is used consistently

## Future Enhancements

- Support for additional languages beyond Italian and English
- Cached translations to improve performance
- Language-specific knowledge bases for frequently asked questions

## Troubleshooting

If you encounter issues with the translation feature:

- **Incorrect Translations**: Ensure your OpenAI API key is valid and has sufficient quota
- **Language Not Switching**: Try clearing your browser cache or refreshing the page
- **Missing UI Elements**: Ensure you're using the latest version of the application 