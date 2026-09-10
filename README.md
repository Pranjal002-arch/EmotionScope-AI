# 🧠 EmotionScope AI

### AI-Powered Emotion Detection & Risk Analysis Platform

EmotionScope AI is an interactive **Artificial Intelligence and Natural Language Processing (NLP)** application that analyzes user text and identifies emotional patterns using a **hybrid Machine Learning + rule-based approach**.

The application combines **TF-IDF, Logistic Regression, custom emotion rules, risk-pattern detection, voice interaction, and interactive data visualization** into a single Streamlit-based platform.

---

## 🚀 Overview

Understanding emotions from textual communication can be useful for conversational systems, sentiment analysis, user-support applications, and behavioral analytics.

EmotionScope AI processes user input and predicts one of **six emotions**:

- 😢 Sadness
- 😊 Joy
- ❤️ Love
- 😠 Anger
- 😨 Fear
- 😲 Surprise

Instead of relying only on a Machine Learning model, the system uses a **hybrid prediction architecture** that combines ML probabilities with a custom rule-based emotion engine.

The application also provides:

- Emotion confidence analysis
- Risk-pattern detection
- Historical emotion tracking
- Interactive visual analytics
- Voice input
- Text-to-speech responses
- Responsive Streamlit interface

> **Note:** EmotionScope AI is an experimental/educational AI system and is not a medical or psychological diagnostic tool.

---

# ✨ Key Features

## 🧠 Hybrid Emotion Detection

EmotionScope AI combines two approaches:

### Machine Learning

The ML pipeline uses:

- TF-IDF for text feature extraction
- Logistic Regression for emotion classification
- Probability-based prediction

### Rule-Based Analysis

A custom keyword/rule engine analyzes emotional expressions and contextual indicators.

The two outputs are combined to produce the final prediction.

Conceptually:

```text
Final Score =
    α × ML Score
    +
    (1 − α) × Rule Score
