# 🤖 Emotion-Aware Study Buddy 📚

An **AI-powered Streamlit application** that detects a student’s emotions and study level, then provides **personalized, human-like study guidance**.

Built using **Streamlit** and **Hugging Face Transformers**, this app acts as a supportive study companion that adapts to how you feel.

---

## 🌟 Features

- 🧠 **Emotion Detection**
  - Detects the **top 3 emotions** from user input
  - Uses a pretrained NLP model for multi-emotion classification

- 🎯 **Study Level Identification**
  - Automatically classifies users as **Beginner, Intermediate, or Advanced**

- 📚 **Personalized Study Guidance**
  - Emotion-based motivational tips
  - Study strategies tailored to experience level
  - Exam preparation techniques
  - Health & productivity advice

- 💬 **Human-like Interaction**
  - Friendly and supportive responses
  - Helps reduce anxiety and improve focus

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **Streamlit**
- **Hugging Face Transformers**
- **DistilRoBERTa Emotion Model**
  - `j-hartmann/emotion-english-distilroberta-base`

---

## 📂 Project Structure
emotion-aware-studdy-buddy/
│
├── app.py
├── requirements.txt
├── README.md
