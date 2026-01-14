import streamlit as st
from transformers import pipeline

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Emotion-Aware Study Buddy",
    page_icon="📚",
    layout="centered"
)

st.title("🎓 Emotion-Aware AI Study Buddy")
st.write("Your smart study coach for emotions, discipline, memory & exams")

# -------------------------
# Load Emotion Model (cached)
# -------------------------
@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )

emotion_classifier = load_emotion_model()

# -------------------------
# Core Functions
# -------------------------
def detect_emotion(text):
    emotions = emotion_classifier(text)[0]
    return max(emotions, key=lambda x: x["score"])["label"]

def detect_study_level(text):
    text = text.lower()
    if any(w in text for w in ["don't understand", "confused", "new topic", "very hard"]):
        return "Beginner"
    elif any(w in text for w in ["understand", "revision", "practice"]):
        return "Intermediate"
    elif any(w in text for w in ["easy", "confident", "strong"]):
        return "Advanced"
    else:
        return "Intermediate"

def generate_guidance(emotion, level):
    emotion_tip = {
        "anger": "Frustration blocks learning. Pause, breathe deeply for 30 seconds.",
        "sadness": "Low mood detected. Focus on consistency, not perfection.",
        "fear": "Anxiety means this matters. Structure will reduce it.",
        "joy": "Great mood! Ideal time for difficult concepts.",
        "neutral": "Stable focus mode. Maintain discipline.",
        "surprise": "Unexpected difficulty. Start with basics."
    }

    level_plan = {
        "Beginner": """
📘 **Beginner Strategy**
• 25 min study + 5 min break
• Read → explain aloud (Feynman technique)
• Write notes by hand
• One topic at a time
""",
        "Intermediate": """
📗 **Intermediate Strategy**
• 50 min study + 10 min break
• Active recall (close book, write answers)
• Daily revision
• Practice questions
""",
        "Advanced": """
📕 **Advanced Strategy**
• 90 min deep work
• Timed exam simulations
• Error notebook
• Teach someone else
"""
    }

    discipline = """
💪 **Discipline Rules**
• No phone during study 📵
• Study same time daily ⏰
• Break = movement (walk/stretch) 🚶
• Drink water every 30–45 min 💧
• Sleep 7–8 hours 🛌
"""

    return emotion_tip.get(emotion), level_plan.get(level), discipline

def exam_plan(days):
    return f"""
📅 **{days}-Day Exam Study Plan**

Daily Routine:
• Morning: Learn new topics (2–3 hrs)
• Afternoon: Practice + recall (2 hrs)
• Evening: Revision + weak areas (1–1.5 hrs)

Rules:
• Revise each topic 3 times
• Every 6th day → light study + rest
• Last 2 days → revision only

Memory Techniques:
• Spaced repetition
• Flashcards
• Mind maps
• Active recall
"""

# -------------------------
# UI
# -------------------------
user_input = st.text_area(
    "How are you feeling about your studies?",
    placeholder="Example: I'm stressed and I don't understand physics"
)

days = st.number_input(
    "Days left for exam (optional)",
    min_value=0,
    max_value=365,
    step=1
)

if st.button("📊 Get Study Guidance"):
    if user_input.strip():
        emotion = detect_emotion(user_input)
        level = detect_study_level(user_input)

        emotion_tip, level_plan, discipline = generate_guidance(emotion, level)

        st.subheader("🧠 Analysis")
        st.write(f"**Detected Emotion:** {emotion}")
        st.write(f"**Study Level:** {level}")

        st.subheader("🎯 Emotion-Based Advice")
        st.info(emotion_tip)

        st.subheader("📚 Personalized Study Plan")
        st.markdown(level_plan)

        st.subheader("🧘 Discipline & Health")
        st.markdown(discipline)

        if days > 0:
            st.subheader("📅 Exam Preparation Plan")
            st.markdown(exam_plan(days))
    else:
        st.warning("Please type your study concern.")

