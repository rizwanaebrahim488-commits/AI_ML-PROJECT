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
st.write("Focus on how you feel, get personalized study tips & exam strategies!")

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
# Functions
# -------------------------
def detect_emotions(text):
    results = emotion_classifier(text)[0]
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:3]  # top 3 emotions

def detect_study_level(text):
    text = text.lower()
    if any(w in text for w in ["don't understand", "confused", "new topic", "hard", "struggling"]):
        return "Beginner"
    elif any(w in text for w in ["understand", "revision", "practice", "somewhat"]):
        return "Intermediate"
    elif any(w in text for w in ["easy", "confident", "strong", "good", "mastered"]):
        return "Advanced"
    else:
        return "Intermediate"

def generate_guidance(top_emotions, level):
    # Multi-emotion tips
    emotion_tips = {
        "anger": "Feeling frustrated? Take 5 min break, do breathing exercises, and approach tasks calmly.",
        "sadness": "Feeling low? Break tasks into small goals, reward yourself for achievements, stay consistent.",
        "fear": "Feeling anxious? Make a structured plan, tackle one topic at a time to reduce stress.",
        "joy": "Feeling good? Great time to revise difficult topics and practice actively.",
        "neutral": "Feeling stable? Stick to your routine, maintain focus and discipline.",
        "surprise": "Unexpected challenge? Start from basics and gradually increase difficulty."
    }

    # Multi-emotion tip
    tips = [emotion_tips.get(e["label"], "") for e in top_emotions]

    # Study level plans
    level_plan = {
        "Beginner": """
📘 **Beginner Strategy**
- Study 25 min + 5 min break (Pomodoro method)
- Explain aloud (Feynman technique)
- Take handwritten notes
- Focus on one topic at a time
""",
        "Intermediate": """
📗 **Intermediate Strategy**
- Study 50 min + 10 min break
- Active recall (write answers from memory)
- Daily revision
- Solve practice questions
""",
        "Advanced": """
📕 **Advanced Strategy**
- Study 90 min deep work
- Timed exam simulations
- Keep error notebook
- Teach someone else to reinforce learning
"""
    }

    # Exam tips
    exam_tips = """
📌 **Effective Exam Tips**
- Review summaries & mind maps before sleeping
- Focus on high-yield topics first
- Practice past papers under timed conditions
- Use active recall instead of rereading
- Teach someone else to reinforce memory
- Take short breaks to stay alert
- Stay hydrated and sleep well
"""

    discipline = """
💪 **Discipline & Health Tips**
- Study at the same time daily ⏰
- Avoid phone and distractions during study 📵
- Take short movement breaks (walk/stretch) 🚶
- Drink water regularly 💧
- Sleep 7–8 hours 🛌
"""

    return tips, level_plan.get(level), exam_tips, discipline

# -------------------------
# Streamlit UI
# -------------------------
st.subheader("How do you feel right now about your studies?")
user_input = st.text_area(
    "Describe your current emotions, focus, or study challenges:",
    placeholder="Example: I'm stressed and struggling to focus on revision."
)

if st.button("📊 Get Personalized Study Guidance"):
    if user_input.strip():
        # Detect emotions
        top_emotions = detect_emotions(user_input)
        emotion_labels = [e["label"] for e in top_emotions]
        emotion_scores = [round(e["score"], 2) for e in top_emotions]

        # Detect study level
        level = detect_study_level(user_input)

        # Generate guidance
        tips, level_plan_text, exam_text, discipline_text = generate_guidance(top_emotions, level)

        # Display results
        st.subheader("🧠 Detected Emotions (Top 3)")
        for e, s in zip(emotion_labels, emotion_scores):
            st.write(f"- {e} ({s*100}%)")

        st.subheader("🎯 Emotion-Based Advice")
        for tip in tips:
            st.info(tip)

        st.subheader("📚 Personalized Study Plan")
        st.markdown(level_plan_text)

        st.subheader("📝 Exam Tips & Tricks")
        st.markdown(exam_text)

        st.subheader("🧘 Discipline & Health Tips")
        st.markdown(discipline_text)

    else:
        st.warning("Please enter your current feelings or study challenges.")
