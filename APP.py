import streamlit as st
from transformers import pipeline

# -------------------------
# Streamlit Page Config #/human
# -------------------------
st.set_page_config(
    page_title="Emotion-Aware Study Buddy",
    page_icon="🤖📚",
    layout="wide"
)

# -------------------------
# Sidebar for User Info #/human
# -------------------------
st.sidebar.title("🧠 Your Study Buddy")
st.sidebar.write("Adjust your current mood and study focus:")

# Sidebar input for emotions #/human
mood_input = st.sidebar.text_input(
    "How are you feeling right now?",
    placeholder="Ex: stressed, confident, confused..."
)

# Sidebar input for optional context #/human
study_context = st.sidebar.text_area(
    "Describe your study situation or challenges:",
    placeholder="Ex: I have an exam tomorrow, and I feel unprepared."
)

# -------------------------
# Load Emotion Detection Model #/human
# -------------------------
@st.cache_resource
def load_emotion_model():
    # Using pre-trained model for multi-emotion detection
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )

emotion_classifier = load_emotion_model()

# -------------------------
# Functions for Logic #/human
# -------------------------
def detect_emotions(text):
    """Detect top 3 emotions from user text"""
    results = emotion_classifier(text)[0]
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:3]

def study_level(text):
    """Determine user's study level"""
    text = text.lower()
    if any(w in text for w in ["confused", "struggling", "new topic", "hard"]):
        return "Beginner"
    elif any(w in text for w in ["practice", "revision", "somewhat"]):
        return "Intermediate"
    elif any(w in text for w in ["confident", "mastered", "easy"]):
        return "Advanced"
    else:
        return "Intermediate"

def generate_response(emotions, level):
    """Create human-like study guidance based on emotions and level"""
    # -------------------------
    # Human-like Emotion Tips #/human
    # -------------------------
    emotion_advice = {
        "anger": "I see you are frustrated 😤. Take a 5-min break and breathe. Small steps work better.",
        "fear": "Feeling anxious? 😟 Let's tackle one topic at a time and build confidence.",
        "joy": "You're in a good mood! 😄 Perfect for revising tough topics or practicing exercises.",
        "sadness": "Low energy? 😔 Break your tasks into smaller goals and reward yourself.",
        "neutral": "Feeling steady 🙂. Stick to your study plan and stay consistent.",
        "surprise": "Unexpected challenges? 😲 Start from basics and slowly build up."
    }

    tips = [emotion_advice.get(e["label"], "") for e in emotions]

    # -------------------------
    # Human-like Study Level Tips #/human
    # -------------------------
    level_tips = {
        "Beginner": "- Study 25 min + 5 min break (Pomodoro)\n- Take handwritten notes\n- Focus on one topic at a time",
        "Intermediate": "- Study 50 min + 10 min break\n- Active recall (write answers from memory)\n- Daily revision & practice questions",
        "Advanced": "- Study 90 min deep work sessions\n- Timed exam simulations\n- Teach someone else to reinforce memory"
    }

    # -------------------------
    # Exam Tips & Tricks #/human
    # -------------------------
    exam_tips = (
        "- Use past papers for timed practice\n"
        "- Focus on high-yield topics\n"
        "- Teach a friend to test understanding\n"
        "- Spaced repetition for long-term memory\n"
        "- Take short breaks to stay alert\n"
        "- Stay hydrated and sleep well"
    )

    # -------------------------
    # Health & Productivity Tips #/human
    # -------------------------
    health_tips = (
        "- Keep a consistent study schedule\n"
        "- Avoid phone distractions during study\n"
        "- Stretch or walk during breaks\n"
        "- Drink water regularly\n"
        "- Sleep 7-8 hours each night"
    )

    return tips, level_tips.get(level, ""), exam_tips, health_tips

# -------------------------
# Main Chat Interface #/human
# -------------------------
st.title("🤖 Human-Like Emotion-Aware Study Buddy Chat")

if st.button("Get Guidance"):
    if not mood_input.strip() and not study_context.strip():
        st.warning("Please enter how you feel or your study context to get guidance.")
    else:
        # Combine mood and context
        user_text = f"{mood_input} {study_context}"

        # Emotion Detection
        top_emotions = detect_emotions(user_text)
        emotion_labels = [e["label"] for e in top_emotions]
        emotion_scores = [round(e["score"], 2) for e in top_emotions]

        # Detect study level
        level = study_level(user_text)

        # Generate response
        emotion_response, level_response, exam_response, health_response = generate_response(top_emotions, level)

        # -------------------------
        # Display Chat-Like Responses #/human
        # -------------------------
        st.subheader("🧠 Detected Emotions (Top 3)")
        for e, s in zip(emotion_labels, emotion_scores):
            st.write(f"- {e} ({s*100}%)")

        st.subheader("🎯 Emotion-Based Advice")
        for tip in emotion_response:
            st.info(tip)

        st.subheader("📚 Study Level Recommendations")
        st.markdown(level_response)

        st.subheader("📝 Exam Tips & Tricks")
        st.markdown(exam_response)

        st.subheader("🧘 Health & Productivity")
        st.markdown(health_response)
