# tot_evaluation.py
import openai

openai.api_key = 'YOUR_API_KEY'

def evaluate_emotion_reasoning(text_emotion, audio_emotion, visual_emotion):
    """
    Evaluate emotion consistency using GPT-4.
    """
    prompt = f"""
    Consider these modality emotion predictions:
    - Text predicts: {text_emotion}
    - Audio predicts: {audio_emotion}
    - Visual predicts: {visual_emotion}

    Which emotion is most accurate given possible inconsistencies, and explain briefly.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=100,
    )
    reasoning = response.choices[0].message.content.strip()
    return reasoning
