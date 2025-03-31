import openai
import json
import os

OPENAI_API_KEY='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'
openai.api_key = os.getenv(OPENAI_API_KEY)

def build_tot_prompt(frame, history):
    history_str = "\n".join(
        f"{h['timestamp']}: Surface={h['surface_emotion']}, Underlying={h['underlying_emotion']}"
        for h in history
    )

    prompt = f"""
You are an expert emotion reasoning agent analyzing multimodal data (text, audio, visual).

Current frame ({frame['timestamp']}):
- Text emotion probabilities: {frame['text']}
- Audio emotion probabilities: {frame['audio']}
- Visual emotion probabilities: {frame['visual']}

Historical emotion trajectory:
{history_str if history else "[No prior context]"}

Please determine clearly:
1. Surface emotion (observable)
2. Underlying emotion (latent/suppressed)
3. Confidence (0-1 for each emotion)
4. Brief justification

Respond strictly in JSON format:

{{
  "surface_emotion": "<emotion>",
  "underlying_emotion": "<emotion>",
  "confidence_surface": <float>,
  "confidence_underlying": <float>,
  "justification": "<concise reasoning>"
}}
"""
    return prompt.strip()

def R_tot_with_history(frame, history, model='gpt-4-turbo'):
    prompt = build_tot_prompt(frame, history)
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200
    )

    content = response.choices[0].message.content.strip()
    try:
        return json.loads(content)
    except:
        return {
            "surface_emotion": "unknown",
            "underlying_emotion": "unknown",
            "confidence_surface": 0.0,
            "confidence_underlying": 0.0,
            "justification": "Parsing failed",
            "raw_response": content
        }
