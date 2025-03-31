import json
from datetime import datetime

def build_tot_prompt(frame, history):
    history_str = "\n".join(
        f"{h['timestamp']}: Surface={h['surface']}, Underlying={h['underlying']}"
        for h in history
    )

    prompt = f"""
You are an expert emotion reasoning agent.

Current multimodal inputs at {frame['timestamp']}:
- Text emotion scores: {frame['text']}
- Audio emotion scores: {frame['audio']}
- Visual emotion scores: {frame['visual']}

Past context:
{history_str or '[No prior context]'}

Your task is to determine:
1. surface_emotion: The most externally visible emotion, The emotion with the greatest overall possibility across modalities.
2. underlying_emotion: The likely latent or regulated emotion, The emotion with the second greatest overall possibility across modalities.
3. Justify your reasoning


Respond ONLY in JSON format (no other explanations or text). Follow exactly this structure:

{{
  "surface_emotion": "<emotion>",
  "underlying_emotion": "<emotion>",
  "confidence_surface": <0-1>,
  "confidence_underlying": <0-1>,
  "justification": "<brief reasoning>"
}}
"""
    return prompt.strip()

