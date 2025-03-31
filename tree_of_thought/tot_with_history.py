import openai
import json
import os
from datetime import datetime

OPENAI_API_KEY='KEY'
openai.api_key = os.getenv(OPENAI_API_KEY)

def build_tot_prompt(frame, history):
    history_str = "\n".join(
        f"{h['timestamp']}: Surface={h['surface']}, Underlying={h['underlying']}"
        for h in history
    )

    return f"""
You are an expert emotion reasoning agent.

Current multimodal inputs at {frame['timestamp']}:
- Text emotion scores: {frame['text']}
- Audio emotion scores: {frame['audio']}
- Visual emotion scores: {frame['visual']}

Past context:
{history_str or '[No prior context]'}

Your task is to determine:
1. The most externally visible emotion (surface)
2. The likely latent or regulated emotion (underlying)
3. Justify your reasoning

Respond in this format:
{{
  "surface_emotion": "<emotion>",
  "underlying_emotion": "<emotion>",
  "confidence_surface": <0-1>,
  "confidence_underlying": <0-1>,
  "justification": "<brief reasoning>"
}}
"""

def R_tot_with_history(frame, history, model='gpt-4', logfile='tot_results.jsonl'):
    prompt = build_tot_prompt(frame, history)
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    
    try:
        result = json.loads(response['choices'][0]['message']['content'])
    except Exception as e:
        result = {
            "surface_emotion": "unknown",
            "underlying_emotion": "unknown",
            "confidence_surface": 0.0,
            "confidence_underlying": 0.0,
            "justification": "Failed to parse LLM output.",
            "raw": response['choices'][0]['message']['content']
        }
    
    # Include metadata
    result.update({
        "timestamp": frame["timestamp"],
        "text_scores": frame["text"],
        "audio_scores": frame["audio"],
        "visual_scores": frame["visual"],
        "datetime_processed": datetime.now().isoformat()
    })

    # Save clearly to file
    with open(logfile, 'a') as f:
        json.dump(result, f)
        f.write('\n')

    return result