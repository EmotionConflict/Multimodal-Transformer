import sys
import os
import torch
import openai
import json
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import MULTModel
from src.dataset import Multimodal_Datasets
from torch.utils.data import DataLoader
from tree_of_thought.tot_utils import build_tot_prompt

# Set your OpenAI API key explicitly here:
openai.api_key = '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'  # <-- Replace explicitly with your actual API key

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load your model explicitly (unchanged):
model_path = './pre_trained_models/mult_MULT.pt'
model = torch.load(model_path, map_location=device, weights_only=False)
model.to(device)
model.eval()

# Prepare dataset explicitly (unchanged):
test_dataset = Multimodal_Datasets(
    dataset_path='./data',
    data='iemocap',
    split_type='test',
    if_align=False
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Function to predict emotions with GPT-4 (explicitly fixed for API key and logging):
def predict_with_tot(frame, history, logfile='tot_results.jsonl'):
    prompt = build_tot_prompt(frame, history)

    response = openai.ChatCompletion.create(
        model='gpt-4-turbo',
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    raw_content = response['choices'][0]['message']['content']

    # Direct JSON parsing explicitly:
    try:
        result = json.loads(raw_content)
    except Exception as e:
        result = {
            "surface_emotion": "unknown",
            "underlying_emotion": "unknown",
            "confidence_surface": 0.0,
            "confidence_underlying": 0.0,
            "justification": "Failed to parse LLM output.",
            "raw": raw_content
        }

    result.update({
        "timestamp": str(frame["timestamp"]),
        "text_scores": {k: v if isinstance(v, list) else v.tolist() for k, v in frame["text"].items()},
        "audio_scores": {k: v if isinstance(v, list) else v.tolist() for k, v in frame["audio"].items()},
        "visual_scores": {k: v if isinstance(v, list) else v.tolist() for k, v in frame["visual"].items()},
        "datetime_processed": datetime.now().isoformat()
    })

    with open(logfile, 'a') as f:
        json.dump(result, f)
        f.write('\n')

    return result



# Run the inference explicitly and update history:
history = []

for idx, (X, Y, META) in enumerate(test_loader):
    _, text, audio, vision = X

    text, audio, vision = text.to(device), audio.to(device), vision.to(device)
    with torch.no_grad():
        preds, _ = model(text, audio, vision)
        logits = preds.cpu().numpy().tolist()

    # Prepare frame explicitly (fixed structure):
    # Prepare frame explicitly (ensure tensor-to-list):
    frame = {
        "timestamp": str(META[0]),
        "text": {"emotion_logits": preds.cpu().tolist()},
        "audio": {"emotion_logits": preds.cpu().tolist()},
        "visual": {"emotion_logits": preds.cpu().tolist()}
    }


    # Call explicitly the prediction with ToT:
    result = predict_with_tot(frame, history)

    # Update history explicitly for next iterations:
    history.append({
        "timestamp": frame["timestamp"],
        "surface": result["surface_emotion"],
        "underlying": result["underlying_emotion"]
    })

    # Print explicitly the result:
    print(json.dumps(result, indent=2))

    # Demonstration: explicitly break after first inference:
    break
