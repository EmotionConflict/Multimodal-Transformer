# tot_main.py
import torch
from modules.transformer import CrossModalTransformer
from src.dataset import MultimodalDataset
from torch.utils.data import DataLoader
from tree_of_thought.tot_generation import generate_top_emotions
from tree_of_thought.tot_search import bfs_emotion_search

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load trained MulT model
model = CrossModalTransformer(embed_dim=512, num_heads=8, num_layers=4).to(device)
model.load_state_dict(torch.load('./pre_trained_models/model_iemocap.pt'))
model.eval()

test_dataset = MultimodalDataset('./data/IEMOCAP', split='test')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

def predict_with_tot(text, audio, vision):
    with torch.no_grad():
        text_logits = model(text.to(device))
        audio_logits = model(audio.to(device))
        vision_logits = model(vision.to(device))

    top_text, _ = generate_top_emotions(text_logits, top_k=2)
    top_audio, _ = generate_top_emotions(audio_logits, top_k=2)
    top_vision, _ = generate_top_emotions(vision_logits, top_k=2)

    final_emotion, reasoning = bfs_emotion_search(top_text[0], top_audio[0], top_vision[0])

    print("\nFinal Predicted Emotions (Text, Audio, Vision):", final_emotion)
    print("Reasoning:", reasoning)

# Example inference
for text, audio, vision, label in test_loader:
    predict_with_tot(text, audio, vision)
    break  # for demonstration, remove break in full run
