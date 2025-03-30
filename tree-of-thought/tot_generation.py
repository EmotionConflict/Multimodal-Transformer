# tot_generation.py
import torch

def generate_top_emotions(logits, top_k=3):
    """
    Generates top-k candidate emotions from model logits.
    """
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)
    return top_indices.cpu().numpy(), top_probs.cpu().detach().numpy()
