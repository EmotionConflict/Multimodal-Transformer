# tot_generation.py
import torch

def generate_top_emotions(logits, top_k=3):
    """
    Generates top-k candidate emotions from model logits clearly.
    Args:
        logits (Tensor): shape (batch_size, num_emotions)
        top_k (int): number of emotions to generate clearly
    Returns:
        tuple: indices and probabilities of top-k emotions clearly
    """
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)
    return top_indices.cpu().numpy(), top_probs.cpu().detach().numpy()
