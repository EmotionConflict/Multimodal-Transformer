import torch
import sys
import os

sys.path.append(os.path.abspath('.'))

from src.models import MULTModel

device = torch.device('cpu')

# Load your original model (now loads without errors)
model = torch.load('./pre_trained_models/mult_MULT.pt', map_location=device)
model.eval()

# Save correctly as a state_dict
torch.save(model.state_dict(), './pre_trained_models/mult_MULT_state_dict.pt')
print("Saved clean state_dict checkpoint!")
