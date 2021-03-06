import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
