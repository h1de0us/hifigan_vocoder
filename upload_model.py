import os
import gdown

from src.utils import ROOT_PATH
from test import DEFAULT_CHECKPOINT_PATH

url = 'https://drive.google.com/uc?id=1ykEAj6gT8b0iX1rBI_7FDh275LznH9Ey'
output = str(ROOT_PATH / "checkpoint.pth")
gdown.download(url, output)