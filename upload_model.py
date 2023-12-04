import os
import gdown


url = 'TODO'
save_dir = 'saved/models/final_run'
os.makedirs(save_dir, exist_ok=True)
output = save_dir + '/weights.pth'
gdown.download(url, output)