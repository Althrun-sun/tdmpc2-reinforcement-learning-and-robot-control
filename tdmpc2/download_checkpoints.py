from huggingface_hub import hf_hub_download
import os
import shutil

# Parameters for downloading a compatible checkpoint
repo_id = "nicklashansen/tdmpc2"
filename = "dmcontrol/dog-run-1-compatible.pt"  # Replace with a compatible version if available

# Ensure the checkpoints directory exists
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Download the checkpoint
file_path = hf_hub_download(repo_id=repo_id, filename=filename)

# Move the file to the checkpoints directory
destination_path = os.path.join(checkpoint_dir, os.path.basename(filename))
if os.path.exists(destination_path):
    os.remove(destination_path)  # Remove any conflicting files
shutil.copy(file_path, destination_path)
print(f"Checkpoint downloaded and saved to: {destination_path}")


# python evaluate.py hydra.run.dir=. task=dog-run checkpoint=checkpoints/dog-run-1.pt save_video=true



