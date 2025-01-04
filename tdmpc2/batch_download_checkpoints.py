from huggingface_hub import list_repo_files, hf_hub_download
import os
import shutil

# 参数
repo_id = "nicklashansen/tdmpc2"
base_dir = "dmcontrol"
checkpoint_dir = "checkpoints"

# 确保检查点目录存在
os.makedirs(checkpoint_dir, exist_ok=True)

# 列出仓库中的所有文件
try:
    files = list_repo_files(repo_id=repo_id)
    # 过滤出目标目录下的文件
    filtered_files = [f for f in files if f.startswith(base_dir)]
    print(f"Found {len(filtered_files)} files under directory '{base_dir}':")
    for file in filtered_files:
        print(file)

    # 下载文件
    for file in filtered_files:
        print(f"\nDownloading: {file}")
        try:
            file_path = hf_hub_download(repo_id=repo_id, filename=file)
            destination_path = os.path.join(checkpoint_dir, os.path.basename(file))
            if os.path.exists(destination_path):
                os.remove(destination_path)  # 如果文件存在，则删除
            shutil.copy(file_path, destination_path)
            print(f"File saved to: {destination_path}")
        except Exception as e:
            print(f"Error downloading {file}: {e}")
except Exception as e:
    print(f"Error listing files: {e}")
