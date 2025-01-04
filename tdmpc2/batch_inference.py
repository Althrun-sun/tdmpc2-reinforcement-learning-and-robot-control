import os
from collections import defaultdict

# 推理函数
def run_inference(checkpoint_dir, mode="minimal"):
    """
    批量推理函数。
    
    参数:
    - checkpoint_dir: 本地存储权重文件的目录。
    - mode: 推理模式，"minimal" 或 "all"。
    """
    # 检查模式是否正确
    if mode not in {"minimal", "all"}:
        raise ValueError("Mode must be either 'minimal' or 'all'.")

    # 收集所有的权重文件
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    task_files = defaultdict(list)

    # 分组文件 (任务 -> 对应的权重文件)
    for checkpoint in checkpoints:
        # 删除 "-<数字>" 部分以生成 task 名称
        task_name = "-".join(checkpoint.split("-")[:-1])
        task_files[task_name].append(checkpoint)

    # 遍历任务并运行推理
    for task, files in task_files.items():
        # 按文件名排序，确保处理顺序一致
        files = sorted(files)

        if mode == "minimal":
            # 仅选择第一个权重文件
            files = [files[0]]

        print(f"Running inference for task: {task}")
        for checkpoint in files:
            # 构建命令
            command = f"python evaluate.py hydra.run.dir=. task={task} checkpoint={os.path.join(checkpoint_dir, checkpoint)} save_video=true"
            print(f"Executing: {command}")
            os.system(command)  # 执行推理

# 设置参数
checkpoint_directory = "checkpoints"  # 权重文件目录
inference_mode = "minimal"  # 推理模式 ("minimal" 或 "all")

# 运行推理
run_inference(checkpoint_directory, mode=inference_mode)
