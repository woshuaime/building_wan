# Training

本文档记录当前实验室服务器上的 A6000 单轨训练、续训和验证方法。当前项目根目录为 /mnt/windowsE/chuanjun/building_wan，环境为 scj。

## 1. 环境和路径

服务器 Python：

    /home/shi/miniconda3/envs/scj/bin/python

检查环境：

    /home/shi/miniconda3/envs/scj/bin/python --version
    /home/shi/miniconda3/envs/scj/bin/python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
    /home/shi/miniconda3/envs/scj/bin/python -m pip check

当前 GPU 应为 1 张 NVIDIA RTX A6000。登录节点没有 GPU 时，torch.cuda.is_available() 为 False 不代表训练节点不可用。

## 2. A6000 单轨配置

训练脚本：

    scripts/train_wan81_a6000.sh

续训脚本：

    scripts/resume_wan81_a6000.sh

验证脚本：

    scripts/validate_wan81_a6000.py

默认参数：

    数据：data/single_orbit_videos
    清单：configs/single_orbit_same_prompt_v1/metadata_train.csv
    划分：训练1992、验证234、测试118
    分辨率：384×384
    帧数：81
    物理 batch：2
    梯度累积：2
    有效 batch：4
    LoRA rank：16
    训练轮数：8
    dataset_num_workers：0
    原输出：checkpoints/building_wan_a6000_single_orbit

统一 prompt 训练会使用一份共享文本 context。逐栋 prompt 必须使用新的缓存设计，不能复用本轮缓存。

## 3. 只读检查

    cd /mnt/windowsE/chuanjun/building_wan
    bash -n scripts/train_wan81_a6000.sh scripts/resume_wan81_a6000.sh
    /home/shi/miniconda3/envs/scj/bin/python -m py_compile code/Python_3D_Scanner/training/wan_train_entry.py scripts/validate_wan81_a6000.py
    find cache/wan21_t2v_13b_single_orbit_same_prompt_384x384_f81_train -type f -name '*.pth' | wc -l
    ps -ef | grep wan_train_entry.py | grep -v grep || true

缓存完整时应为 1992 个 .pth，并同时存在共享文本 context 文件。

## 4. 原训练故障

原训练曾在约第 2742 个 batch 退出：

    free(): invalid next size (normal)
    中止（核心转储）

这不是显存不足。完整缓存仍可复用，step-2400.safetensors 已确认可读。最可能原因是多进程数据读取触发 native 内存释放问题，因此当前训练和续训都默认 dataset_num_workers=0。

曾出现的 checkpoint 报错：

    600 keys are unexpected

原因是把 LoRA 权重传给了 --resume_from_checkpoint。LoRA 续训必须使用 --lora_checkpoint；--resume_from_checkpoint 只适合完整训练状态。

## 5. 正确续训

不要删除或覆盖原输出。建议先进入 tmux：

    tmux new -s wan-resume
    cd /mnt/windowsE/chuanjun/building_wan
    WAN_PYTHON=/home/shi/miniconda3/envs/scj/bin/python bash scripts/resume_wan81_a6000.sh 2>&1 | tee logs/a6000_single_orbit_resume_$(date +%Y%m%d_%H%M%S).log

脚本默认：

- 从 checkpoints/building_wan_a6000_single_orbit/step-2400.safetensors 读取 LoRA。
- 跳过前 2400 个缓存 batch。
- 把初始全局 step 设为 2400。
- 物理 batch 2、梯度累积 2、worker 0。
- 把新权重写入 checkpoints/building_wan_a6000_single_orbit_resumed。

如果使用其他 checkpoint，必须同时设置：

    WAN_RESUME_FROM_CHECKPOINT=/path/to/step-N.safetensors WAN_RESUME_SKIP_BATCHES=N WAN_RESUME_INITIAL_STEPS=N WAN_PYTHON=/home/shi/miniconda3/envs/scj/bin/python bash scripts/resume_wan81_a6000.sh

查看训练：

    tmux attach -t wan-resume
    tail -F logs/a6000_single_orbit_resume_*.log
    watch -n 2 nvidia-smi

断开 tmux 使用 Ctrl+B，再按 D；不要用 Ctrl+C，除非要停止训练。

## 6. 续训完成后的验证

先只读预检：

    cd /mnt/windowsE/chuanjun/building_wan
    /home/shi/miniconda3/envs/scj/bin/python scripts/validate_wan81_a6000.py --checkpoint-dir checkpoints/building_wan_a6000_single_orbit_resumed

预检通过且确认要生成视频后：

    /home/shi/miniconda3/envs/scj/bin/python scripts/validate_wan81_a6000.py --checkpoint-dir checkpoints/building_wan_a6000_single_orbit_resumed --run

结果目录类似：

    outputs/validation/building_wan_a6000_single_orbit_resumed/step-N

脚本比较基础模型和最新 LoRA 的固定 prompt、seed、384×384、81 帧视频。验证集用于路径和 prompt 一致性检查，不会自动给出建筑质量分数；需要人工检查体块、环绕完整性、几何变形、闪烁和时序稳定性。

## 7. 状态和排错

    ps -ef | grep wan_train_entry.py | grep -v grep
    du -sh cache/wan21_t2v_13b_single_orbit_same_prompt_384x384_f81_train
    find checkpoints/building_wan_a6000_single_orbit_resumed -maxdepth 1 -name '*.safetensors' -printf '%f\n' | sort -V | tail
    tail -n 100 logs/a6000_single_orbit_resume_*.log

NVML 初始化警告、黄色感叹号和短时低 GPU 利用率不一定是错误；优先检查 traceback、进程是否仍在、loss.csv 是否增长和 checkpoint 是否保存。不要在原因未确认时重复启动第二个训练进程。

## 8. A100 历史命令

A100 的 /home/share/CHUANJUN/building_wan 和 mv2v 流程已完成并保留在历史文档中。当前 A6000 训练不要使用 A100 的脚本、环境或输出目录。
