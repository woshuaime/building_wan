# Experiments

## Experiment: building_train_1912_all_256_cooled_lora

- Date：2026-09-04至2026-09-08（根据保留manifest和文件时间）
- Goal：在本地双RTX 3060上完成建筑领域LoRA训练链路。
- GPU：2×RTX 3060
- Environment：Windows旧环境 `D:\scj\envs\diffsynth\python.exe`
- Dataset：1912个源训练视频被切为9560个17帧片段
- Cache：旧Windows缓存路径，服务器未保留该缓存
- Base model：Wan2.1-T2V-1.3B
- Resolution：256×256
- Frames：17
- LoRA rank：8
- Learning rate：5e-5
- Epochs：1
- Save steps：500
- Output path：服务器保留副本 `checkpoints/building_train_1912_all_256_cooled_lora`
- Log path：`checkpoints/building_train_1912_all_256_cooled_lora/loss.csv`
- Start from：基础模型
- Resume：否
- Status：训练完成
- Result：完成4780个optimizer steps，末步loss为0.0214707479；保留step-500至step-4780等权重。用户对阶段视频的主观评价是视角覆盖不足、变化较少，因此不作为新A100实验起点。
- Notes：历史实验必须保留，不要删除或覆盖。

## Experiment: building_wan_a100_preparation_failures

- Date：2026-09-13
- Goal：验证服务器A100环境并生成384×384、81帧缓存。
- GPU：1×NVIDIA A100 80GB PCIe
- Environment：先后使用过 `mv2v_dual` 和 `mv2v`；最终环境确定为 `mv2v`
- Dataset：`building_v1` train，1912个视频
- Cache：旧版大缓存格式，后于2026-09-14按用户要求删除222份残缺文件
- Base model：Wan2.1-T2V-1.3B
- Resolution：384×384
- Frames：81
- LoRA rank：16
- Learning rate：2e-5（正式训练未开始）
- Epochs：8（正式训练未开始）
- Save steps：400（正式训练未开始）
- Output path：`checkpoints/building_wan`，没有生成权重
- Log path：`logs/slurm`
- Start from：基础模型
- Resume：否
- Status：失败，仅停留在缓存生成阶段
- Result：A100、CUDA和BF16可用得到确认，但没有训练结果。
- Notes：Job 1144/1145因DiffSynth导入路径失败；Job 1148因缺少librosa失败；Job 1149/1150并发写同一缓存，并在约210/221份时发生 `torch.save` iostream错误。禁止重复提交同一缓存目录。

## Experiment: building_wan

- Date：2026-09-14开始
- Goal：让Wan2.1学习一般建筑物外观、完整81帧旋转视角和时序一致性。
- GPU：1×A100 80GB
- Environment：`mv2v`；Python `/home/u2025171963/miniconda3/envs/mv2v/bin/python`
- Dataset：`data/splits/building_v1/metadata_train.csv`，1912个训练样本
- Cache：`cache/wan21_t2v_13b_building_384x384_f81_train`，格式标记目标为`compact-sft-shared-context-bf16-v2`；共享BF16 context实测约4.20MB，样本latent因Accelerate输出回转实际以FP32落盘，单份约3.10MB，完整预计约5.52GiB
- Base model：Wan2.1-T2V-1.3B
- Resolution：384×384
- Frames：81
- LoRA rank：16
- Learning rate：2e-5
- Epochs：8
- Save steps：400
- Output path：`checkpoints/building_wan`
- Log path：`logs/slurm/building_wan-<jobid>.out` 和 `.err`
- Start from：step 0 / epoch 0
- Resume：否；不使用旧双3060权重
- Status：训练完成；Job 1155于2026-09-14T23:20:08+08:00以`COMPLETED / 0:0`结束，总耗时13:04:29；推理验证待提交
- Result：完成15,296个optimizer steps，保存step-400至step-15296共39份LoRA；最终loss 0.0207207091，第1轮平均loss 0.033480，第8轮平均loss 0.027926。视觉效果待固定条件视频验证。
- Notes：Job 1153/1154曾重复提交后取消，均未留下缓存；正式结果来自Job 1155。已准备base、step-4000、step-8000、step-12000和step-15296的对比验证配置。

## Experiment: building_wan_a6000_single_orbit

- Date：2026-09-23准备
- Goal：在单张RTX A6000上先使用2344条单轨建筑视频和统一 prompt，以更大的物理batch训练建筑领域LoRA；逐栋 prompt 留作后续实验。
- GPU：目标为1×NVIDIA RTX A6000 48GB
- Resolution：384×384
- Frames：81
- Physical batch：2（可用 `WAN_TRAIN_BATCH_SIZE` 降为1）
- Gradient accumulation：2（默认有效batch为4）
- LoRA rank：16
- Learning rate：2e-5
- Epochs：8
- Split：`configs/single_orbit_same_prompt_v1`，训练1992、验证234、测试118，固定随机种子20260924，训练占85%。
- Prompt：本轮所有样本使用同一条统一建筑旋转 prompt；不使用尚未完全核对的 Cap3D 逐栋描述。
- Cache：新建 `compact-sft-shared-context-bf16-v2` 缓存，共享一份文本 context，物理batch由项目入口合并缓存样本。
- Output path：`checkpoints/building_wan_a6000_single_orbit`
- Start from：基础模型，独立输出，不resume A100或双3060权重。
- Status：统一 prompt 划分和脚本已准备；尚未执行Slurm提交，A6000实际峰值显存待短测确认。
