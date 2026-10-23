# 历史T2V：双3060与A100阶段

本目录仅归拢历史记录，不把旧环境当作当前服务器命令。

- 双RTX3060阶段：building_train_1912_all_256_cooled_lora，256×256、17帧、rank8、1 epoch；该权重目录当前不在服务器。
- A100 building_wan：历史1912个building_v1训练样本、384×384、81帧、rank16；历史记录为15296更新，当前 [checkpoints/building_wan](../../../checkpoints/building_wan/) 保留39份LoRA。
- [保留的loss图](runs/building_wan/)，原位置 outputs/validation/building_wan/。
- [早期比较视频](runs/comparison_0928/)，原位置 outputs/validation/9.28/。
- 详细来源与故障过程见 [历史台账](../../../docs/history/20261023_before_structure/EXPERIMENTS.md)。

历史A100路径 /home/share/CHUANJUN/building_wan 和mv2v环境仅为来源记录。更早的smoke/pilot代码保留在code/Python_3D_Scanner/training/；不混入当前VACE训练。
