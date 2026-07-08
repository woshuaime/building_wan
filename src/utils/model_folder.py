import os
import re


def allocate_next_model_folder(parent_dir):
    """
    在 parent_dir 下创建下一个 model00000、model00001…（五位数字递增，支持万级以上）。
    兼容旧版 model001 等任意位数：取已有 model<数字> 目录中最大编号 +1。
    """
    os.makedirs(parent_dir, exist_ok=True)
    pat = re.compile(r"^model(\d+)$")
    max_n = -1
    for name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, name)):
            continue
        m = pat.match(name)
        if m:
            max_n = max(max_n, int(m.group(1)))
    n = max_n + 1
    path = os.path.join(parent_dir, f"model{n:05d}")
    os.makedirs(path, exist_ok=True)
    return path
