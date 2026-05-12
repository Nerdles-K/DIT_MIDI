import json
import os
import matplotlib.pyplot as plt

# 路径设置
wandb_dir = os.path.join('logs', 'wandb', 'offline-run-20251201_044113-1zprtzxv', 'files')
summary_path = os.path.join(wandb_dir, 'wandb-summary.json')

# 读取 summary 文件
with open(summary_path, 'r', encoding='utf-8') as f:
    summary = json.load(f)

# 提取所有包含 'loss' 的字段
loss_keys = [k for k in summary.keys() if 'loss' in k.lower()]
if not loss_keys:
    print('未找到包含 loss 的字段。')
    exit(1)

# 打印可用的 loss 字段
print('可用的 loss 字段:', loss_keys)

# 画图
plt.figure(figsize=(8, 5))
for key in loss_keys:
    value = summary[key]
    # 如果是单个数值，跳过
    if isinstance(value, (int, float)):
        continue
    # 如果是列表，画曲线
    if isinstance(value, list):
        plt.plot(value, label=key)

plt.xlabel('Step/Epoch')
plt.ylabel('Loss')
plt.title('WandB Loss Curve')
plt.legend()
plt.tight_layout()
plt.savefig('plot_wandb_loss.png')
plt.show()
print('已保存为 plot_wandb_loss.png')
