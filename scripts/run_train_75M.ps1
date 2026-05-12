# 激活midi_dit环境并启动训练，加载最佳ckpt部分权重
$env_name = "midi_dit"
$ckpt_path = "logs/ckpts/midi_dit_2025-11-28-02-08-07/epoch=79-val_loss=57.9975.ckpt"

# 激活环境
conda activate $env_name

# 正确的Hydra命令行方式
python train_midi.py --config-path=exp --config-name=train_babyslakh_midi_dit model.pretrained_ckpt_path=$ckpt_path
