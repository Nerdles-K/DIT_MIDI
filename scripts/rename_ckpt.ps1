# PowerShell脚本：重命名ckpt文件，去除等号，避免Hydra解析冲突
Rename-Item -Path "logs/ckpts/midi_dit_2025-11-28-02-08-07/epoch=79-val_loss=57.9975.ckpt" -NewName "epoch79-val_loss57.9975.ckpt"
