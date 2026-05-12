import json
import os

# 路径配置
train_json = "train.json"
output_json = "midi_data/lmd_full/.cache/text_features.json"


features = []
midi_files = []
missing = 0
with open(train_json, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        midi_path = item["location"]
        # 检查文件是否存在（相对路径转绝对路径）
        abs_path = os.path.join("midi_data", midi_path) if not os.path.isabs(midi_path) else midi_path
        if os.path.exists(abs_path):
            features.append({
                "text_description": item["caption"]
            })
            midi_files.append(midi_path)
        else:
            missing += 1

os.makedirs(os.path.dirname(output_json), exist_ok=True)
with open(output_json, "w", encoding="utf-8") as f:
    json.dump({"midi_files": midi_files, "text_features": features}, f, ensure_ascii=False, indent=2)

print(f"已生成 {output_json}，共包含 {len(features)} 条样本。缺失文件数量: {missing}")

os.makedirs(os.path.dirname(output_json), exist_ok=True)
with open(output_json, "w", encoding="utf-8") as f:
    json.dump({"midi_files": midi_files, "text_features": features}, f, ensure_ascii=False, indent=2)

print(f"已生成 {output_json}，共包含 {len(features)} 条样本。")
