import json

def record_json(inputs, output_path):
    with open(output_path, 'w', encoding = 'utf-8') as f:
        json.dump(inputs, ensure_ascii=False, indent = 4)

