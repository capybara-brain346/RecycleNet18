import json

input_file = "./data_snapshot.json"
output_file = "./data_snapshot_formatted.json"

with open(input_file, "r", encoding="utf-8") as f:
    json_data = json.load(f)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(json_data, f, indent=4)

print(f"Formatted JSON has been saved to {output_file}")
