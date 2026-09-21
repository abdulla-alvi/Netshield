import json
import random
from datetime import datetime

print("Injecting Level-5 Volumetric Threat into SOC Dashboard...")

# Create a realistic 5x5 Edge-BERT attention matrix
# We simulate high attention (0.8 - 1.0) on the 4th and 5th sequence steps (the attack spike)
attn_matrix = [[random.uniform(0.01, 0.2) for _ in range(5)] for _ in range(5)]
for i in range(3, 5):
    for j in range(5):
        attn_matrix[i][j] = random.uniform(0.7, 1.0)

mock_alert = {
    "timestamp": datetime.utcnow().isoformat(),
    "prediction": "Malicious",
    "confidence": 0.99,
    "attention_weights": attn_matrix
}

# Append the malicious alert to the JSON file
with open("live_alerts.json", "a") as f:
    f.write(json.dumps(mock_alert) + "\n")

print("Threat successfully injected! Check your Streamlit Dashboard.")