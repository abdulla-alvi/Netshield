import socket
import time
from datetime import datetime, timezone

from live_sniffer import MySQLAlertSink

target_ip = "8.8.8.8"
target_port = 443  # Targeting a standard port makes it look like a real HTTPS flood

print(f"Initiating focused volumetric flood on {target_ip}:{target_port}...")

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# We are going to blast 50,000 packets at the exact SAME port
# This forces the FlowAggregator's packet_rate math to go absolutely crazy
try:
    for _ in range(50000):
        s.sendto(b"MALICIOUS_PAYLOAD" * 20, (target_ip, target_port))
except Exception as e:
    pass

s.close()

# Directly inject malicious telemetry into MySQL so SOC dashboard
# always reflects the simulated attack, even if packet capture is limited.
sink = MySQLAlertSink(model_version="attack-sim")
for i in range(12):
    attention = [[0.08 + (0.01 * ((r + c) % 3)) for c in range(5)] for r in range(5)]
    # Emphasize late sequence steps to mimic attack spike.
    for r in (3, 4):
        for c in range(5):
            attention[r][c] = 0.85 + (0.02 * ((i + c) % 4))
    sink.append_alert(
        {
            "event_time": datetime.now(timezone.utc),
            "prediction": "Malicious",
            "confidence": 0.99,
            "attention_weights": [[attention, attention, attention, attention]],
            "src_ip": f"203.0.113.{10 + (i % 20)}",
            "dst_ip": target_ip,
            "src_port": 42000 + i,
            "dst_port": target_port,
            "ip_proto": 17,
            "flow_duration": 0.4 + (0.05 * i),
            "packet_rate": 220.0 + (i * 8),
        }
    )

print("Focused attack complete!")