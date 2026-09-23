"""
Example script to launch an on-demand 2-hour compute/mining task via Python SDK or REST.
"""
import httpx
import time


def trigger_mining_session(master_url: str = "http://localhost:8000", duration_hours: float = 2.0):
    duration_seconds = duration_hours * 3600
    print(f"Triggering {duration_hours}-hour compute session on cluster at {master_url}...")
    
    with httpx.Client(timeout=10.0) as client:
        resp = client.post(f"{master_url}/api/mine/start", json={
            "duration_seconds": duration_seconds,
            "difficulty": 4,
            "threads": 0,  # Auto-threads based on CPU/GPU cores
            "algorithm": "sha256_custom",
        })
        if resp.status_code == 200:
            print("Session started successfully:", resp.json())
        else:
            print("Failed to start session:", resp.text)


if __name__ == "__main__":
    trigger_mining_session()
