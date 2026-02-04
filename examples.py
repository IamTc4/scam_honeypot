"""
Example Usage Scripts for GUVI Hackathon Solutions
"""

import requests
import base64
import json

# ============================================
# VOICE DETECTION EXAMPLE
# ============================================

def test_voice_detection_api():
    """Example: Send a Base64 audio to Voice Detection API"""
    
    # Replace with your deployed URL or use localhost
    url = "http://localhost:8000/api/voice-detection"
    api_key = "sk_test_123456789"
    
    # Example: Read an MP3 file and convert to Base64
    # (Replace with actual MP3 file path)
    try:
        with open("sample_audio.mp3", "rb") as audio_file:
            audio_bytes = audio_file.read()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    except FileNotFoundError:
        print("⚠️ sample_audio.mp3 not found. Using dummy base64.")
        # Minimal MP3 header for testing (won't work for real detection)
        audio_base64 = "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMAAAAAAA"
    
    # Prepare request
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key
    }
    
    payload = {
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": audio_base64
    }
    
    # Send request
    print("📤 Sending Voice Detection Request...")
    response = requests.post(url, headers=headers, json=payload)
    
    # Display response
    print(f"📥 Status Code: {response.status_code}")
    print(f"📄 Response:\n{json.dumps(response.json(), indent=2)}")
    
    return response.json()


# ============================================
# SCAM HONEYPOT EXAMPLE
# ============================================

def test_honeypot_api():
    """Example: Simulate a scam conversation"""
    
    url = "http://localhost:8001/api/scam-honeypot"
    api_key = "sk_test_987654321"
    session_id = "demo-session-12345"
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key
    }
    
    # Message 1: Initial scam attempt
    print("\n" + "="*60)
    print("🎭 Scam Conversation Simulation")
    print("="*60)
    
    msg1_payload = {
        "sessionId": session_id,
        "message": {
            "sender": "scammer",
            "text": "URGENT: Your bank account has been compromised. Click here to verify: http://fake-bank.com",
            "timestamp": "2026-02-02T10:00:00Z"
        },
        "conversationHistory": [],
        "metadata": {
            "channel": "SMS",
            "language": "English",
            "locale": "IN"
        }
    }
    
    print("\n🔴 Scammer: URGENT: Your bank account has been compromised...")
    response1 = requests.post(url, headers=headers, json=msg1_payload)
    reply1 = response1.json()["reply"]
    print(f"🟢 Agent: {reply1}")
    
    # Message 2: Follow-up
    msg2_payload = {
        "sessionId": session_id,
        "message": {
            "sender": "scammer",
            "text": "Send money to testscam@paytm or call +919876543210 immediately",
            "timestamp": "2026-02-02T10:01:00Z"
        },
        "conversationHistory": [
            msg1_payload["message"],
            {"sender": "user", "text": reply1, "timestamp": "2026-02-02T10:00:30Z"}
        ]
    }
    
    print(f"\n🔴 Scammer: Send money to testscam@paytm...")
    response2 = requests.post(url, headers=headers, json=msg2_payload)
    reply2 = response2.json()["reply"]
    print(f"🟢 Agent: {reply2}")
    
    # Message 3: More pressure
    msg3_payload = {
        "sessionId": session_id,
        "message": {
            "sender": "scammer",
            "text": "Your account will be blocked in 10 minutes. Transfer to account 1234567890123456",
            "timestamp": "2026-02-02T10:02:00Z"
        },
        "conversationHistory": [
            msg1_payload["message"],
            {"sender": "user", "text": reply1, "timestamp": "2026-02-02T10:00:30Z"},
            msg2_payload["message"],
            {"sender": "user", "text": reply2, "timestamp": "2026-02-02T10:01:30Z"}
        ]
    }
    
    print(f"\n🔴 Scammer: Your account will be blocked in 10 minutes...")
    response3 = requests.post(url, headers=headers, json=msg3_payload)
    reply3 = response3.json()["reply"]
    print(f"🟢 Agent: {reply3}")
    
    print("\n" + "="*60)
    print("✅ Conversation complete!")
    print("📊 Intelligence should be extracted and sent to GUVI endpoint")
    print("="*60)


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("🚀 GUVI Hackathon API Examples\n")
    
    choice = input("Choose test:\n1. Voice Detection\n2. Scam Honeypot\n3. Both\nEnter (1/2/3): ")
    
    if choice in ["1", "3"]:
        print("\n" + "="*60)
        print("🎤 Testing Voice Detection API")
        print("="*60)
        try:
            test_voice_detection_api()
        except Exception as e:
            print(f"❌ Error: {e}")
    
    if choice in ["2", "3"]:
        print("\n" + "="*60)
        print("🕵️ Testing Scam Honeypot API")
        print("="*60)
        try:
            test_honeypot_api()
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n✨ Testing complete!")
