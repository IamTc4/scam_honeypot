"""
Test script for Scam Honeypot API
Tests multi-turn conversation and intelligence extraction
"""
import requests
import json
from datetime import datetime

# Configuration
API_URL = "http://localhost:8001/api/scam-honeypot"  # Change to your deployed URL
API_KEY = "sk_test_987654321"

def test_honeypot(session_id, message_text, conversation_history=None):
    """Test honeypot endpoint"""
    
    if conversation_history is None:
        conversation_history = []
    
    payload = {
        "sessionId": session_id,
        "message": {
            "sender": "scammer",
            "text": message_text,
            "timestamp": datetime.now().isoformat() + "Z"
        },
        "conversationHistory": conversation_history,
        "metadata": {
            "channel": "SMS",
            "language": "English",
            "locale": "IN"
        }
    }
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY
    }
    
    print(f"\n📨 Scammer: {message_text}")
    print("-" * 50)
    
    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                reply = data.get("reply")
                print(f"🤖 Agent: {reply}")
                return reply, conversation_history
            else:
                print(f"❌ ERROR: {data.get('message')}")
                return None, conversation_history
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None, conversation_history
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None, conversation_history

def test_multi_turn_conversation():
    """Test a complete multi-turn scam conversation"""
    
    print("\n" + "="*50)
    print("🎭 Testing Multi-Turn Scam Conversation")
    print("="*50)
    
    session_id = "test-session-123"
    conversation_history = []
    
    # Conversation flow
    scammer_messages = [
        "Your bank account will be blocked today. Verify immediately.",
        "Share your UPI ID to avoid account suspension.",
        "We need your OTP to verify your identity.",
        "Please send the 6-digit code we sent to your phone.",
        "Transfer ₹100 to verify your account: scammer@paytm",
    ]
    
    for msg in scammer_messages:
        reply, _ = test_honeypot(session_id, msg, conversation_history)
        
        if reply:
            # Add scammer message to history
            conversation_history.append({
                "sender": "scammer",
                "text": msg,
                "timestamp": datetime.now().isoformat() + "Z"
            })
            
            # Add agent reply to history
            conversation_history.append({
                "sender": "user",
                "text": reply,
                "timestamp": datetime.now().isoformat() + "Z"
            })
        
        print()
    
    print("\n✅ Conversation complete!")
    print(f"Total messages exchanged: {len(conversation_history)}")

def test_scam_detection():
    """Test scam detection with various scam types"""
    
    print("\n" + "="*50)
    print("🔍 Testing Scam Detection")
    print("="*50)
    
    test_cases = [
        ("Bank Fraud", "Your account will be suspended. Click http://fake-bank.com to verify"),
        ("UPI Scam", "Send your UPI PIN to receive cashback"),
        ("Lottery Scam", "Congratulations! You won ₹10 lakhs. Claim now!"),
        ("OTP Phishing", "Share your OTP immediately or account will be blocked"),
        ("Tech Support", "Your computer is infected! Call us now: +91-9876543210"),
    ]
    
    for scam_type, message in test_cases:
        print(f"\n🧪 Testing: {scam_type}")
        test_honeypot(f"test-{scam_type.lower().replace(' ', '-')}", message)

def test_error_cases():
    """Test error handling"""
    
    print("\n" + "="*50)
    print("⚠️  Testing Error Cases")
    print("="*50)
    
    # Test invalid API key
    print("\n1. Testing invalid API key...")
    headers = {
        "Content-Type": "application/json",
        "x-api-key": "invalid_key"
    }
    payload = {
        "sessionId": "test",
        "message": {
            "sender": "scammer",
            "text": "Test",
            "timestamp": datetime.now().isoformat() + "Z"
        },
        "conversationHistory": []
    }
    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json() if response.status_code != 401 else response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("🎯 Scam Honeypot API Test Suite")
    print("="*50)
    
    # Test multi-turn conversation
    test_multi_turn_conversation()
    
    # Uncomment to test different scam types
    # test_scam_detection()
    
    # Uncomment to test error cases
    # test_error_cases()
    
    print("\n✅ All tests complete!")
