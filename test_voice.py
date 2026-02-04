"""
Test script for Voice Detection API
Tests the API with sample audio data
"""
import requests
import base64
import json

# Configuration
API_URL = "http://localhost:8000/api/voice-detection"  # Change to your deployed URL
API_KEY = "sk_test_123456789"

def test_voice_detection(language="English", audio_base64=None):
    """Test voice detection endpoint"""
    
    # Sample base64 audio (you should replace with real audio)
    if audio_base64 is None:
        # This is a placeholder - replace with actual MP3 base64
        audio_base64 = "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMAAAAAAA"
    
    payload = {
        "language": language,
        "audioFormat": "mp3",
        "audioBase64": audio_base64
    }
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY
    }
    
    print(f"\n🧪 Testing Voice Detection API")
    print(f"URL: {API_URL}")
    print(f"Language: {language}")
    print("-" * 50)
    
    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                print(f"\n✅ SUCCESS!")
                print(f"Classification: {data.get('classification')}")
                print(f"Confidence: {data.get('confidenceScore')}")
                print(f"Explanation: {data.get('explanation')}")
            else:
                print(f"\n❌ ERROR: {data.get('message')}")
        else:
            print(f"\n❌ HTTP Error: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request failed: {e}")

def test_all_languages():
    """Test all supported languages"""
    languages = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    
    print("\n" + "="*50)
    print("Testing All Supported Languages")
    print("="*50)
    
    for lang in languages:
        test_voice_detection(language=lang)
        print()

def test_error_cases():
    """Test error handling"""
    print("\n" + "="*50)
    print("Testing Error Cases")
    print("="*50)
    
    # Test invalid language
    print("\n1. Testing invalid language...")
    test_voice_detection(language="French")
    
    # Test invalid API key
    print("\n2. Testing invalid API key...")
    headers = {
        "Content-Type": "application/json",
        "x-api-key": "invalid_key"
    }
    payload = {
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": "test"
    }
    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("🎯 Voice Detection API Test Suite")
    print("="*50)
    
    # Test basic functionality
    test_voice_detection()
    
    # Uncomment to test all languages
    # test_all_languages()
    
    # Uncomment to test error cases
    # test_error_cases()
    
    print("\n✅ Testing complete!")
