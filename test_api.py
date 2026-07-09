import os
import sys
from openai import OpenAI

# Try to load API key from environment
api_key = os.getenv("FIREWORKS_API_KEY")

print("====================================")
print("     FIREWORKS AI DIAGNOSTIC       ")
print("====================================")
print(f"API Key set in environment: {bool(api_key)}")

if not api_key:
    print("❌ ERROR: FIREWORKS_API_KEY environment variable is not set!")
    print("Please set it in your terminal using:")
    print("  $env:FIREWORKS_API_KEY=\"your_key_here\"")
    sys.exit(1)

print(f"API Key starting chars: {api_key[:6]}...")

# Initialize Client
client = OpenAI(
    api_key=api_key,
    base_url="https://api.fireworks.ai/inference/v1"
)

# Test Kimi K2.6 Text
print("\n[Test 1] Testing Kimi K2.6 (kimi-k2p6) Text...")
try:
    response = client.chat.completions.create(
        model="accounts/fireworks/models/kimi-k2p6",
        messages=[{"role": "user", "content": "Hello, write a 1-word reply."}],
        max_tokens=10
    )
    print("✅ SUCCESS!")
    print(f"Response: {response.choices[0].message.content.strip()}")
except Exception as e:
    print("❌ FAILED!")
    print(f"Error Details: {e}")

# Test Kimi K2.6 Vision with a dummy 1x1 transparent pixel
print("\n[Test 2] Testing Kimi K2.6 (kimi-k2p6) Vision...")
dummy_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
try:
    response = client.chat.completions.create(
        model="accounts/fireworks/models/kimi-k2p6",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe what you see in this image in one word."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{dummy_b64}"
                        }
                    }
                ]
            }
        ],
        max_tokens=20
    )
    print("✅ SUCCESS!")
    print(f"Response: {response.choices[0].message.content.strip()}")
except Exception as e:
    print("❌ FAILED!")
    print(f"Error Details: {e}")

print("\n====================================")
