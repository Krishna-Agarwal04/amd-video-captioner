import os
from openai import OpenAI

api_key = os.getenv("FIREWORKS_API_KEY")
client = OpenAI(
    api_key=api_key,
    base_url="https://api.fireworks.ai/inference/v1"
)

# Test 1: DeepSeek V4 Pro Text response
print("--- Test 1: DeepSeek V4 Pro Text Response ---")
try:
    response = client.chat.completions.create(
        model="accounts/fireworks/models/deepseek-v4-pro",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Reply in one short sentence."},
            {"role": "user", "content": "Explain what a database is."}
        ],
        max_tokens=50
    )
    print("Response:", response.choices[0].message.content)
except Exception as e:
    print("Error:", e)

# Test 2: Kimi K2.6 VLM response with a simple prompt
print("\n--- Test 2: Kimi K2.6 simple image description ---")
dummy_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
try:
    response = client.chat.completions.create(
        model="accounts/fireworks/models/kimi-k2p6",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the main color of this image. Keep it to one word."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{dummy_b64}"
                        }
                    }
                ]
            }
        ],
        max_tokens=50
    )
    print("Response:", response.choices[0].message.content)
except Exception as e:
    print("Error:", e)
