from openai import OpenAI
import traceback



# Your credentials
api_endpoint="https://genailab.tcs.in/openai/v1"
api_key="sk-ZalSmsmKK5U9yBHG08zVHQ"
model="azure/genailab-maas-gpt-4.1"

# Create client
client = OpenAI(
    api_key=api_key,
    base_url=api_endpoint,
)

try:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Hello! If you can read this, reply with 'API key is working.'"
            }
        ],
        max_tokens=20,
    )

    print("✅ API Key is working!")
    print(response.choices[0].message.content)

except Exception as e:
    print(type(e).__name__)
    print(e)
    traceback.print_exc()
