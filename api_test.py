from dotenv import load_dotenv
load_dotenv()

import os
from openai import OpenAI

# Get API key
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

print("🧪 Testing GPT-5 with Correct Responses API\n")

# Test 1: GPT-5 Responses API (CORRECT WAY)
print("1️⃣ Testing GPT-5 with Responses API (correct method)...")
try:
    response = client.responses.create(
        model="gpt-5",
        input="Just say 'hi' and nothing else",
        reasoning={"effort": "minimal"},  # For speed
        text={"verbosity": "low"}         # For brevity
    )
    print(f"✅ GPT-5 Responses API: '{response.output_text}'")
except Exception as e:
    print(f"❌ GPT-5 Responses API failed: {e}")

# Test 2: GPT-5 with JSON format (Responses API)
print("\n2️⃣ Testing GPT-5 JSON with Responses API...")
try:
    response = client.responses.create(
        model="gpt-5",
        input="Return JSON with just: {\"message\": \"hi\"}",
        reasoning={"effort": "minimal"},
        text={"verbosity": "low"}
    )
    print(f"✅ GPT-5 JSON via Responses: '{response.output_text}'")
except Exception as e:
    print(f"❌ GPT-5 JSON via Responses failed: {e}")

# Test 3: GPT-5 Chat Completions (fallback method)
print("\n3️⃣ Testing GPT-5 via Chat Completions (fallback)...")
try:
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": "Say 'hi'"}],
        max_completion_tokens=10  # GPT-5 uses max_completion_tokens
    )
    print(f"✅ GPT-5 Chat Completions: '{response.choices[0].message.content}'")
except Exception as e:
    print(f"❌ GPT-5 Chat Completions failed: {e}")

# Test 4: Mini test - extraction task
print("\n4️⃣ Testing GPT-5 for simple extraction...")
try:
    sample_text = """
    1. Clean up debris from window sills.
    2. Replace missing threshold at front door.
    3. Repair garbage disposal noise.
    """
    
    response = client.responses.create(
        model="gpt-5",
        input=f"""Extract repair items as JSON array:
{sample_text}

Return: {{"items": [{{"category": "Cosmetic/Plumbing/etc", "description": "..."}}]}}""",
        reasoning={"effort": "minimal"},
        text={"verbosity": "low"}
    )
    print(f"✅ GPT-5 extraction test: '{response.output_text}'")
except Exception as e:
    print(f"❌ GPT-5 extraction failed: {e}")

print("\n🏁 GPT-5 API test complete!")
print("="*50)
print("Key insights:")
print("✅ Use client.responses.create() for GPT-5")
print("✅ Use reasoning={'effort': 'minimal'} for speed")
print("✅ Use text={'verbosity': 'low'} for brevity")
print("✅ Access output with response.output_text")
print("❌ Don't use max_tokens with Responses API")