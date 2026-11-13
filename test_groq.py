import os
from dotenv import load_dotenv
from groq import Groq

# ==========================================================
# Load .env
# ==========================================================

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found in .env"
    )

# ==========================================================
# Groq Client
# ==========================================================

client = Groq(
    api_key=API_KEY
)

# ==========================================================
# Get Available Models
# ==========================================================

print("=" * 70)
print("CHECKING AVAILABLE GROQ MODELS")
print("=" * 70)

try:

    models = client.models.list()

    print("\nModels available to this API key:\n")

    model_ids = []

    for model in models.data:

        model_id = model.id

        model_ids.append(model_id)

        print(model_id)

    print("\n")
    print("=" * 70)
    print("TOTAL MODELS")
    print("=" * 70)

    print(len(model_ids))

except Exception as e:

    print("\nERROR WHILE GETTING MODELS:")
    print(str(e))
    raise


# ==========================================================
# Test Model
# ==========================================================

print("\n")
print("=" * 70)
print("TESTING A MODEL")
print("=" * 70)

# Prefer these models if they are available.
preferred_models = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant"
]

test_model = None

for model_name in preferred_models:

    if model_name in model_ids:

        test_model = model_name

        break


if test_model is None:

    print(
        "\nNone of the preferred models were found."
    )

    print(
        "\nPlease use one of the model IDs printed above."
    )

else:

    print(
        f"\nTesting model: {test_model}"
    )

    try:

        response = client.chat.completions.create(

            model=test_model,

            messages=[
                {
                    "role": "user",
                    "content": "Say hello in one short sentence."
                }
            ],

            temperature=0.2,

            max_tokens=50
        )

        print("\nSUCCESS!")
        print("=" * 70)

        print(
            response.choices[0].message.content
        )

        print("=" * 70)

        print(
            f"\nWORKING MODEL = {test_model}"
        )

    except Exception as e:

        print("\nMODEL TEST FAILED")
        print("=" * 70)
        print(str(e))
        print("=" * 70)