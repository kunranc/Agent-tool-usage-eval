"""
Smoke test: verify Claude function calling works end-to-end.

Run:    python hello.py
Expect: the model emits a get_weather tool call for "Tokyo", we return a
        hardcoded result, the model uses it to produce a final answer.

If this prints all three blocks (tool call -> tool result -> final answer),
the SDK + API key + function-calling format are all wired up correctly
and you can move on to building the real tools.
"""

import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY or API_KEY == "your_key_here":
    raise SystemExit(
        "ANTHROPIC_API_KEY not set. Edit .env and replace 'your_key_here' "
        "with your real key from https://console.anthropic.com/settings/keys"
    )

MODEL = "claude-haiku-4-5-20251001"


# --- the fake tool ----------------------------------------------------------
def get_weather(city: str) -> str:
    """Hardcoded for the smoke test."""
    return f"It is 18C and cloudy in {city}."


# --- declare it to Claude ---------------------------------------------------
TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a given city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'Tokyo'.",
                }
            },
            "required": ["city"],
        },
    }
]


def main() -> None:
    client = anthropic.Anthropic(api_key=API_KEY)

    messages = [
        {"role": "user", "content": "What's the weather like in Tokyo right now?"}
    ]

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        temperature=0,
        tools=TOOLS,
        messages=messages,
    )

    # Step 1: model should emit a tool call
    tool_use = next((b for b in response.content if b.type == "tool_use"), None)
    if not tool_use:
        print("Model did not call a tool. Raw response:")
        for block in response.content:
            if block.type == "text":
                print(block.text)
        return

    print(f"Model called tool: {tool_use.name}({tool_use.input})")

    # Step 2: execute the tool, send the result back
    result = get_weather(**tool_use.input)
    print(f"Tool returned: {result}")

    messages.append({"role": "assistant", "content": response.content})
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result,
                }
            ],
        }
    )

    followup = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        temperature=0,
        tools=TOOLS,
        messages=messages,
    )

    # Step 3: model uses the tool result to produce a final answer
    print("\nFinal answer:")
    for block in followup.content:
        if block.type == "text":
            print(block.text)


if __name__ == "__main__":
    main()
