import litellm
from litellm import ModelResponse, Choices, Message
from devtools import debug
from datetime import date
import json

tools = [
    {
        "type": "function",
        "function": {
            "name": "current_date",
            "description": "Provides the current date.",
            "parameters":  { "type": "object", "properties": {} }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Provides the current weather.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get the weather for.",
                    }
                },
                "required": ["location"]
            }
        }
    }
]

def current_date() -> dict:
    return {"current_date": date.today().strftime("%-d. %-m. %Y")}

def get_weather(location: str) -> dict:
    return {"location": location, "temperature": 30, "unit": "celsius", "condition": "Sunny"}

available_tools = {
    "current_date": current_date,
    "get_weather": get_weather
}

messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Hey, tell me what is current date"
        }
    ]

response = litellm.completion(
    model="ollama/mistral-small3.2",
    messages=messages,
    tools=tools,
    tool_choice="auto",
    api_base="http://host.docker.internal:11434"
)

if isinstance(response, ModelResponse) and len(response.choices) > 0:
    choice = response.choices[0]
    if isinstance(choice, Choices):
        response_message = choice.message

        if response_message.tool_calls and len(response_message.tool_calls) > 0:
            tool_call = response_message.tool_calls[0]
            messages.append({
                "role": "assistant",
                "content": response_message.content or "",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        }
                    }
                ]
            })

            function_name = tool_call.function.name
            function_arguments = json.loads(tool_call.function.arguments)
            tool_id = tool_call.id
            if function_name in available_tools:
                tool_response = available_tools[function_name](**function_arguments)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": json.dumps(tool_response)
                })

                final_response = litellm.completion(
                    model="ollama/mistral-small3.2",
                    messages=messages,
                    api_base="http://host.docker.internal:11434"
                )
                
                debug(final_response)