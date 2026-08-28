from openai import OpenAI
import json
import os
from function_calling_tools.products import get_product_info, calculate_bulk_price
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

client = OpenAI(
    api_key = OPENAI_API_KEY
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_product_info",
            "description": "Get detailed information about a GlobalJava Roasters product including name, origin, flavor profile, and current price. Use this when a customer asks about a specific product.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "The product identifier, e.g., 'ethiopian-yirgacheffe', 'house-blend', 'geisha-reserve'"
                    }
                },
                "required": ["product_id"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_bulk_price",
            "description": "Calculate total price for a bulk order with volume discounts. Discounts: 5% for 25+ bags, 10% for 50+ bags, 15% for 100+ bags.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "The product identifier"
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Number of bags to order"
                    }
                },
                "required": ["product_id", "quantity"],
                "additionalProperties": False
            }
        }
    }
]

TOOL_FUNCTIONS = {
    "get_product_info": get_product_info,
    "calculate_bulk_price": calculate_bulk_price
}

def execute_tool(function_name: str, arguments: dict) -> str:
    """Safely execute a tool and return JSON result."""
    if function_name not in TOOL_FUNCTIONS:
        return json.dumps({"error": f"Unknown function: {function_name}"})

    try:
        result = TOOL_FUNCTIONS[function_name](**arguments)
        return json.dumps(result)
    except TypeError as e:
        return json.dumps({"error": f"Invalid arguments: {e}"})
    except Exception as e:
        return json.dumps({"error": f"Tool execution failed: {e}"})

messages = [
    {"role": "system", "content": "You are a helpful product assistant for GlobalJava Roasters. Always use the available tools to look up current product information and pricing. Do not rely on general knowledge about coffee."},
    {"role": "user", "content": "What can you tell me about the Ethiopian Yirgacheffe?"}
]

response = client.chat.completions.create(
    model="gpt-5.6-luna",
    messages=messages,
    tools=tools,
    tool_choice="auto",
    temperature=1,
    reasoning_effort="none"
)

assistant_message = response.choices[0].message

if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
    # Append the assistant message containing the tool call(s) exactly as required
    messages.append({
        "role": "assistant",
        "content": assistant_message.content,
        "tool_calls": [tc.model_dump() for tc in assistant_message.tool_calls]
    })

    # Process each tool call (handle all)
    for tool_call in assistant_message.tool_calls:
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        print(f"Model requested: {function_name}({arguments})")
        # Execute the function (currently only get_product_info is defined)
        result = get_product_info(**arguments)
        # Append the corresponding tool response
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)
        })


    # Now ask the model for a final response using the updated message list
    final_response = client.chat.completions.create(
        model="gpt-5.6-luna",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=1,
        reasoning_effort="none"
    )

    print(final_response.choices[0].message.content)
else:
    # Model responded with text - just print it
    print(assistant_message.content)

# agent loop
def run_agent(user_message: str, tools: list, max_iterations: int = 10) -> str:
    """Run the agentic loop until the model produces a final response."""

    messages = [
        {"role": "system", "content": "You are a product assistant for GlobalJava Roasters. Always use the available tools to look up current product information and pricing. Do not rely on general knowledge about coffee."},
        {"role": "user", "content": user_message}
    ]

    for i in range(max_iterations):
        response = client.chat.completions.create(
            model="gpt-5.6-luna",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=1,
            reasoning_effort="none"
        )

        assistant_message = response.choices[0].message

        messages.append(assistant_message)

        if not assistant_message.tool_calls:
            return assistant_message.content

        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            print(f"  Tool call: {function_name}({arguments})")

            result = execute_tool(function_name, arguments)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result  # Already JSON-serialized by execute_tool
            })
        
    return "Max iterations reached"

# Test the function
if __name__ == "__main__":
    question = "What is the price of 50 bags of Yirgacheffe?"

    print("User:", question)
    print("\nProcessing...")
    response = run_agent(question, tools)
    print("\nAssistant:", response)
