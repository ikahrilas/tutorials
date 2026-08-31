from openai import OpenAI
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
from product_server import mcp
from dotenv import load_dotenv
from product_server import get_product_info, calculate_bulk_price

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

def run_sync(coro):
    """Run an async coroutine to completion, whether or not an event loop is already running (e.g. in Jupyter)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(1) as pool:
        return pool.submit(asyncio.run, coro).result()

def get_openai_tools(mcp_server):
    """Convert MCP tools to OpenAI tools format."""
    tools = []

    for tool in run_sync(mcp_server.list_tools()):
        tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
        })

    return tools

def get_tool_functions(mcp_server):
    """Get a mapping of tool names to their functions."""
    return {tool.name: tool.fn for tool in run_sync(mcp_server.list_tools())}

def execute_tool(function_name: str, arguments: dict, tool_functions: dict) -> str:
    """Safely execute a tool and return JSON result."""
    if function_name not in tool_functions:
        return json.dumps({"error": f"Unknown function: {function_name}"})

    try:
        result = tool_functions[function_name](**arguments)
        return json.dumps(result)
    except TypeError as e:
        return json.dumps({"error": f"Invalid arguments: {e}"})
    except Exception as e:
        return json.dumps({"error": f"Tool execution failed: {e}"})


def run_agent(user_message: str, mcp_server, max_iterations: int = 10) -> str:
    """Run the agentic loop using tools from an MCP server."""

    # Convert MCP tools to OpenAI format
    tools = get_openai_tools(mcp_server)
    tool_functions = get_tool_functions(mcp_server)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a product assistant for GlobalJava Roasters. "
                "Always use the available tools to look up current product information and pricing. "
                "Do not rely on general knowledge about coffee."
            )
        },
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

        if not assistant_message.tool_calls:
            return assistant_message.content

        messages.append(assistant_message)

        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            print(f"  Tool call: {function_name}({arguments})")

            result = execute_tool(function_name, arguments, tool_functions)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

    return "Max iterations reached"

if __name__ == "__main__":
    question = "What is the price of 50 bags of Yirgacheffe?"

    print("User:", question)
    print("\nProcessing...")
    response = run_agent(question, mcp)
    print("\nAssistant:", response)