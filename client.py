import asyncio
from typing import Optional, Union
from contextlib import AsyncExitStack
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AzureOpenAI
from dotenv import load_dotenv
import json
import logging
logging.basicConfig(level=logging.DEBUG)

load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-15-preview",
            base_url=os.getenv("AZURE_OPENAI_ENDPOINT")
        )

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
            
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))

        self.stdio, self.write = stdio_transport

        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        
        await self.session.initialize()
        
        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])
        
    async def process_query(self, query: str) -> str:
        """Process a query using available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        available_tools = [
            {
                "type": "function",
                "description": tool.description,
                "function": {
                    "name": tool.name,
                    "parameters": tool.inputSchema
                }
            } for tool in response.tools
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1000,
            messages=messages,
            tools=available_tools
        )

        tool_results = []
        final_text = []

        # Iterate over the choices in the response
        for choice in response.choices:
            message = choice.message

            if message.role == 'assistant' and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    # Execute tool call
                    result = await self.session.call_tool(name=tool_name, arguments=tool_args)

                    tool_results.append({"call": tool_name, "result": result})
                    final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                    # Continue conversation with tool results
                    if hasattr(message, 'content') and message.content:
                        messages.append({
                            "role": "assistant",
                            "content": message.content
                        })
                    messages.append({
                        "role": "user", 
                        "content": result['content']
                    })

                    response = self.client.chat.completions.create(
                        model="gpt-4o",
                        max_tokens=1000,
                        messages=messages
                    )

                    final_text.append(response.choices[0].message.content)
            elif message.role == 'assistant' and hasattr(message, 'content'):
                final_text.append(message.content)
        final_text = [text for text in final_text if text is not None]
        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            query = input("\nQuery: ").strip()
            
            if query.lower() == 'quit':
                break
                
            try:
                response = await self.process_query(query)
                print("\n" + response)
            except UnicodeDecodeError as e:
                print(f"Decoding error: {e}")
            except Exception as e:
                print(f"Error: {e}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()
    
async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)
        
    client = MCPClient()

    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())
