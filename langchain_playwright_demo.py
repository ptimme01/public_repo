import asyncio
from langchain_ollama import ChatOllama
from mcp_use import MCPAgent, MCPClient

async def main():
    # 1. Server Configuration
    config = {
        "mcpServers": {
            "playwright": {
                "command": "npx",
                "args": ["@playwright/mcp@latest"],
                "env": { "DISPLAY": ":0" } # Required for some environments
            }
        }
    }

    # 2. Initialize MCP Client
    client = MCPClient(config)

    # 3. Setup LLM and Agent
    llm = ChatOllama(
        model="qwen3.5:9b",
        base_url="http://localhost:11434",
        temperature=0,
    )

    agent = MCPAgent(llm=llm, client=client)

    # 4. Execute a Task
    result = await agent.run("Navigate to example.com and tell me the header text.")
    print(f"Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
