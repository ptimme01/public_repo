import asyncio
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent


async def main():
    client = MultiServerMCPClient(
        {
            "playwright": {
                "command": "npx",
                "args": ["@playwright/mcp@latest"],
                "transport": "stdio",
                "env": {"DISPLAY": ":0"},
            }
        }
    )

    tools = await client.get_tools()

    llm = ChatOllama(
        model="qwen3.5:9b",
        base_url="http://localhost:11434",
        temperature=0,
    )

    # Some local models need explicit tool binding
    llm_with_tools = llm.bind_tools(tools)

    agent = create_react_agent(llm_with_tools, tools)

    result = await agent.ainvoke(
        {
            "messages": [
                (
                    "user",
                    "Navigate to example.com and tell me the header text.",
                )
            ]
        }
    )

    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
