import asyncio
import os

from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent


async def main():
    env = os.environ.copy()

    # Needed if you want a headed browser on Ubuntu Wayland/XWayland
    env["DISPLAY"] = os.environ.get("DISPLAY", ":0")

    if os.environ.get("XAUTHORITY"):
        env["XAUTHORITY"] = os.environ["XAUTHORITY"]

    client = MultiServerMCPClient(
        {
            "playwright": {
                "command": "npx",
                "args": [
                    "@playwright/mcp@latest"
                ],
                "transport": "stdio",
                "env": env,
            }
        }
    )

    tools = await client.get_tools()

    llm = ChatOllama(
        model="qwen3.5:9b",
        base_url="http://localhost:11434",
        temperature=0,
    )

    agent = create_react_agent(
        llm.bind_tools(tools),
        tools,
    )

    prompt = """
Use the browser tools to open https://example.com.

Wait for the page to fully load, then extract:
1. the document title
2. the H1 text
3. the first paragraph text
4. all hyperlinks and their destinations

Return the result as JSON.
"""

    result = await agent.ainvoke(
        {
            "messages": [
                ("user", prompt)
            ]
        }
    )

    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
