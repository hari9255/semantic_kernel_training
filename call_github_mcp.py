import asyncio
import os
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.anthropic import AnthropicChatCompletion , AnthropicChatPromptExecutionSettings

from semantic_kernel.connectors.mcp import MCPStdioPlugin


os.environ["ANTHROPIC_API_KEY"] = ""
async def main():
    # Initialize the GitHub MCP Server via npx or Docker
    github_plugin = MCPStdioPlugin(
        name="GitHub",
        description="GitHub MCP Server plugin",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env={"GITHUB_PERSONAL_ACCESS_TOKEN": ""}
    )

    async with github_plugin:
        # Create an agent with the MCP plugin attached
        agent = ChatCompletionAgent(
            service= AnthropicChatCompletion(ai_model_id= "claude-haiku-4-5"),
            name="GitHubAssistant",
            instructions="Use the GitHub tools to inspect repositories, issues, and PRs.",
            plugins=[github_plugin],
        )

        response = await agent.get_response(
            messages="what are different files present in repo hari9255/semantic_kernel_training"
        )
        print(response.content)

if __name__ == "__main__":
    asyncio.run(main())
