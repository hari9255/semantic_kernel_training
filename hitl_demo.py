import asyncio, os
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.anthropic import AnthropicChatCompletion, AnthropicChatPromptExecutionSettings
from semantic_kernel.filters import FilterTypes
from semantic_kernel.functions import FunctionResult, KernelArguments, kernel_function

import os
os.environ["ANTHROPIC_API_KEY"] = ""
class FilePlugin:
    @kernel_function(description="Save the finished article to a text file")
    def save_to_file(self, content: str) -> str:
        with open("article.txt", "w") as f:
            f.write(content)
        return "Saved to article.txt"

async def human_review(context, next):
    print("\n--- DRAFT ---\n" + context.arguments["content"] + "\n-------------")
    feedback = input("Press Enter to approve, or type feedback: ")
    if feedback:
        context.function_result = FunctionResult(
            function=context.function.metadata,
            value=f"Human rejected the draft. Revise using this feedback, then save again: {feedback}")
        return
    await next(context)

kernel = Kernel()
kernel.add_service(AnthropicChatCompletion(ai_model_id="claude-sonnet-5-5", api_key=os.environ["ANTHROPIC_API_KEY"]))
kernel.add_plugin(FilePlugin(), "File")
kernel.add_filter(FilterTypes.AUTO_FUNCTION_INVOCATION, human_review)

settings = AnthropicChatPromptExecutionSettings(
    function_choice_behavior=FunctionChoiceBehavior.Auto(), max_tokens=1000)

async def main():
    result = await kernel.invoke_prompt(
        "Write a 100-word article on why US taxpayers should keep receipts, then save it to a file.",
        arguments=KernelArguments(settings=settings))
    print(result)

if __name__=="__main__":
    asyncio.run(main())