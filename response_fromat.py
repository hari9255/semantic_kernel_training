import asyncio
from pydantic import BaseModel, ConfigDict, Field
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.anthropic import (
    AnthropicChatCompletion,
    AnthropicChatPromptExecutionSettings,
)
from semantic_kernel.functions import KernelArguments

# 1. Define the desired output schema
# Note: Anthropic structured outputs require extra="forbid" (additionalProperties: False)
class PurchaseOrderInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    customer_name: str = Field(description="Full name of the customer or company")
    po_number: str = Field(description="Purchase Order (PO) number")
    date: str = Field(description="Order date in YYYY-MM-DD format")

async def main():
    kernel = Kernel()
    
    # 2. Add Anthropic Chat Completion service
    service_id = "anthropic"
    kernel.add_service(
        AnthropicChatCompletion(
            service_id=service_id,
            ai_model_id="claude-sonnet-4-5",  # Ensure you use a Claude model supporting structured outputs
        )
    )

    # 3. Configure execution settings with response_format
    settings = AnthropicChatPromptExecutionSettings(
        service_id=service_id,
        response_format=PurchaseOrderInfo,
    )

    # 4. Initialize the ChatCompletionAgent
    agent = ChatCompletionAgent(
        kernel=kernel,
        name="POExtractorAgent",
        instructions="Extract the customer name, PO number, and date from the user's message.",
        arguments=KernelArguments(settings=settings),
    )

    # 5. Invoke the agent
    user_input = "Hi, this is Acme Corp. Please process PO #98432 dated November 25, 2026."
    response = await agent.get_response(messages=user_input)
    
    # Parse the guaranteed JSON response back into your Pydantic model
    extracted_data = PurchaseOrderInfo.model_validate_json(str(response.content))
    print(f"Customer: {extracted_data.customer_name}")
    print(f"PO: {extracted_data.po_number}")
    print(f"Date: {extracted_data.date}")

if __name__ == "__main__":
    asyncio.run(main())
