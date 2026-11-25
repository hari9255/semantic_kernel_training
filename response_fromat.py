import asyncio
from pydantic import BaseModel
from anthropic import transform_schema   # turns a Pydantic class into a schema Claude accepts
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.anthropic import (
    AnthropicChatCompletion,
    AnthropicChatPromptExecutionSettings,
)
from semantic_kernel.functions import KernelArguments
import os
os.environ["ANTHROPIC_API_KEY"] = ""
from pypdf import PdfReader

reader = PdfReader("invoice.pdf")
pdf_text = "\n".join(page.extract_text() for page in reader.pages)

class PurchaseOrderInfo(BaseModel):
    invoice_number: str
    PO_number: str
    total_amount: float


# Add the field Semantic Kernel doesn't expose
class AnthropicStructuredSettings(AnthropicChatPromptExecutionSettings):
    output_config: dict | None = None


settings = AnthropicStructuredSettings(
    max_tokens=1024,
    output_config={
        "format": {"type": "json_schema", "schema": transform_schema(PurchaseOrderInfo)}
    },
)

agent = ChatCompletionAgent(
    service=AnthropicChatCompletion(ai_model_id="claude-haiku-4-5"),  # reads ANTHROPIC_API_KEY
    name="POExtractor",
    instructions="Extract purchase order details from the text.",
    arguments=KernelArguments(settings=settings),
)


async def main():
    response = await agent.get_response(messages=pdf_text)
    po = PurchaseOrderInfo.model_validate_json(str(response.content))
    print(po)

asyncio.run(main())
