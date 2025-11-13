# app.py
# Full Chainlit app for generating detailed notes on provided code snippets.
# Uses Azure AI Inference with Grok-3 model and a custom system prompt for structured MD output.
# To run: pip install chainlit azure-ai-inference azure-core
# Set GITHUB_TOKEN: export GITHUB_TOKEN="your_token_here"
# Then: chainlit run app.py

import os
import asyncio
import chainlit as cl
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

# Configuration
endpoint = "https://models.github.ai/inference"
model = "xai/grok-3"
token = os.environ["GITHUB_TOKEN"]

# Global client
client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

# Custom system prompt for detailed notes generation
SYSTEM_PROMPT = """Create in details Notes 
If needed as Table and mermaid diagram
Explain as much as you can 
Be grounded to content 
Create only one .md file 
if you found math equaction then write all step with github markdown supported latex"""

@cl.on_chat_start
async def start():
    # Initialize conversation history with system message
    messages = [SystemMessage(SYSTEM_PROMPT)]
    cl.user_session.set("messages", messages)
    
    await cl.Message(content="Welcome! Paste a code snippet, and I'll generate detailed notes in Markdown format, including tables, diagrams, and explanations where applicable.").send()

@cl.on_message
async def main(message: cl.Message):
    # Retrieve conversation history
    messages = cl.user_session.get("messages")
    
    # Add the user's message (code snippet)
    user_msg = UserMessage(content=f"Analyze this code and generate detailed notes:\n\n```{message.content}```")
    messages.append(user_msg)
    
    # Call the model with retry logic for rate limits
    max_retries = 3
    retry_delay = 1  # Start with 1 second
    
    for attempt in range(max_retries):
        try:
            response = await cl.make_async(client.complete)(
                messages=messages,
                temperature=0.3,  # Lower for more deterministic, structured output
                top_p=0.9,
                model=model,
                stream=False  # Set to True if streaming is needed and supported
            )
            
            # Add the assistant's response to history
            assistant_msg = AssistantMessage(content=response.choices[0].message.content)
            messages.append(assistant_msg)
            cl.user_session.set("messages", messages)
            
            # Send the response as Markdown
            await cl.Message(content=assistant_msg.content, markdown=True).send()
            break  # Success, exit retry loop
            
        except HttpResponseError as e:
            if e.status_code == 429:  # Rate limit error
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    await cl.Message(content=f"⚠️ Rate limit reached. Retrying in {wait_time} seconds...").send()
                    await asyncio.sleep(wait_time)
                else:
                    await cl.Message(content="❌ Rate limit exceeded. Please wait a moment and try again later.").send()
            else:
                await cl.Message(content=f"Error: {str(e)}").send()
                break
        except Exception as e:
            await cl.Message(content=f"Error generating notes: {str(e)}. Please check your setup and try again.").send()
            break

@cl.on_chat_end
async def end():
    # Clear session on chat end
    cl.user_session.clear()