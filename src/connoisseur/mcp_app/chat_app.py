"""Lab 12: "Connoisseur Companion" - Gradio chat UI over a ReAct agent loop using the MCP tools.

Run:  python -m connoisseur.mcp_app.chat_app
"""
from __future__ import annotations

from pathlib import Path

from fastmcp.client import Client, PythonStdioTransport
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from connoisseur.llm_client import get_langchain_chat_model

SERVER_SCRIPT = str(Path(__file__).parent / "server.py")
SYSTEM_PROMPT = (
    "You are Connoisseur Companion, an expert AI restaurant advisor for California. "
    "Use the available MCP tools to search for restaurant details, match vibes, and retrieve reviews "
    "to answer user questions concisely and accurately."
)
MAX_TOOL_ROUNDS = 10


def make_model():
    return get_langchain_chat_model(temperature=0.7)


def build_messages(history: list[dict], user_message: str) -> list:
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for msg in history:
        role, content = msg.get("role", ""), msg.get("content", "")
        if role == "user" and content:
            messages.append(HumanMessage(content=content))
        elif role == "assistant" and content:
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=user_message))
    return messages


def _flatten_content(raw) -> str:
    if isinstance(raw, list):
        return " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in raw)
    return str(raw)


async def chat_with_agent(user_message: str, history: list) -> str:
    """Connect to the MCP server, discover tools, and loop: LLM -> tool calls -> LLM ... -> final text."""
    transport = PythonStdioTransport(script_path=SERVER_SCRIPT)

    async with Client(transport) as client:
        mcp_tools = await client.list_tools()
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.inputSchema,
                },
            }
            for t in mcp_tools
        ]

        model = make_model().bind_tools(openai_tools)
        messages = build_messages(history, user_message)

        for _ in range(MAX_TOOL_ROUNDS):
            response = await model.ainvoke(messages)
            messages.append(response)

            if not response.tool_calls:  # no tool calls -> final answer
                return _flatten_content(response.content)

            for tool_call in response.tool_calls:
                result = await client.call_tool(tool_call["name"], tool_call["args"])
                tool_output = (
                    " ".join(item.text if hasattr(item, "text") else str(item) for item in result.content)
                    if result.content
                    else "(no result)"
                )
                messages.append(ToolMessage(content=tool_output, tool_call_id=tool_call["id"]))

        return "I wasn't able to complete that request. Please try again."


async def handle_chat(user_message, history):
    if history is None:
        history = []
    if not user_message or not user_message.strip():
        yield history
        return

    history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": "Thinking..."},
    ]
    yield history

    response_text = await chat_with_agent(user_message, history[:-2])
    history[-1] = {"role": "assistant", "content": response_text}
    yield history


def build_demo():
    import gradio as gr  # lazy: only needed to launch the UI

    with gr.Blocks(title="Connoisseur Companion", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Connoisseur Companion\nYour AI guide to California's restaurant scene. "
            "Ask me about restaurants by name, cuisine, or vibe!"
        )
        chatbot = gr.Chatbot(height=500, type="messages")
        msg_input = gr.Textbox(
            label="Ask about restaurants",
            placeholder='e.g., "Find me a moody spot in DTLA" or "Tell me about Sakura Garden"',
        )
        with gr.Row():
            btn1 = gr.Button("Find moody restaurants", size="sm")
            btn2 = gr.Button("Tell me about Iron & Embers", size="sm")
            btn3 = gr.Button("Zen dining in Little Tokyo?", size="sm")

        msg_input.submit(handle_chat, [msg_input, chatbot], [chatbot])
        msg_input.submit(lambda: "", None, msg_input)

        btn1.click(handle_chat, [gr.State("Find me some moody restaurants"), chatbot], [chatbot])
        btn2.click(handle_chat, [gr.State("Tell me about Iron & Embers"), chatbot], [chatbot])
        btn3.click(handle_chat, [gr.State("What's a zen dining experience in Little Tokyo?"), chatbot], [chatbot])
    return demo


if __name__ == "__main__":
    print("Starting Connoisseur Companion...")
    build_demo().launch(share=False)
