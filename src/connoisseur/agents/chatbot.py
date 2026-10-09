"""Lab 9: intent-routed food chatbot (Gradio UI on top of the recommendation flow).

Run:  python -m connoisseur.agents.chatbot            # smoke tests (needs API key)
      python -m connoisseur.agents.chatbot --launch   # start the Gradio UI
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from connoisseur.llm_client import get_langchain_chat_model
from connoisseur.utils import parse_json_response

VALID_INTENTS = ["restaurant", "recipe", "both", "clarification", "database"]

DEFAULT_PREFERENCES: dict[str, Any] = {
    "favorite_cuisines": [],
    "dietary_restrictions": [],
    "dining_occasion": "not specified",
    "price_range": "not specified",
    "flavor_preferences": [],
    "other_preferences": "",
}


@lru_cache(maxsize=1)
def get_llm():
    return get_langchain_chat_model(temperature=0.7)


def classify_intent(user_message: str, llm) -> str:
    system_prompt = """You are an intent classifier for a food recommendation system.
Analyze the user's message and classify it as ONE of:
- "restaurant" - User wants restaurant recommendations
- "recipe" - User wants recipe recommendations
- "both" - User wants both restaurant and recipe recommendations
- "clarification" - User needs help or is asking a question
- "database" - User wants to add/edit/delete database entries
Respond with ONLY the classification label."""
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_message)])
    intent = response.content.strip().lower().strip('"')
    return intent if intent in VALID_INTENTS else "clarification"


def extract_preferences(user_message: str, llm) -> dict[str, Any]:
    system_prompt = """You are a preference extractor for a food recommendation system.
Extract user preferences from their message and return JSON with these keys:
- favorite_cuisines: List of mentioned cuisines
- dietary_restrictions: List of dietary needs
- dining_occasion: Type of dining
- price_range: Price preference
- flavor_preferences: List of flavor preferences
- other_preferences: Any other relevant details
Respond with ONLY valid JSON."""
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_message)])
    try:
        return parse_json_response(response.content)
    except ValueError:
        return dict(DEFAULT_PREFERENCES)


def run_recommendation_workflow(preferences: dict[str, Any], recommendation_type: str) -> dict[str, Any]:
    """Placeholder results (as in the lab); swap in the Lab 8 workflow / Lab 6 retrieval later."""
    mock = {
        "restaurants": [
            {"name": "Green Leaf Bistro", "cuisine": "Mediterranean", "price": "$$", "reasoning": "Healthy, plant-based menu."},
            {"name": "Spice Route", "cuisine": "Indian", "price": "$$", "reasoning": "Extensive vegetarian options."},
        ],
        "recipes": [
            {"name": "One-Pot Chickpea Curry", "cuisine": "Indian", "difficulty": "Easy", "reasoning": "Protein-rich and bold flavors."},
            {"name": "Mediterranean Quinoa Bowl", "cuisine": "Mediterranean", "difficulty": "Easy", "reasoning": "Nutritious and satisfying."},
        ],
    }
    if recommendation_type == "restaurant":
        return {"restaurants": mock["restaurants"]}
    if recommendation_type == "recipe":
        return {"recipes": mock["recipes"]}
    return mock


def format_recommendations(recommendations: dict[str, Any]) -> str:
    output = ""
    if recommendations.get("restaurants"):
        output += "🍽️ **Restaurant Recommendations:**\n\n"
        for i, r in enumerate(recommendations["restaurants"], 1):
            output += f"**{i}. {r['name']}**\n   - Cuisine: {r['cuisine']}\n   - Price: {r['price']}\n   - Why: {r['reasoning']}\n\n"
    if recommendations.get("recipes"):
        output += "👨‍🍳 **Recipe Recommendations:**\n\n"
        for i, r in enumerate(recommendations["recipes"], 1):
            output += f"**{i}. {r['name']}**\n   - Cuisine: {r['cuisine']}\n   - Difficulty: {r['difficulty']}\n   - Why: {r['reasoning']}\n\n"
    return output or "I couldn't generate recommendations. Please try again with more details."


def recommendation_chatbot(message: str, history: list) -> str:
    try:
        llm = get_llm()
        intent = classify_intent(message, llm)
        if intent == "clarification":
            return "I'm your food recommendation assistant! I can help you with restaurants, recipes, and database management."
        if intent == "database":
            return "To manage the database, please use the tabs above (Add Restaurant, Add Recipe)."
        preferences = extract_preferences(message, llm)
        return format_recommendations(run_recommendation_workflow(preferences, intent))
    except Exception as e:  # noqa: BLE001
        return f"I encountered an error: {e}. Please make sure DEEPSEEK_API_KEY is set."


# Database placeholders (the lab only mocks these)
def add_restaurant(name, cuisine, price, location, description):
    return f"✅ Successfully added '{name}' to the database!"


def add_recipe(name, cuisine, difficulty, time, ingredients, instructions):
    return f"✅ Successfully added '{name}' recipe to the database!"


def build_demo():
    import gradio as gr  # lazy: only needed to launch the UI

    with gr.Blocks(title="Food Recommendation Chatbot", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🍽️ Food Recommendation Chatbot\nYour personal AI assistant!")
        with gr.Tabs():
            with gr.Tab("💬 Chat"):
                gr.ChatInterface(fn=recommendation_chatbot, type="messages")
            with gr.Tab("➕ Add Restaurant"):
                gr.Markdown("### Add a New Restaurant")
                with gr.Row():
                    with gr.Column():
                        r_name = gr.Textbox(label="Restaurant Name")
                        r_cuisine = gr.Textbox(label="Cuisine")
                        r_price = gr.Dropdown(choices=["$", "$$", "$$$", "$$$$"], label="Price Range")
                    with gr.Column():
                        r_location = gr.Textbox(label="Location")
                        r_desc = gr.Textbox(label="Description", lines=3)
                r_btn = gr.Button("Add Restaurant", variant="primary")
                r_out = gr.Textbox(label="Status")
                r_btn.click(add_restaurant, inputs=[r_name, r_cuisine, r_price, r_location, r_desc], outputs=r_out)
            with gr.Tab("➕ Add Recipe"):
                gr.Markdown("### Add a New Recipe")
                with gr.Row():
                    with gr.Column():
                        rec_name = gr.Textbox(label="Recipe Name")
                        rec_cuisine = gr.Textbox(label="Cuisine")
                        rec_diff = gr.Dropdown(choices=["Easy", "Medium", "Hard"], label="Difficulty")
                    with gr.Column():
                        rec_time = gr.Textbox(label="Prep Time")
                        rec_ing = gr.Textbox(label="Ingredients", lines=3)
                rec_inst = gr.Textbox(label="Instructions", lines=5)
                rec_btn = gr.Button("Add Recipe", variant="primary")
                rec_out = gr.Textbox(label="Status")
                rec_btn.click(add_recipe, inputs=[rec_name, rec_cuisine, rec_diff, rec_time, rec_ing, rec_inst], outputs=rec_out)
    return demo


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Food recommendation chatbot (Lab 9).")
    parser.add_argument("--launch", action="store_true", help="start the Gradio UI")
    args = parser.parse_args(argv)

    if args.launch:
        build_demo().launch()
        return

    print("--- TASK 1: preference extraction ---")
    try:
        prefs = extract_preferences("I'm looking for affordable vegetarian Mexican restaurants", get_llm())
        print(json.dumps(prefs, indent=2))
    except Exception as e:  # noqa: BLE001
        print(f"Requires a valid DEEPSEEK_API_KEY. Error: {e}")

    print("\n--- TASK 2: chatbot recipe request ---")
    print(recommendation_chatbot("Can you recommend some easy Indian recipes for dinner?", []))


if __name__ == "__main__":
    main()
