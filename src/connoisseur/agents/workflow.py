"""Lab 8: shared-state multi-agent workflow (profile -> retrieve -> parallel analysis -> recommend).

Run:  python -m connoisseur.agents.workflow
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from connoisseur.agents.agent_configs import AGENT_CONFIGS, create_agent_prompt
from connoisseur.llm_client import chat
from connoisseur.utils import parse_json_response

WORKFLOW_CLOSING = "Respond with structured, actionable output."


def new_state(user_input: str) -> dict:
    return {
        "user_input": user_input,
        "user_profile": {},
        "retrieved_restaurants": [],
        "retrieved_recipes": [],
        "trend_analysis": {},
        "style_analysis": {},
        "nutrition_analysis": {},
        "final_recommendations": {},
        "workflow_step": "start",
    }


def call_agent(agent_key: str, user_message: str) -> str:
    """Call one agent persona with a message and return its raw reply."""
    system_prompt = create_agent_prompt(AGENT_CONFIGS[agent_key], closing=WORKFLOW_CLOSING)
    return chat(system_prompt, user_message, temperature=0.7)


def _run_json_agent(agent_key: str, message: str) -> dict:
    return parse_json_response(call_agent(agent_key, message))


def node_generate_profile(state: dict) -> dict:
    print("\n[Phase 1] Generating user profile...")
    message = f"""Analyze this user data and create a comprehensive profile:
{state['user_input']}
Provide output in JSON format with these keys:
- favorite_cuisines (list)
- dietary_restrictions (list)
- dining_occasions (list)
- price_range (string)
- adventurousness_score (1-10)
- flavor_preferences (list)
- summary (string)
"""
    try:
        profile = _run_json_agent("user_profile_generator", message)
        print(f"✓ User profile generated: {profile.get('summary', 'No summary')}")
    except Exception as e:  # noqa: BLE001
        print(f"⚠ Error generating profile: {e}")
        profile = {"error": str(e)}
    state["user_profile"] = profile
    state["workflow_step"] = "profile_generated"
    return state


def node_retrieve_candidates(state: dict) -> dict:
    print("\n[Phase 2] Retrieving candidates from vector database...")
    message = f"""Based on this user profile:
{json.dumps(state['user_profile'], indent=2)}
Simulate retrieving top 20 restaurants and top 20 recipes from a vector database.
Return JSON with two arrays:
- restaurants: [{{"name": str, "cuisine": str, "price": str, "rating": float, "description": str}}]
- recipes: [{{"name": str, "cuisine": str, "difficulty": str, "prep_time": str, "description": str}}]
Make the results realistic and diverse.
"""
    try:
        data = _run_json_agent("rag_retriever", message)
        restaurants, recipes = data.get("restaurants", []), data.get("recipes", [])
        print(f"✓ Retrieved {len(restaurants)} restaurants and {len(recipes)} recipes")
    except Exception as e:  # noqa: BLE001
        print(f"⚠ Error retrieving candidates: {e}")
        restaurants, recipes = [], []
    state["retrieved_restaurants"] = restaurants
    state["retrieved_recipes"] = recipes
    state["workflow_step"] = "candidates_retrieved"
    return state


def node_analyze_trends(state: dict) -> dict:
    print("\n[Phase 3a] Analyzing food trends...")
    message = f"""Analyze current food trends in these options:
Restaurants: {json.dumps(state['retrieved_restaurants'][:5], indent=2)}
Recipes: {json.dumps(state['retrieved_recipes'][:5], indent=2)}
Identify 3-5 relevant trends and explain how they align with modern dining culture.
Return JSON: {{"trends": [{{"name": str, "description": str, "relevance": str}}]}}
"""
    try:
        analysis = _run_json_agent("food_trend_analyst", message)
        print(f"✓ Identified {len(analysis.get('trends', []))} trends")
    except Exception as e:  # noqa: BLE001
        print(f"⚠ Error analyzing trends: {e}")
        analysis = {"error": str(e)}
    state["trend_analysis"] = analysis
    return state


def node_analyze_styles(state: dict) -> dict:
    print("\n[Phase 3b] Analyzing food styles...")
    message = f"""Analyze the cuisine types and flavor profiles of these options to match the user's preferences:
User Profile: {json.dumps(state['user_profile'], indent=2)}
Restaurants: {json.dumps(state['retrieved_restaurants'][:5], indent=2)}
Recipes: {json.dumps(state['retrieved_recipes'][:5], indent=2)}
Return JSON: {{"style_matches": [{{"item_name": str, "flavor_profile": str, "match_reason": str}}]}}
"""
    try:
        analysis = _run_json_agent("food_style_expert", message)
        print("✓ Style analysis completed")
    except Exception as e:  # noqa: BLE001
        print(f"⚠ Error analyzing styles: {e}")
        analysis = {"error": str(e)}
    state["style_analysis"] = analysis
    return state


def node_evaluate_nutrition(state: dict) -> dict:
    print("\n[Phase 3c] Evaluating nutrition...")
    message = f"""Evaluate the nutritional fit of these options:
User Profile: {json.dumps(state['user_profile'], indent=2)}
Restaurants: {json.dumps(state['retrieved_restaurants'][:5], indent=2)}
Recipes: {json.dumps(state['retrieved_recipes'][:5], indent=2)}
Check dietary restrictions, allergens, and nutritional balance.
Return JSON: {{"compliant_items": [], "flagged_items": [], "nutritional_highlights": []}}
"""
    try:
        analysis = _run_json_agent("nutrition_expert", message)
        print("✓ Nutrition evaluation completed")
    except Exception as e:  # noqa: BLE001
        print(f"⚠ Error evaluating nutrition: {e}")
        analysis = {"error": str(e)}
    state["nutrition_analysis"] = analysis
    return state


def node_generate_recommendations(state: dict) -> dict:
    print("\n[Phase 4] Generating final recommendations...")
    message = f"""Synthesize these insights into top 5 restaurant and top 5 recipe recommendations:
User Profile: {json.dumps(state['user_profile'], indent=2)}
Restaurants: {json.dumps(state['retrieved_restaurants'][:10], indent=2)}
Recipes: {json.dumps(state['retrieved_recipes'][:10], indent=2)}
Trends: {json.dumps(state['trend_analysis'], indent=2)}
Styles: {json.dumps(state['style_analysis'], indent=2)}
Nutrition: {json.dumps(state['nutrition_analysis'], indent=2)}
Return JSON:
{{
  "restaurants": [{{"name": str, "reasoning": str}}],
  "recipes": [{{"name": str, "reasoning": str}}]
}}
Each reasoning should be 2-3 sentences explaining why it's a great match.
"""
    try:
        recs = _run_json_agent("recommendation_expert", message)
        print(f"✓ Generated {len(recs.get('restaurants', []))} restaurant recommendations")
        print(f"✓ Generated {len(recs.get('recipes', []))} recipe recommendations")
    except Exception as e:  # noqa: BLE001
        print(f"⚠ Error generating recommendations: {e}")
        recs = {"error": str(e)}
    state["final_recommendations"] = recs
    state["workflow_step"] = "complete"
    return state


def run_workflow(user_input: str) -> dict:
    state = new_state(user_input)
    state = node_generate_profile(state)
    state = node_retrieve_candidates(state)

    print("\n[Phase 3] Running analysis agents in parallel...")
    with ThreadPoolExecutor(max_workers=3) as executor:
        # each node gets its own shallow copy and returns it with one key filled in
        f_trends = executor.submit(node_analyze_trends, dict(state))
        f_styles = executor.submit(node_analyze_styles, dict(state))
        f_nutrition = executor.submit(node_evaluate_nutrition, dict(state))
        state["trend_analysis"] = f_trends.result()["trend_analysis"]
        state["style_analysis"] = f_styles.result()["style_analysis"]
        state["nutrition_analysis"] = f_nutrition.result()["nutrition_analysis"]

    return node_generate_recommendations(state)


def evaluate_recommendations(result: dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("RECOMMENDATION EVALUATION")
    print("=" * 80)
    profile = result.get("user_profile", {})
    recs = result.get("final_recommendations", {})
    restaurants, recipes = recs.get("restaurants", []), recs.get("recipes", [])

    print(f"\n✓ Number of restaurant recommendations: {len(restaurants)}")
    print(f"✓ Number of recipe recommendations: {len(recipes)}")

    if profile.get("dietary_restrictions"):
        print(f"\n✓ Dietary restrictions identified: {', '.join(profile['dietary_restrictions'])}")
        print("  → Check if recommendations respect these restrictions")
    if profile.get("favorite_cuisines"):
        print(f"\n✓ Favorite cuisines: {', '.join(profile['favorite_cuisines'])}")
        print("  → Check if recommendations include these cuisines")

    if restaurants:
        print("\n✓ First restaurant recommendation:")
        print(f"  Name: {restaurants[0].get('name', 'N/A')}")
        print(f"  Reasoning: {restaurants[0].get('reasoning', 'N/A')}")
    if recipes:
        print("\n✓ First recipe recommendation:")
        print(f"  Name: {recipes[0].get('name', 'N/A')}")
        print(f"  Reasoning: {recipes[0].get('reasoning', 'N/A')}")
    print("\n" + "=" * 80)


TEST_USER_1 = """
Restaurant Visit History:
- Visited "Green Bowl" (Vegan, $$) 8 times
- Visited "Mediterranean Grill" (Mediterranean, $$) 5 times
- Visited "Juice Lab" (Smoothies, $) 3 times
Dietary Restrictions: Vegan, Gluten-Free
"""

if __name__ == "__main__":
    try:
        evaluate_recommendations(run_workflow(TEST_USER_1))
    except Exception as e:  # noqa: BLE001
        print(f"Workflow failed (check DEEPSEEK_API_KEY): {e}")
