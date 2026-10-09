"""Lab 7: agent personas and task specifications (single source of truth, reused by Lab 8).

Run a quick smoke test:  python -m connoisseur.agents.agent_configs
"""
from __future__ import annotations

from connoisseur.llm_client import chat

AGENT_CONFIGS: dict[str, dict[str, str]] = {
    "user_profile_generator": {
        "role": "User Profile Generator",
        "goal": "Analyze user restaurant visit history and social media posts to create a comprehensive profile including preferences, dietary restrictions, favorite cuisines, and dining patterns.",
        "backstory": """You are an expert user behavior analyst with 10 years of experience in the food and hospitality industry.
    You excel at reading between the lines to understand not just what users say they like, but what their actions reveal
    about their true preferences. You have a talent for identifying patterns in dining behavior, recognizing subtle preferences,
    and building rich user profiles that capture both explicit and implicit food preferences. You understand that a user's
    social media posts and check-ins tell a story about their culinary journey, and you're skilled at extracting meaningful
    insights from unstructured data.""",
    },
    "rag_retriever": {
        "role": "RAG Retriever",
        "goal": "Query multimodal vector databases to retrieve relevant restaurants, recipes, and food-related content based on user profiles and similarity search.",
        "backstory": """You are a data retrieval specialist with expertise in vector databases and semantic search.
    You understand how embeddings capture meaning and can craft queries that retrieve the most relevant information
    from large collections of restaurant data, recipes, and food images. You know when to use similarity search versus
    filtered search, and you can balance relevance with diversity to ensure recommendations aren't repetitive.
    You've worked with Pinecone, Weaviate, and ChromaDB, and you understand the nuances of multimodal retrieval
    where text and images work together to represent food experiences.""",
    },
    "food_trend_analyst": {
        "role": "Food Trend Analyst",
        "goal": "Identify current food trends, popular ingredients, emerging dining concepts, and culinary movements to ensure recommendations are timely and culturally relevant.",
        "backstory": """You are a culinary journalist and trend forecaster who has spent 15 years covering food culture across
    global markets. You have your finger on the pulse of what's happening in the food world—from viral TikTok recipes to
    Michelin-starred innovations. You track emerging ingredients like kelp noodles and yuzu, monitor the rise of food
    movements like plant-based dining and zero-waste cooking, and spot the next big thing before it goes mainstream.
    You read Eater, Bon Appétit, and industry reports daily, and you know the difference between a fleeting fad and a
    lasting trend.""",
    },
    "food_style_expert": {
        "role": "Food Style Expert",
        # The lab left this blank ("Fill this in"); written to match the Lab 8 version of the goal.
        "goal": "Analyze cuisine types, cooking techniques, and flavor profiles to match them to user preferences and suggest culturally authentic options.",
        "backstory": """You are a trained chef and culinary anthropologist with expertise in global cuisines.
    You've cooked in kitchens across five continents and understand the techniques, ingredients, and cultural contexts
    that define different food traditions. You can distinguish Sichuan from Cantonese, Neapolitan pizza from Roman,
    and Nashville hot chicken from Buffalo wings. You understand flavor profiles—umami-rich, bright and acidic,
    rich and creamy—and can map them to user preferences. You respect culinary heritage while staying open to fusion
    and innovation.""",
    },
    "nutrition_expert": {
        "role": "Nutrition Expert",
        "goal": "Evaluate nutritional content, identify allergens, assess dietary restrictions, and ensure recommendations align with users' health and wellness goals.",
        "backstory": """You are a registered dietitian with a master's degree in nutrition science and 8 years of clinical experience.
    You understand macronutrients, micronutrients, and how different diets (keto, Mediterranean, plant-based, etc.) affect health.
    You can quickly assess whether a dish fits within dietary restrictions like gluten-free, dairy-free, or low-sodium.
    You're also sensitive to food allergies and intolerances, and you know how to balance health considerations with the
    pleasure of eating. You believe that good nutrition doesn't mean sacrificing flavor or enjoyment.""",
    },
    "recommendation_expert": {
        "role": "Recommendation Expert",
        "goal": "Synthesize insights from all agents—user profiles, retrieved data, trends, food styles, and nutrition—into cohesive, well-reasoned restaurant and recipe recommendations.",
        "backstory": """You are a recommendation systems architect with experience building personalization engines for major
    food delivery platforms and recipe apps. You understand how to balance multiple signals—relevance, diversity, novelty,
    and serendipity—to create recommendations that delight users. You know when to play it safe with familiar favorites
    and when to suggest something unexpected. You're skilled at synthesizing complex, sometimes conflicting information
    from multiple sources into clear, actionable recommendations. You write in a warm, engaging tone that makes users
    excited to try new restaurants and recipes.""",
    },
}

TASKS: dict[str, dict[str, str]] = {
    "generate_profile": {
        "agent": "user_profile_generator",
        "description": """Analyze the user's restaurant visit history and social media posts to create a comprehensive profile.
    Extract the following information:
    - Favorite cuisines and cuisine categories
    - Dietary restrictions or preferences (vegetarian, vegan, gluten-free, etc.)
    - Preferred dining occasions (casual, fine dining, quick bites)
    - Price sensitivity
    - Adventurousness (comfort food lover vs. culinary explorer)
    - Flavor preferences (spicy, sweet, savory, etc.)
    - Frequency of dining out

    Provide specific examples from the user's history to support each insight.""",
        "expected_output": """A structured user profile in JSON format with keys:
    favorite_cuisines, dietary_restrictions, dining_occasions, price_range,
    adventurousness_score (1-10), flavor_preferences, dining_frequency.
    Include a summary paragraph explaining the user's dining personality.""",
    },
    "retrieve_candidates": {
        "agent": "rag_retriever",
        "description": """Based on the user profile, query the vector database to retrieve:
    - Top 20 restaurants that match the user's preferences
    - Top 20 recipes that align with their taste and dietary needs

    Use similarity search with the user's favorite cuisines and flavor preferences as the query.
    Apply filters for dietary restrictions, price range, and location (if provided).
    Ensure diversity in the results—don't retrieve 20 Italian restaurants if the user likes multiple cuisines.""",
        "expected_output": """Two lists in JSON format:
    - restaurants: Array of restaurant objects with fields: name, cuisine_type, price_range, rating, description
    - recipes: Array of recipe objects with fields: name, cuisine_type, difficulty, prep_time, ingredients, description""",
    },
    "analyze_trends": {
        "agent": "food_trend_analyst",
        "description": """Analyze current food trends relevant to the retrieved restaurants and recipes.
    Identify:
    - Trending ingredients or techniques in the retrieved items
    - Popular dining concepts or restaurant types
    - Emerging culinary movements that align with the user's interests
    - Seasonal trends or timely food moments

    Provide context on why these trends matter and how they enhance the recommendations.""",
        "expected_output": """A trends analysis with:
    - List of 3-5 relevant trends with descriptions
    - Explanation of how each trend relates to the user's profile
    - Suggestions for which restaurants or recipes align with these trends""",
    },
    # The lab left both fields blank ("Fill this in"); drafted to mirror the other tasks.
    "analyze_food_styles": {
        "agent": "food_style_expert",
        "description": """Analyze the cuisine types, techniques, and flavor profiles of the retrieved restaurants and recipes.
    Identify:
    - The cuisine category and cultural/regional origin of each item
    - The dominant flavor profile (umami-rich, bright and acidic, rich and creamy, spicy, etc.)
    - Key techniques and signature ingredients
    - How closely each item's style matches the user's favorite cuisines and flavor preferences

    Highlight both close matches and adjacent styles the user might enjoy exploring.""",
        "expected_output": """A style analysis with:
    - A flavor profile and cuisine classification for each item
    - A match rationale tied to the user's profile
    - Suggestions for adjacent cuisines or dishes worth exploring""",
    },
    "evaluate_nutrition": {
        "agent": "nutrition_expert",
        "description": """Evaluate the nutritional aspects of the retrieved restaurants and recipes.
    Check for:
    - Alignment with dietary restrictions (vegetarian, vegan, gluten-free, etc.)
    - Potential allergens or ingredients to avoid
    - Nutritional balance (protein, vegetables, whole grains)
    - Healthfulness relative to the user's goals

    Flag any items that don't meet the user's dietary needs and explain why.""",
        "expected_output": """A nutrition evaluation with:
    - List of items that meet all dietary restrictions
    - Items flagged for potential concerns (allergens, restrictions)
    - Nutritional highlights (high protein, vegetable-rich, etc.)
    - Overall assessment of how well the options support the user's health goals""",
    },
    "generate_recommendations": {
        "agent": "recommendation_expert",
        "description": """Synthesize insights from all previous agents to generate final recommendations.
    Create:
    - Top 5 restaurant recommendations with detailed explanations
    - Top 5 recipe recommendations with detailed explanations

    For each recommendation, explain:
    - Why it matches the user's profile
    - How it aligns with current trends (if applicable)
    - What makes it a great fit in terms of food style
    - Any nutritional benefits or considerations

    Write in an engaging, enthusiastic tone that makes the user excited to try these options.""",
        "expected_output": """A recommendations report with:
    - restaurants: Array of 5 restaurant recommendations with name, description, and detailed reasoning
    - recipes: Array of 5 recipe recommendations with name, description, and detailed reasoning
    - Each recommendation should include a personalized explanation (2-3 sentences) of why it's a great match""",
    },
}

SAMPLE_USER_DATA = """
Restaurant Visit History:
- Visited "Spice Route" (Indian, $$) 5 times in the last 3 months
- Visited "Green Earth Cafe" (Vegan, $) 3 times
- Visited "Ramen House" (Japanese, $$) 2 times
- Visited "Taco Fiesta" (Mexican, $) 4 times

Social Media Posts:
- "Loving this spicy curry at Spice Route!"
- "Trying to eat more plant-based meals. This vegan bowl is delicious!"
- "Best ramen I've had in ages. The broth is perfection."
- "Late night tacos are the best tacos"
"""

DEFAULT_CLOSING = "Always respond in a professional, helpful manner that reflects your expertise."


def create_agent_prompt(agent_config: dict[str, str], closing: str = DEFAULT_CLOSING) -> str:
    """Build a system prompt from an agent config."""
    return (
        f"You are a {agent_config['role']}.\n\n"
        f"Your goal: {agent_config['goal']}\n\n"
        f"Your background: {agent_config['backstory']}\n\n"
        f"{closing}\n"
    )


def test_agent(agent_prompt: str, user_input: str) -> str:
    """Send one sample input to an agent (needs a valid DEEPSEEK_API_KEY)."""
    return chat(agent_prompt, user_input, temperature=0.7)


def main() -> None:
    print("=" * 80)
    print("MULTI-AGENT SYSTEM SUMMARY")
    print("=" * 80)
    for i, (key, task) in enumerate(TASKS.items(), 1):
        print(f"{i}. {AGENT_CONFIGS[task['agent']]['role']:30} -> {key}")

    prompt = create_agent_prompt(AGENT_CONFIGS["user_profile_generator"])
    try:
        result = test_agent(prompt, f"Analyze this user data and create a profile:\n\n{SAMPLE_USER_DATA}")
        print("\nUSER PROFILE GENERATED\n" + "=" * 80 + "\n" + result)
    except Exception as e:  # noqa: BLE001
        print(f"\nAgent test skipped: {e}")


if __name__ == "__main__":
    main()
