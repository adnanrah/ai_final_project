import json
import time
from datetime import datetime
import sys
import requests
import os

try:
    import openai
except ImportError:
    print("Error: The 'openai' package is not installed.")
    print("Please install it using: pip install openai")
    print("Continuing with requests library as a fallback...")
    openai = None

try:
    from tqdm import tqdm
except ImportError:
    print("Warning: The 'tqdm' package is not installed. Using simple progress indicator instead.")
    print("You can install tqdm using: pip install tqdm")
    class SimpleTqdm:
        def __init__(self, iterable, desc=None):
            self.iterable = iterable
            self.total = len(iterable)
            self.desc = desc
            self.current = 0
        def __iter__(self):
            if self.desc:
                print(f"\n{self.desc}:")
            return self
        def __next__(self):
            if self.current < self.total:
                item = self.iterable[self.current]
                self.current += 1
                if self.current % 5 == 0 or self.current == self.total:
                    percent = (self.current / self.total) * 100
                    print(f"Progress: {self.current}/{self.total} ({percent:.1f}%)")
                return item
            raise StopIteration
    tqdm = SimpleTqdm

API_KEY = os.environ.get("OPENAI_API_KEY", "")

if openai is not None and API_KEY:
    client = openai.OpenAI(api_key=API_KEY)

def load_food_data(filename="uva_dining_foods.json"):
    """Load scraped food data from JSON file"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: The file {filename} is not valid JSON.")
        sys.exit(1)

def save_enriched_data(data, filename="uva_dining_foods_enriched.json"):
    """Save enriched food data to JSON file"""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Enriched data saved to {filename}")

def get_food_info_from_llm(food_name, dining_hall):
    """Get nutritional information using LLM"""
    prompt = f"""
    I need detailed nutritional information about the food item "{food_name}" from a dining hall menu at the University of Virginia.

    Please provide the following information:
    1. A brief description of the dish
    2. Likely main ingredients
    3. Estimated calories per serving
    4. Estimated protein (in grams)
    5. Estimated fat (in grams)
    6. Estimated carbohydrates (in grams)
    
    Please provide this information in a structured format as JSON. If you're uncertain about any values, provide reasonable estimates based on similar dishes.
    
    Example format:
    {{
        "name": "Grilled Chicken Sandwich",
        "description": "A grilled chicken breast served on a toasted bun with lettuce and tomato",
        "ingredients": ["chicken breast", "bun", "lettuce", "tomato"],
        "calories": 320,
        "protein": 28,
        "fat": 12,
        "carbs": 24
    }}
    """
    try:
        if not API_KEY:
            raise Exception("No API key provided. Set OPENAI_API_KEY environment variable.")
        if openai is None:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}"
            }
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "You are a nutritional information expert familiar with common dining hall foods."},
                    {"role": "user", "content": prompt}
                ],
                "response_format": {"type": "json_object"}
            }
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            if response.status_code != 200:
                raise Exception(f"API error: {response.status_code}, {response.text}")
            response_data = response.json()
            nutrition_data = json.loads(response_data["choices"][0]["message"]["content"])
        else:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a nutritional information expert familiar with common dining hall foods."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            nutrition_data = json.loads(response.choices[0].message.content)
        nutrition_data["dining_hall"] = dining_hall
        nutrition_data["retrieved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return nutrition_data
    except Exception as e:
        print(f"Error querying LLM for {food_name}: {e}")
        return {
            "name": food_name,
            "description": "Information not available",
            "ingredients": [],
            "calories": None,
            "protein": None,
            "fat": None,
            "carbs": None,
            "dining_hall": dining_hall,
            "error": str(e),
            "retrieved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

def create_mock_food_info(food_name, dining_hall):
    """Create realistic mock food data when API is unavailable"""
    ingredients = []
    calories = 0
    protein = 0
    fat = 0
    carbs = 0
    description = ""
    food_lower = food_name.lower()
    if "egg" in food_lower:
        description = "A classic breakfast dish made with farm fresh eggs"
        ingredients = ["eggs", "salt", "pepper"]
        calories = 140
        protein = 12
        fat = 10
        carbs = 1
    elif "pancake" in food_lower or "waffle" in food_lower:
        description = "A fluffy breakfast favorite served with syrup"
        ingredients = ["flour", "milk", "eggs", "butter", "sugar"]
        calories = 260
        protein = 6
        fat = 9
        carbs = 40
    elif "bacon" in food_lower or "sausage" in food_lower:
        description = "Savory breakfast meat, cooked until perfectly done"
        ingredients = ["pork", "salt", "spices"]
        calories = 180
        protein = 10
        fat = 16
        carbs = 0
    elif "potato" in food_lower or "fries" in food_lower or "tots" in food_lower:
        description = "Crispy potato side dish, seasoned to perfection"
        ingredients = ["potatoes", "vegetable oil", "salt", "seasonings"]
        calories = 220
        protein = 3
        fat = 12
        carbs = 28
    elif "cheese" in food_lower:
        description = "Flavorful dairy product, perfect for adding to various dishes"
        ingredients = ["milk", "salt", "enzymes"]
        calories = 110
        protein = 7
        fat = 9
        carbs = 1
    elif "spinach" in food_lower or "vegetable" in food_lower:
        description = "Fresh vegetables, providing essential nutrients"
        ingredients = ["fresh vegetables", "olive oil", "seasonings"]
        calories = 45
        protein = 2
        fat = 1
        carbs = 8
    elif "fruit" in food_lower or "berry" in food_lower:
        description = "Sweet and refreshing fruit selection"
        ingredients = ["assorted fruits", "honey"]
        calories = 85
        protein = 1
        fat = 0
        carbs = 22
    elif "toast" in food_lower or "bread" in food_lower or "bagel" in food_lower:
        description = "Freshly baked bread product"
        ingredients = ["flour", "yeast", "salt", "water"]
        calories = 160
        protein = 5
        fat = 2
        carbs = 30
    elif "smoothie" in food_lower:
        description = "Refreshing blended beverage made with fruit and dairy"
        ingredients = ["fruits", "yogurt", "milk", "ice"]
        calories = 180
        protein = 5
        fat = 3
        carbs = 35
    else:
        description = f"A dining hall dish named {food_name}"
        ingredients = ["various ingredients"]
        calories = 200
        protein = 8
        fat = 7
        carbs = 25
    return {
        "name": food_name,
        "description": description,
        "ingredients": ingredients,
        "calories": calories,
        "protein": protein,
        "fat": fat,
        "carbs": carbs,
        "dining_hall": dining_hall,
        "retrieved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "note": "This is mock data generated for demonstration purposes"
    }

def enrich_food_data(food_data, use_mock=False):
    """Add nutritional info to food items"""
    enriched_data = {}
    total_items = sum(len(hall_data["items"]) for hall_name, hall_data in food_data.items())
    processed_items = 0
    print(f"Starting to enrich {total_items} food items with {'mock' if use_mock else 'LLM'} data...")
    for hall_name, hall_data in food_data.items():
        print(f"\nProcessing {hall_name}...")
        enriched_data[hall_name] = {
            "timestamp": hall_data["timestamp"],
            "items": []
        }
        for food_name in tqdm(hall_data["items"], desc=hall_name):
            if use_mock:
                food_info = create_mock_food_info(food_name, hall_name)
            else:
                food_info = get_food_info_from_llm(food_name, hall_name)
            enriched_data[hall_name]["items"].append(food_info)
            processed_items += 1
            if not use_mock:
                time.sleep(0.5)
            if not isinstance(tqdm, type) and processed_items % 10 == 0:
                print(f"Processed {processed_items}/{total_items} items")
    return enriched_data

def main():
    print("Starting food data enrichment process...")
    use_mock = False
    if not API_KEY:
        print("WARNING: No OpenAI API key found in environment variables.")
        print("Continuing with mock data generation instead of API calls.")
        use_mock = True
    food_data = load_food_data()
    enriched_data = enrich_food_data(food_data, use_mock=use_mock)
    save_enriched_data(enriched_data)
    if use_mock:
        print("\nNOTE: The saved data contains mock nutritional information for demonstration purposes.")
        print("To get actual nutritional estimates, set the OPENAI_API_KEY environment variable.")
    print("Food data enrichment complete!")

if __name__ == "__main__":
    main()