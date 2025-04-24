import json
import time
from datetime import datetime
import sys
import requests

# Try importing required packages with error handling
try:
    import openai
except ImportError:
    print("Error: The 'openai' package is not installed.")
    print("Please install it using: pip install openai")
    print("Continuing with requests library as a fallback...")
    openai = None

# Create a simple progress indicator if tqdm is not available
try:
    from tqdm import tqdm
except ImportError:
    print("Warning: The 'tqdm' package is not installed. Using simple progress indicator instead.")
    print("You can install tqdm using: pip install tqdm")
    
    # Define a simple progress class as fallback
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
                # Print progress every 5 items
                if self.current % 5 == 0 or self.current == self.total:
                    percent = (self.current / self.total) * 100
                    print(f"Progress: {self.current}/{self.total} ({percent:.1f}%)")
                return item
            raise StopIteration
    
    tqdm = SimpleTqdm

# API key - replace with your actual API key (don't include "Bearer " prefix)
API_KEY = "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"  # Replace this with your actual key

# Initialize the OpenAI client if the package is available
if openai is not None:
    client = openai.OpenAI(api_key=API_KEY)

def load_food_data(filename="uva_dining_foods.json"):
    """Load the scraped food data from JSON file"""
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
    """Save the enriched food data to a new JSON file"""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Enriched data saved to {filename}")

def get_food_info_from_llm(food_name, dining_hall):
    """
    Query the LLM to get nutritional information for a food item.
    
    Returns a dictionary with the structured nutritional information.
    """
    # Create the prompt for the LLM
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
        # Handle the case where openai package is not available or not initialized
        if openai is None:
            # Fallback to using requests library
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
            # Use the openai package if available
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a nutritional information expert familiar with common dining hall foods."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            # Extract and parse the response
            nutrition_data = json.loads(response.choices[0].message.content)
        
        # Add the dining hall information
        nutrition_data["dining_hall"] = dining_hall
        
        # Add timestamp
        nutrition_data["retrieved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return nutrition_data
        
    except Exception as e:
        print(f"Error querying LLM for {food_name}: {e}")
        # Return a basic structure with error information
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
    """
    Create mock food information for demo mode.
    This function is used when no API key is provided.
    """
    # Create realistic but fake nutritional information
    ingredients = []
    calories = 0
    protein = 0
    fat = 0
    carbs = 0
    description = ""
    
    # Generate data based on food name
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
        # Generic response for anything else
        description = f"A dining hall dish named {food_name}"
        ingredients = ["various ingredients"]
        calories = 200
        protein = 8
        fat = 7
        carbs = 25
    
    # Return mock data
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
    """
    Enrich the food data with nutritional information from the LLM.
    
    Parameters:
    - food_data: Dictionary containing food items by dining hall
    - use_mock: If True, generates mock data instead of calling the API
    
    Returns a dictionary with the same structure but with enriched food items.
    """
    enriched_data = {}
    
    # Keep track of total items to process
    total_items = sum(len(hall_data["items"]) for hall_name, hall_data in food_data.items())
    processed_items = 0
    
    print(f"Starting to enrich {total_items} food items with {'mock' if use_mock else 'LLM'} data...")
    
    # Process each dining hall
    for hall_name, hall_data in food_data.items():
        print(f"\nProcessing {hall_name}...")
        
        enriched_data[hall_name] = {
            "timestamp": hall_data["timestamp"],
            "items": []
        }
        
        # Process each food item
        for food_name in tqdm(hall_data["items"], desc=hall_name):
            # Get nutritional information
            if use_mock:
                food_info = create_mock_food_info(food_name, hall_name)
            else:
                food_info = get_food_info_from_llm(food_name, hall_name)
            
            # Add to enriched data
            enriched_data[hall_name]["items"].append(food_info)
            
            processed_items += 1
            
            # Sleep briefly to avoid rate limiting (only if using API)
            if not use_mock:
                time.sleep(0.5)
            
            # Display progress (only if not using tqdm)
            if not isinstance(tqdm, type) and processed_items % 10 == 0:
                print(f"Processed {processed_items}/{total_items} items")
    
    return enriched_data

def main():
    print("Starting food data enrichment process...")
    
    # Check if API key is set
    use_mock = False
    if API_KEY == "your-actual-api-key-here":
        print("WARNING: You need to replace 'your-actual-api-key-here' with your actual OpenAI API key.")
        print("Continuing with mock data generation instead of API calls.")
        use_mock = True
    
    # Load the scraped food data
    food_data = load_food_data()
    
    # Enrich the food data
    enriched_data = enrich_food_data(food_data, use_mock=use_mock)
    
    # Save the enriched data
    save_enriched_data(enriched_data)
    
    if use_mock:
        print("\nNOTE: The saved data contains mock nutritional information for demonstration purposes.")
        print("To get actual nutritional estimates, update the API_KEY variable with your OpenAI API key.")
    
    print("Food data enrichment complete!")

if __name__ == "__main__":
    main()