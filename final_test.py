from integrator import MealRecommendationSystem
import pandas as pd
import numpy as np
import json
import os

def load_food_data_from_json(file_path):
    """
    Load food data from JSON file and convert to DataFrame format
    suitable for NBC training
    """
    try:
        with open(file_path, 'r') as f:
            food_data = json.load(f)
        
        # Create a list to hold all food items
        all_food_items = []
        food_id_counter = 0
        
        # Process each dining hall
        for hall_name, hall_data in food_data.items():
            if 'items' in hall_data:
                for item in hall_data['items']:
                    # For enriched data format
                    if isinstance(item, dict):
                        food_item = item.copy()
                        # Ensure the dining hall is set
                        food_item['dining_hall'] = hall_name
                        
                        # Generate placeholder ID if missing
                        if 'food_id' not in food_item:
                            food_item['food_id'] = f"F{food_id_counter:03d}"
                            food_id_counter += 1
                            
                        # Assign default category
                        food_item['category'] = ['healthy']
                            
                        all_food_items.append(food_item)
                    # For simple list of food names
                    else:
                        food_item = {
                            'name': item,
                            'description': f"{item} from {hall_name}",
                            'food_id': f"F{food_id_counter:03d}",
                            'dining_hall': hall_name,
                            'ingredients': [],  # Empty placeholder
                            'category': ['healthy'],  # Default category
                            # Assign reasonable mock values for nutrition
                            'calories': 300,
                            'protein': 15,
                            'fat': 10,
                            'carbs': 30
                        }
                        food_id_counter += 1
                        all_food_items.append(food_item)
        
        # Convert to DataFrame
        food_df = pd.DataFrame(all_food_items)
        
        # Process specific fields for compatibility
        processed_items = []
        for item in all_food_items:
            processed_item = item.copy()
            
            # Make sure categories are lists
            if 'category' not in processed_item or processed_item['category'] is None:
                processed_item['category'] = ['healthy']
            elif isinstance(processed_item['category'], str):
                processed_item['category'] = [processed_item['category']]
                
            # Make sure ingredients are lists
            if 'ingredients' not in processed_item or processed_item['ingredients'] is None:
                processed_item['ingredients'] = []
            elif isinstance(processed_item['ingredients'], str):
                processed_item['ingredients'] = [processed_item['ingredients']]
                
            # Make sure name and description are strings
            for field in ['name', 'description']:
                if field not in processed_item or processed_item[field] is None:
                    processed_item[field] = ""
                processed_item[field] = str(processed_item[field])
                
            # Create a full_description field
            processed_item['full_description'] = f"{processed_item['name']} {processed_item['description']}"
            
            processed_items.append(processed_item)
            
        # Create a new DataFrame with processed items
        food_df = pd.DataFrame(processed_items)
        
        print(f"Loaded {len(food_df)} food items from {file_path}")
        return food_df
        
    except Exception as e:
        print(f"Error loading food data from {file_path}: {e}")
        return pd.DataFrame()

def test_meal_recommendation_system():
    """
    Test the meal recommendation system with data from JSON files
    """
    print("=== Testing Meal Recommendation System ===")

    # Initialize the system
    system = MealRecommendationSystem()
    
    # 1. Load food data from JSON files
    print("\n1. Loading food data from JSON...")
    
    # Try to load enriched food data first
    food_df = pd.DataFrame()
    if os.path.exists("uva_dining_foods_enriched.json"):
        food_df = load_food_data_from_json("uva_dining_foods_enriched.json")
    
    # If no enriched data or it's empty, try the regular food data
    if food_df.empty and os.path.exists("uva_dining_foods.json"):
        food_df = load_food_data_from_json("uva_dining_foods.json")
    
    # If we still don't have data, use the backup JSON
    if food_df.empty:
        # Try any other JSON files we have
        json_files = [f for f in os.listdir('.') if f.endswith('.json') and 'uva_dining' in f]
        if json_files:
            food_df = load_food_data_from_json(json_files[0])
    
    # If still no data, use mock data
    if food_df.empty:
        print("No food data found in JSON files. Using mock data instead.")
        # Create some mock food data
        food_data = [
            {
                'food_id': 'B001',
                'name': 'Oatmeal with Berries',
                'price': 4.50,
                'calories': 320,
                'protein': 12,
                'fat': 6,
                'carbs': 54,
                'category': ['healthy', 'balanced'],
                'ingredients': ['oats', 'milk', 'berries', 'honey'],
                'description': 'Hearty breakfast oatmeal with fresh berries'
            },
            # Add more mock data as needed
        ]
        food_df = pd.DataFrame(food_data)
    
    # Add food data to system
    system.food_db = food_df
    print(f"Added {len(food_df)} items to the database")
    
    # 2. Test NBC component
    print("\n2. Testing NBC (Naive Bayes Classifier)...")
    system.initialize_categorizer(multi_label=True)
    
    # Test categorizing new foods
    new_foods = [
        "Greek yogurt with berries and honey, quick breakfast option",
        "Cheeseburger with french fries, high calorie meal",
        "Grilled tofu with vegetables and quinoa, plant-based meal"
    ]
    
    categorized_foods = system.categorize_foods(new_foods)
    print("NBC categorization results:")
    for food in categorized_foods:
        if 'full_description' in food:
            food_desc = food['full_description'][:30]
        elif 'description' in food:
            food_desc = food['description'][:30]
        else:
            food_desc = str(food)[:30]
        
        pred_cat = food.get('predicted_category', 'No category')
        print(f"- {food_desc}... → {pred_cat}")
    
    # 3. Test MDP component
    print("\n3. Testing MDP (Markov Decision Process)...")
    user_preferences = {
        'dietary_restrictions': [],
        'budget': 10.0,
        'nutrition_goals': {
            'calories': 2000,
            'protein': 120,
            'fat': 65,
            'carbs': 250
        },
        'meal_preferences': {
            'breakfast': ['high-protein', 'quick'],
            'lunch': ['balanced', 'vegetarian'],
            'dinner': ['high-protein', 'low-carb']
        }
    }
    
    system.initialize_meal_planner(user_preferences)
    
    meal_plan = system.recommend_meals(days=5)
    print("MDP meal plan results:")
    for day, meals in enumerate(meal_plan, 1):
        print(f"\nDay {day}:")
        for i, meal in enumerate(meals):
            meal_type = ["Breakfast", "Lunch", "Dinner"][i]
            print(f"  {meal_type}: {meal['name']}")
            if 'category' in meal:
                categories = meal['category']
                if isinstance(categories, list):
                    categories = ', '.join(categories)
                print(f"    Category: {categories}")
            print(f"    Nutrition: {meal.get('calories', 'N/A')} cal, {meal.get('protein', 'N/A')}g protein")
    
    # 4. Test the full workflow with new food items
    print("\n4. Testing full workflow with new food items...")
    
    # Foods to add
    food_items = [
        "Avocado toast with poached eggs on whole grain bread",
        "Quinoa bowl with mixed vegetables and tahini dressing",
        "Berry smoothie with protein powder and almond milk"
    ]
    
    # Run the workflow
    results = system.run_full_workflow(food_items, user_preferences)
    
    if results['success']:
        print(f"Successfully processed {results['new_foods']} new food items")
        print(f"Categorized {results['categorized_foods']} foods")
        
        if results['meal_plan']:
            print("\nUpdated meal plan with new foods:")
            for day, meals in enumerate(results['meal_plan'], 1):
                print(f"\nDay {day}:")
                for i, meal in enumerate(meals):
                    meal_type = ["Breakfast", "Lunch", "Dinner"][i]
                    print(f"  {meal_type}: {meal['name']}")
    else:
        print("Workflow had errors:")
        for error in results['errors']:
            print(f"- {error}")
    
    # 5. Test user feedback
    print("\n5. Testing user feedback loop...")
    
    # Get a food ID from history
    if system.user_history:
        food_id = system.user_history[0]['food_id']
        print(f"Providing feedback for: {food_id}")
        
        # Give a high rating
        system.process_user_feedback(food_id, rating=5, consumed=True)
        print("Feedback processed - high rating (5)")
        
        # Get another food ID and give a low rating
        if len(system.user_history) > 1:
            food_id = system.user_history[1]['food_id']
            print(f"Providing feedback for: {food_id}")
            system.process_user_feedback(food_id, rating=2, consumed=True)
            print("Feedback processed - low rating (2)")
    
    print("\n=== Test completed successfully ===")
    return system

if __name__ == "__main__":
    test_meal_recommendation_system()