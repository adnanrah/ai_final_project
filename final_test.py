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
                            food_item['food_id'] = f"F{len(all_food_items):03d}"
                            
                        all_food_items.append(food_item)
                    # For simple list of food names
                    else:
                        food_item = {
                            'name': item,
                            'description': f"{item} from {hall_name}",
                            'food_id': f"F{len(all_food_items):03d}",
                            'dining_hall': hall_name,
                            'ingredients': [],  # Empty placeholder
                            # Assign reasonable mock values for nutrition
                            'calories': 300,
                            'protein': 15,
                            'fat': 10,
                            'carbs': 30
                        }
                        all_food_items.append(food_item)
        
        # Convert to DataFrame
        food_df = pd.DataFrame(all_food_items)
        
        # Ensure columns needed by NBC exist
        if 'ingredients' not in food_df.columns:
            food_df['ingredients'] = food_df['name'].apply(lambda x: [])
            
        for col in ['calories', 'protein', 'fat', 'carbs']:
            if col not in food_df.columns:
                food_df[col] = None
        
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
            {
                'food_id': 'B002',
                'name': 'Protein Pancakes',
                'price': 6.75,
                'calories': 450,
                'protein': 25,
                'fat': 12,
                'carbs': 45,
                'category': ['high-protein', 'balanced'],
                'ingredients': ['flour', 'protein powder', 'eggs', 'milk'],
                'description': 'Fluffy pancakes with added protein powder'
            },
            {
                'food_id': 'L001',
                'name': 'Grilled Chicken Salad',
                'price': 8.50,
                'calories': 380,
                'protein': 32,
                'fat': 15,
                'carbs': 22,
                'category': ['high-protein', 'low-carb'],
                'ingredients': ['chicken', 'lettuce', 'tomato', 'cucumber', 'olive oil'],
                'description': 'Fresh salad with grilled chicken breast'
            },
            {
                'food_id': 'L002',
                'name': 'Vegetarian Wrap',
                'price': 7.25,
                'calories': 420,
                'protein': 18,
                'fat': 14,
                'carbs': 52,
                'category': ['vegetarian', 'balanced'],
                'ingredients': ['tortilla', 'hummus', 'lettuce', 'tomato', 'cucumber', 'avocado'],
                'description': 'Whole grain wrap with hummus and fresh vegetables'
            },
            {
                'food_id': 'D001',
                'name': 'Salmon with Roasted Vegetables',
                'price': 12.50,
                'calories': 480,
                'protein': 38,
                'fat': 22,
                'carbs': 25,
                'category': ['high-protein', 'low-carb'],
                'ingredients': ['salmon', 'broccoli', 'carrots', 'olive oil', 'garlic'],
                'description': 'Baked salmon fillet with seasonal roasted vegetables'
            },
            {
                'food_id': 'D002',
                'name': 'Spaghetti with Meatballs',
                'price': 9.75,
                'calories': 720,
                'protein': 35,
                'fat': 28,
                'carbs': 75,
                'category': ['high-protein', 'balanced'],
                'ingredients': ['pasta', 'beef', 'tomato sauce', 'garlic', 'onion'],
                'description': 'Classic spaghetti with homemade beef meatballs'
            },
            {
                'food_id': 'B003',
                'name': 'Avocado Toast',
                'price': 5.50,
                'calories': 350,
                'protein': 12,
                'fat': 18,
                'carbs': 30,
                'category': ['balanced', 'vegetarian'],
                'ingredients': ['bread', 'avocado', 'olive oil', 'salt'],
                'description': 'Toasted whole grain bread with smashed avocado'
            },
            {
                'food_id': 'L003',
                'name': 'Lentil Soup',
                'price': 6.25,
                'calories': 280,
                'protein': 16,
                'fat': 8,
                'carbs': 40,
                'category': ['vegetarian', 'balanced'],
                'ingredients': ['lentils', 'carrots', 'celery', 'onion', 'broth'],
                'description': 'Hearty soup made with green lentils and vegetables'
            },
            {
                'food_id': 'D003',
                'name': 'Beef Stir Fry',
                'price': 11.00,
                'calories': 490,
                'protein': 30,
                'fat': 20,
                'carbs': 42,
                'category': ['high-protein', 'balanced'],
                'ingredients': ['beef', 'bell peppers', 'broccoli', 'soy sauce', 'rice'],
                'description': 'Stir-fried beef strips with vegetables over rice'
            },
            {
                'food_id': 'B004',
                'name': 'Fruit Smoothie',
                'price': 5.00,
                'calories': 250,
                'protein': 8,
                'fat': 3,
                'carbs': 50,
                'category': ['healthy', 'vegetarian'],
                'ingredients': ['banana', 'berries', 'yogurt', 'honey', 'milk'],
                'description': 'Creamy smoothie with mixed fruits and yogurt'
            }
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
        print(f"- {food.get('full_description', '')[:30]}... → {food.get('predicted_category', '')}")
    
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
    
    meal_plan = system.recommend_meals(days=2)
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