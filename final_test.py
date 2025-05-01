from integrator import MealRecommendationSystem
import pandas as pd
import numpy as np
import json
import os
from modify_mdp_planner import modify_mdp_planner
from nbc import EnhancedFoodCategorizer  # Add this import

# Apply the MDP modifications
EnhancedMealPlannerMDP = modify_mdp_planner()

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
                        
                        # If category exists, keep it; otherwise it will be determined by the classifier
                        if 'category' not in food_item:
                            # Try to infer a preliminary category from the name or ingredients
                            # This will help the classifier have some initial data to work with
                            name = food_item.get('name', '').lower()
                            ingredients = food_item.get('ingredients', [])
                            
                            if isinstance(ingredients, list):
                                ingredients_str = ' '.join(ingredients).lower()
                            else:
                                ingredients_str = str(ingredients).lower()
                            
                            # Make a preliminary categorization guess
                            prelim_categories = []
                            
                            # Check for breakfast items
                            breakfast_terms = ['egg', 'bacon', 'sausage', 'pancake', 'waffle', 
                                              'toast', 'bagel', 'breakfast', 'cereal', 'oatmeal']
                            if any(term in name for term in breakfast_terms) or \
                               any(term in ingredients_str for term in breakfast_terms):
                                prelim_categories.append('breakfast')
                            
                            # Check for protein-rich items
                            protein_terms = ['chicken', 'beef', 'pork', 'fish', 'tofu', 'egg', 
                                           'turkey', 'protein', 'meat', 'cheese']
                            if any(term in name for term in protein_terms) or \
                               any(term in ingredients_str for term in protein_terms):
                                if 'protein' in food_item and food_item['protein'] >= 15:
                                    prelim_categories.append('high-protein')
                                else:
                                    prelim_categories.append('balanced')
                            
                            # Check for vegetarian items
                            meat_terms = ['chicken', 'beef', 'pork', 'fish', 'meat', 'bacon', 'sausage']
                            if not any(term in name for term in meat_terms) and \
                               not any(term in ingredients_str for term in meat_terms):
                                prelim_categories.append('vegetarian')
                            
                            # If we have no categories yet, assign balanced as a default
                            if not prelim_categories:
                                prelim_categories.append('balanced')
                            
                            food_item['category'] = prelim_categories
                            
                        all_food_items.append(food_item)
                    # For simple list of food names
                    else:
                        # For simple strings, make a more intelligent initial category assessment
                        name = item.lower()
                        
                        # Determine preliminary category based on food name
                        if any(term in name for term in ['egg', 'bacon', 'sausage', 'pancake', 'waffle', 'toast']):
                            prelim_category = ['breakfast']
                        elif any(term in name for term in ['salad', 'vegetable', 'fruit']):
                            prelim_category = ['healthy']
                        elif any(term in name for term in ['chicken', 'beef', 'pork', 'turkey']):
                            prelim_category = ['high-protein']
                        else:
                            prelim_category = ['balanced']
                        
                        food_item = {
                            'name': item,
                            'description': f"{item} from {hall_name}",
                            'food_id': f"F{food_id_counter:03d}",
                            'dining_hall': hall_name,
                            'ingredients': [],  # Empty placeholder
                            'category': prelim_category,
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
        for _, row in food_df.iterrows():
            item = row.to_dict()
            
            # Ensure required fields exist
            if 'food_id' not in item:
                item['food_id'] = f"F{food_id_counter:03d}"
                food_id_counter += 1
            
            # Ensure category is present, but don't override existing categories
            if 'category' not in item:
                item['category'] = ['balanced']
            
            if 'ingredients' not in item:
                item['ingredients'] = []
            
            processed_items.append(item)
        
        return pd.DataFrame(processed_items)
        
    except Exception as e:
        print(f"Error loading food data: {e}")
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
                'category': ['breakfast', 'balanced'],
                'ingredients': ['oats', 'milk', 'berries', 'honey'],
                'description': 'Hearty breakfast oatmeal with fresh berries'
            },
            {
                'food_id': 'L001',
                'name': 'Grilled Chicken Salad',
                'price': 8.50,
                'calories': 380,
                'protein': 32,
                'fat': 15,
                'carbs': 22,
                'category': ['high-protein', 'healthy'],
                'ingredients': ['chicken', 'lettuce', 'tomato', 'cucumber', 'olive oil'],
                'description': 'Fresh salad with grilled chicken'
            },
            {
                'food_id': 'D001',
                'name': 'Tofu Stir-fry',
                'price': 9.25,
                'calories': 320,
                'protein': 18,
                'fat': 14,
                'carbs': 28,
                'category': ['vegetarian', 'vegan', 'healthy'],
                'ingredients': ['tofu', 'broccoli', 'carrots', 'bell pepper', 'soy sauce'],
                'description': 'Tofu with mixed vegetables'
            },
        ]
        food_df = pd.DataFrame(food_data)
    
    # Add food data to system
    system.food_db = food_df
    print(f"Added {len(food_df)} items to the database")
    
    # 2. Test NBC component
    print("\n2. Testing NBC (Naive Bayes Classifier)...")
    
    # Create and pre-train the categorizer with sample data
    system.food_categorizer = EnhancedFoodCategorizer(multi_label=True)
    
    # Add sample training data to help classifier recognize food categories
    pretrain_data = [
        {
            'name': 'Scrambled Eggs',
            'description': 'Fluffy scrambled eggs cooked with butter',
            'ingredients': ['eggs', 'milk', 'butter', 'salt', 'pepper'],
            'calories': 210,
            'protein': 14,
            'fat': 16,
            'carbs': 2,
            'category': ['breakfast', 'high-protein']
        },
        {
            'name': 'Grilled Chicken Salad',
            'description': 'Fresh salad with grilled chicken breast',
            'ingredients': ['chicken', 'lettuce', 'tomato', 'cucumber', 'olive oil'],
            'calories': 380,
            'protein': 32,
            'fat': 15,
            'carbs': 22,
            'category': ['high-protein', 'healthy']
        },
        {
            'name': 'Yogurt and Berries',
            'description': 'Greek yogurt with mixed berries and honey',
            'ingredients': ['yogurt', 'strawberries', 'blueberries', 'honey'],
            'calories': 220,
            'protein': 12,
            'fat': 8,
            'carbs': 25,
            'category': ['breakfast', 'healthy']
        },
        {
            'name': 'Tofu Stir-fry',
            'description': 'Stir-fried tofu with mixed vegetables',
            'ingredients': ['tofu', 'broccoli', 'carrots', 'bell pepper', 'soy sauce'],
            'calories': 320,
            'protein': 18,
            'fat': 14,
            'carbs': 28,
            'category': ['vegetarian', 'vegan', 'healthy']
        },
        {
            'name': 'Bagel with Cream Cheese',
            'description': 'Fresh bagel with cream cheese spread',
            'ingredients': ['bagel', 'cream cheese'],
            'calories': 350,
            'protein': 12,
            'fat': 9,
            'carbs': 54,
            'category': ['breakfast']
        }
    ]
    
    # Train on combined data
    train_df = pd.concat([pd.DataFrame(pretrain_data), food_df], ignore_index=True)
    system.food_categorizer.train(train_df)
    
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
        if isinstance(pred_cat, list):
            pred_cat = ', '.join(pred_cat)
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
    
    # Use the enhanced MDP planner
    system.meal_planner = EnhancedMealPlannerMDP(system.food_db, user_preferences)
    
    meal_plan = system.recommend_meals(days=10)
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
    
    # 4. Testing full workflow with new food items
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
                    if 'category' in meal:
                        categories = meal['category']
                        if isinstance(categories, list):
                            categories = ', '.join(categories)
                        print(f"    Category: {categories}")
    else:
        print("Workflow had errors:")
        for error in results['errors']:
            print(f"- {error}")
    
    # 5. Testing user feedback
    print("\n5. Testing user feedback loop...")
    
    # Reset meal history to start fresh
    system.meal_planner.reset_meal_history()
    
    # Generate a new plan
    new_plan = system.recommend_meals(days=7)
    
    # Give feedback on a few items
    if new_plan and len(new_plan) > 0 and len(new_plan[0]) > 0:
        # Like the first breakfast
        breakfast_id = new_plan[0][0].get('food_id')
        print(f"Giving positive feedback (rating 5) to: {new_plan[0][0].get('name')}")
        system.meal_planner.update_from_feedback(breakfast_id, 5)
        
        # Dislike the first dinner
        if len(new_plan[0]) >= 3:
            dinner_id = new_plan[0][2].get('food_id')
            print(f"Giving negative feedback (rating 1) to: {new_plan[0][2].get('name')}")
            system.meal_planner.update_from_feedback(dinner_id, 1)
    
    # Generate another plan after feedback
    print("\nGenerating new plan after feedback...")
    feedback_plan = system.recommend_meals(days=7)
    
    # Print the new plan
    for day, meals in enumerate(feedback_plan, 1):
        print(f"\nDay {day}:")
        for i, meal in enumerate(meals):
            meal_type = ["Breakfast", "Lunch", "Dinner"][i]
            print(f"  {meal_type}: {meal.get('name')}")
            if 'category' in meal:
                categories = meal['category']
                if isinstance(categories, list):
                    categories = ', '.join(categories)
                print(f"    Category: {categories}")

if __name__ == "__main__":
    test_meal_recommendation_system()