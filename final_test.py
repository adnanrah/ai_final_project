from integrator import MealRecommendationSystem
import pandas as pd
import numpy as np
import json
import os
from modify_mdp_planner import modify_mdp_planner
from nbc import EnhancedFoodCategorizer

# Get enhanced version of the MDP planner
EnhancedMealPlannerMDP = modify_mdp_planner()

def load_food_data_from_json(file_path):
    """Load food data from JSON file and convert to DataFrame for NBC training"""
    try:
        with open(file_path, 'r') as f:
            food_data = json.load(f)
        
        all_food_items = []
        food_id_counter = 0
        
        for hall_name, hall_data in food_data.items():
            if 'items' in hall_data:
                for item in hall_data['items']:
                    if isinstance(item, dict):
                        food_item = item.copy()
                        food_item['dining_hall'] = hall_name
                        
                        if 'food_id' not in food_item:
                            food_item['food_id'] = f"F{food_id_counter:03d}"
                            food_id_counter += 1
                        
                        if 'category' not in food_item:
                            name = food_item.get('name', '').lower()
                            ingredients = food_item.get('ingredients', [])
                            
                            if isinstance(ingredients, list):
                                ingredients_str = ' '.join(ingredients).lower()
                            else:
                                ingredients_str = str(ingredients).lower()
                            
                            prelim_categories = []
                            
                            breakfast_terms = ['egg', 'bacon', 'sausage', 'pancake', 'waffle', 
                                              'toast', 'bagel', 'breakfast', 'cereal', 'oatmeal']
                            if any(term in name for term in breakfast_terms) or \
                               any(term in ingredients_str for term in breakfast_terms):
                                prelim_categories.append('breakfast')
                            
                            protein_terms = ['chicken', 'beef', 'pork', 'fish', 'tofu', 'egg', 
                                           'turkey', 'protein', 'meat', 'cheese']
                            if any(term in name for term in protein_terms) or \
                               any(term in ingredients_str for term in protein_terms):
                                if 'protein' in food_item and food_item['protein'] >= 15:
                                    prelim_categories.append('high-protein')
                                else:
                                    prelim_categories.append('balanced')
                            
                            meat_terms = ['chicken', 'beef', 'pork', 'fish', 'meat', 'bacon', 'sausage']
                            if not any(term in name for term in meat_terms) and \
                               not any(term in ingredients_str for term in meat_terms):
                                prelim_categories.append('vegetarian')
                            
                            if not prelim_categories:
                                prelim_categories.append('balanced')
                            
                            food_item['category'] = prelim_categories
                            
                        all_food_items.append(food_item)
                    else:
                        name = item.lower()
                        
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
                            'ingredients': [],
                            'category': prelim_category,
                            'calories': 300,
                            'protein': 15,
                            'fat': 10,
                            'carbs': 30
                        }
                        food_id_counter += 1
                        all_food_items.append(food_item)
        
        food_df = pd.DataFrame(all_food_items)
        
        processed_items = []
        for _, row in food_df.iterrows():
            item = row.to_dict()
            
            if 'food_id' not in item:
                item['food_id'] = f"F{food_id_counter:03d}"
                food_id_counter += 1
            
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
    """Test the meal recommendation system with data from JSON files"""
    print("=== Testing Meal Recommendation System ===")

    system = MealRecommendationSystem()
    
    print("\n1. Loading food data from JSON...")
    
    food_df = pd.DataFrame()
    if os.path.exists("uva_dining_foods_enriched.json"):
        food_df = load_food_data_from_json("uva_dining_foods_enriched.json")
    
    if food_df.empty and os.path.exists("uva_dining_foods.json"):
        food_df = load_food_data_from_json("uva_dining_foods.json")
    
    if food_df.empty:
        json_files = [f for f in os.listdir('.') if f.endswith('.json') and 'uva_dining' in f]
        if json_files:
            food_df = load_food_data_from_json(json_files[0])
    
    if food_df.empty:
        print("No food data found in JSON files. Using mock data instead.")
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
    
    system.food_db = food_df
    print(f"Added {len(food_df)} items to the database")
    
    print("\n2. Testing NBC (Naive Bayes Classifier)...")
    
    system.food_categorizer = EnhancedFoodCategorizer(multi_label=True)
    
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
    
    train_df = pd.concat([pd.DataFrame(pretrain_data), food_df], ignore_index=True)
    system.food_categorizer.train(train_df)
    
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
    
    print("\n4. Testing full workflow with new food items...")
    
    food_items = [
        "Avocado toast with poached eggs on whole grain bread",
        "Quinoa bowl with mixed vegetables and tahini dressing",
        "Berry smoothie with protein powder and almond milk"
    ]
    
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
    
    print("\n5. Testing user feedback loop...")
    
    system.meal_planner.reset_meal_history()
    
    new_plan = system.recommend_meals(days=7)
    
    if new_plan and len(new_plan) > 0 and len(new_plan[0]) > 0:
        breakfast_id = new_plan[0][0].get('food_id')
        print(f"Giving positive feedback (rating 5) to: {new_plan[0][0].get('name')}")
        system.meal_planner.update_from_feedback(breakfast_id, 5)
        
        if len(new_plan[0]) >= 3:
            dinner_id = new_plan[0][2].get('food_id')
            print(f"Giving negative feedback (rating 1) to: {new_plan[0][2].get('name')}")
            system.meal_planner.update_from_feedback(dinner_id, 1)
    
    print("\nGenerating new plan after feedback...")
    feedback_plan = system.recommend_meals(days=7)
    
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