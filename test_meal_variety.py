import pandas as pd
import numpy as np
from datetime import datetime
import json

# Import your MDP class and apply the modifications
from mdp import MealPlannerMDP
from modify_mdp_planner import modify_mdp_planner

# Apply the modifications
EnhancedMealPlannerMDP = modify_mdp_planner()

def test_meal_variety():
    """Test if the MDP planner provides varied meal recommendations"""
    # Load the food data
    try:
        with open("uva_dining_foods_enriched.json", 'r') as f:
            food_data = json.load(f)
    except FileNotFoundError:
        print("Could not find enriched food data. Using basic food data.")
        try:
            with open("uva_dining_foods.json", 'r') as f:
                food_data = json.load(f)
        except FileNotFoundError:
            print("No food data found. Using mock data.")
            # Create mock data
            return
    
    # Convert to DataFrame format
    all_food_items = []
    for hall_name, hall_data in food_data.items():
        if 'items' in hall_data:
            for item in hall_data['items']:
                if isinstance(item, dict):
                    # Add dining hall info
                    item['dining_hall'] = hall_name
                    
                    # Generate food ID if not present
                    if 'food_id' not in item:
                        item['food_id'] = f"F{len(all_food_items):03d}"
                    
                    all_food_items.append(item)
    
    # Create DataFrame
    food_df = pd.DataFrame(all_food_items)
    print(f"Loaded {len(food_df)} food items")
    
    # Define user preferences
    user_prefs = {
        'dietary_restrictions': [],
        'budget': 15.0,
        'nutrition_goals': {
            'calories': 2000,
            'protein': 120,
            'fat': 65,
            'carbs': 250
        }
    }
    
    # Create and train the enhanced MDP planner
    planner = EnhancedMealPlannerMDP(food_df, user_prefs)
    planner.value_iteration()
    planner.extract_policy()
    
    # Generate a 5-day meal plan
    print("\nGenerating meal plan for 5 days...")
    meal_plan = planner.plan_meals(days=5)
    
    # Track food items to check for repetition
    recommended_items = {}
    
    # Print the plan and check for variety
    for day, meals in enumerate(meal_plan, 1):
        print(f"\nDay {day}:")
        for i, meal in enumerate(meals):
            meal_type = ["Breakfast", "Lunch", "Dinner"][i]
            name = meal.get('name', 'Unknown')
            print(f"  {meal_type}: {name}")
            
            # Track repetition
            if name not in recommended_items:
                recommended_items[name] = 1
            else:
                recommended_items[name] += 1
    
    # Check for repetition
    repeat_count = sum(1 for count in recommended_items.values() if count > 1)
    total_items = len(meal_plan) * 3  # 3 meals per day
    unique_items = len(recommended_items)
    
    print(f"\nVariety Analysis:")
    print(f"Total meals recommended: {total_items}")
    print(f"Unique items recommended: {unique_items}")
    print(f"Items that repeated: {repeat_count}")
    print(f"Variety score: {unique_items/total_items:.2f} (higher is better)")
    
    # Test user feedback
    print("\nTesting user feedback...")
    
    # Reset meal history to start fresh
    planner.reset_meal_history()
    
    # Generate a new plan
    new_plan = planner.plan_meals(days=2)
    
    # Give feedback on a few items
    if new_plan and new_plan[0]:
        # Like the first breakfast
        breakfast_id = new_plan[0][0].get('food_id')
        print(f"Giving positive feedback (rating 5) to: {new_plan[0][0].get('name')}")
        planner.update_from_feedback(breakfast_id, 5)
        
        # Dislike the first dinner
        if len(new_plan[0]) >= 3:
            dinner_id = new_plan[0][2].get('food_id')
            print(f"Giving negative feedback (rating 1) to: {new_plan[0][2].get('name')}")
            planner.update_from_feedback(dinner_id, 1)
    
    # Generate another plan after feedback
    print("\nGenerating new plan after feedback...")
    feedback_plan = planner.plan_meals(days=2)
    
    # Print the new plan
    for day, meals in enumerate(feedback_plan, 1):
        print(f"\nDay {day}:")
        for i, meal in enumerate(meals):
            meal_type = ["Breakfast", "Lunch", "Dinner"][i]
            print(f"  {meal_type}: {meal.get('name')}")

if __name__ == "__main__":
    test_meal_variety()