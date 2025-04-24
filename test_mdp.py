from mdp import MealPlannerMDP, demo_meal_planner
import pandas as pd

# Option 1: Run the demo planner
planner = demo_meal_planner()

# Option 2: Test with custom data
"""
# Create sample food database
food_data = [
    {
        'food_id': 'B001',
        'name': 'Oatmeal with Fruit',
        'price': 4.50,
        'calories': 320,
        'protein': 12,
        'fat': 6,
        'carbs': 54,
        'category': ['healthy', 'balanced'],
        'ingredients': ['oats', 'milk', 'berries', 'honey'],
        'meal_type': 'breakfast'
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
        'meal_type': 'lunch'
    },
    {
        'food_id': 'D001',
        'name': 'Baked Salmon',
        'price': 12.50,
        'calories': 480,
        'protein': 38,
        'fat': 22,
        'carbs': 25,
        'category': ['high-protein', 'low-carb'],
        'ingredients': ['salmon', 'broccoli', 'olive oil', 'garlic'],
        'meal_type': 'dinner'
    }
]

# Create user preferences
user_prefs = {
    'dietary_restrictions': [],
    'budget': 15.0,
    'nutrition_goals': {
        'calories': 1800,
        'protein': 110,
        'fat': 60,
        'carbs': 200
    }
}

# Create food database
food_df = pd.DataFrame(food_data)

# Create and train MDP planner
planner = MealPlannerMDP(food_df, user_prefs)
planner.value_iteration()
planner.extract_policy()

# Generate meal plan
meal_plan = planner.plan_meals(days=2)

# Print the plan
print("\nGenerated 2-Day Meal Plan:")
for day, meals in enumerate(meal_plan, 1):
    print(f"\nDay {day}:")
    for i, meal in enumerate(meals):
        meal_type = ["Breakfast", "Lunch", "Dinner"][i]
        print(f"  {meal_type}: {meal['name']}")
        print(f"    Calories: {meal['calories']}, Protein: {meal['protein']}g")
"""