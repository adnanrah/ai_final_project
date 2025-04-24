from integrator import MealRecommendationSystem, demo_meal_recommendation_system

# Option 1: Run the demo function
system = demo_meal_recommendation_system()
print("\nTesting complete!")

# Option 2 (more detailed): Set up your own test
"""
# Initialize system
system = MealRecommendationSystem()

# Sample food items to test
food_items = [
    "Grilled chicken breast with roasted vegetables and quinoa",
    "Greek yogurt with honey and mixed berries",
    "Lentil soup with whole grain bread"
]

# Sample user preferences
user_preferences = {
    'dietary_restrictions': [],
    'budget': 15.0,
    'nutrition_goals': {
        'calories': 2000,
        'protein': 120,
        'carbs': 220,
        'fat': 60
    },
    'meal_preferences': {
        'breakfast': ['quick', 'high-protein'],
        'lunch': ['balanced'],
        'dinner': ['high-protein', 'low-carb']
    }
}

# Run full workflow and print results
results = system.run_full_workflow(food_items, user_preferences)

if results['success']:
    print(f"Processed {results['new_foods']} food items")
    print(f"Categorized into {results['categorized_foods']} categories")
    
    if results['meal_plan']:
        print("\nRecommended Meal Plan:")
        for day, meals in enumerate(results['meal_plan'], 1):
            print(f"\nDay {day}:")
            for i, meal in enumerate(meals):
                meal_type = ["Breakfast", "Lunch", "Dinner"][i]
                print(f"  {meal_type}: {meal['name']}")
else:
    print("Test failed with errors:", results['errors'])
"""