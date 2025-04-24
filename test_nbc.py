from nbc import EnhancedFoodCategorizer
import pandas as pd

# Create sample data
food_data = [
    {
        'name': 'Grilled Chicken Salad',
        'description': 'Fresh salad with grilled chicken breast',
        'ingredients': ['chicken', 'lettuce', 'tomato', 'cucumber'],
        'calories': 350,
        'protein': 30,
        'fat': 12,
        'carbs': 18
    },
    {
        'name': 'Chocolate Chip Pancakes',
        'description': 'Fluffy pancakes with chocolate chips',
        'ingredients': ['flour', 'eggs', 'milk', 'chocolate chips', 'butter'],
        'calories': 650,
        'protein': 15,
        'fat': 25,
        'carbs': 85
    },
    {
        'name': 'Tofu Vegetable Stir-Fry',
        'description': 'Stir-fried tofu with mixed vegetables',
        'ingredients': ['tofu', 'broccoli', 'carrots', 'bell pepper', 'soy sauce'],
        'calories': 320,
        'protein': 18,
        'fat': 14,
        'carbs': 28
    }
]

# Convert to DataFrame
df = pd.DataFrame(food_data)

# Initialize and train categorizer
categorizer = EnhancedFoodCategorizer(multi_label=True)
categorizer.train(df)

# Test predictions
new_foods = [
    "Greek yogurt with berries and honey, high in protein and calcium",
    "Cheeseburger with french fries and soda, classic fast food meal"
]

predictions = categorizer.predict(new_foods)

# Print results
print("\nNBC Test Results:")
for food, cats in zip(new_foods, predictions):
    print(f"\nFood: {food}")
    print(f"Predicted categories: {', '.join(cats)}")