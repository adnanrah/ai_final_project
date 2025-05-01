****Artificial Intelligence Final Project****

# Running the Program 
There are several ways to run the meal planning system:

1. Main Test Script

The final_test.py script demonstrates the full functionality of the system. This will load the food database from available JSON files, initialize and train the food categorizer, generate a 10-day meal plan, and test the feedback loop system. This, however, does not run the scraper and instead takes data from a JSON file (uva_dining_foods_enriched.json) that has food and nutritional information. 

python final_test.py

2. Testing Individual Components

Test just the Naive Bayes Classifier: python test_nbc.py

Test just the MDP meal planner: python test_mdp.py

Test the meal variety improvements: python test_meal_variety.py

3. Data Collection

To collect new dining hall data, you must scrape UVA dining hall websites with: python dining_hall_scraper.py

Enrich scraped data with nutritional information: python find_food_info.py

# Project Structure

nbc.py: Naive Bayes Classifier for food categorization

mdp.py: Markov Decision Process for meal planning

integrator.py: System that combines NBC and MDP components

dining_hall_scraper.py: Web scraper for UVA dining halls

find_food_info.py: Enriches food data with nutritional information

final_test.py: Main test script

modify_mdp_planner.py: Enhances MDP with learning and variety features

# Configuration
User preferences can be configured in the code. Here's an example:
```python
user_preferences = {
    'dietary_restrictions': ['vegetarian'],  # Options: vegetarian, vegan, gluten-free
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
