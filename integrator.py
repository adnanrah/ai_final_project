import pandas as pd
import numpy as np
import json
import logging
import os
import requests
from typing import Dict, List, Union, Any, Optional
from datetime import datetime
from nbc import EnhancedFoodCategorizer
from mdp import MealPlannerMDP
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MealRecommendationSystem')
class MealRecommendationSystem:
    """Integrated system combining LLM nutrition extraction, NBC, and MDP for meal planning"""
    def __init__(self, llm_api_key: str = None, food_db_path: str = None):
        """Initialize the meal recommendation system"""
        self.llm_api_key = llm_api_key
        self.food_db = None
        if food_db_path and os.path.exists(food_db_path):
            self.food_db = pd.read_csv(food_db_path)
            logger.info(f"Loaded food database with {len(self.food_db)} items")
        else:
            self.food_db = pd.DataFrame(columns=[
                'food_id', 'name', 'description', 'ingredients', 
                'calories', 'protein', 'fat', 'carbs', 'category',
                'price', 'location'
            ])
            logger.info("Initialized empty food database")
        self.food_categorizer = None
        self.meal_planner = None
        self.user_preferences = {}
        self.user_history = []
        logger.info("Meal recommendation system initialized")

    def initialize_categorizer(self, multi_label: bool = True):
        """Initialize the Naive Bayes food categorizer"""
        try:
            self.food_categorizer = EnhancedFoodCategorizer(multi_label=multi_label)
            if len(self.food_db) >= 10:
                logger.info("Training food categorizer on existing data")
                self.food_categorizer.train(self.food_db)
            else:
                logger.info("Not enough data to train categorizer yet")
            return True
        except Exception as e:
            logger.error(f"Error initializing food categorizer: {e}")
            return False

    def initialize_meal_planner(self, user_preferences: Dict = None):
        """Initialize the MDP meal planner"""
        try:
            if user_preferences:
                self.user_preferences = user_preferences
            elif not self.user_preferences:
                self.user_preferences = {
                    'dietary_restrictions': [],
                    'budget': 15.0,
                    'nutrition_goals': {
                        'calories': 2000,
                        'protein': 120,
                        'carbs': 250,
                        'fat': 65
                    }
                }
            self.meal_planner = MealPlannerMDP(self.food_db, self.user_preferences)
            logger.info("Computing optimal meal planning policy")
            self.meal_planner.value_iteration()
            self.meal_planner.extract_policy()
            return True
        except Exception as e:
            logger.error(f"Error initializing meal planner: {e}")
            return False

    def query_llm_for_nutrition(self, food_items: List[str]) -> List[Dict]:
        """Query LLM to extract nutritional information from food descriptions"""
        if not self.llm_api_key:
            logger.warning("No LLM API key provided, using mock data")
            return self._mock_nutrition_extraction(food_items)
        try:
            results = []
            for item in food_items:
                prompt = f"""
                Extract nutritional information from the following food description:
                {item}
                
                Return ONLY a JSON object with the following structure:
                {{
                    "name": "Food name",
                    "description": "Brief description",
                    "ingredients": ["ingredient1", "ingredient2", ...],
                    "nutrition_info": {{
                        "calories": number,
                        "protein": number (grams),
                        "fat": number (grams),
                        "carbs": number (grams)
                    }}
                }}
                """
                response = self._call_llm_api(prompt)
                try:
                    nutrition_data = json.loads(response)
                    results.append(nutrition_data)
                    logger.info(f"Successfully extracted nutrition for: {nutrition_data.get('name', item[:20])}")
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse LLM response as JSON: {response[:100]}...")
                    results.append({
                        "name": item[:50],
                        "description": item,
                        "ingredients": [],
                        "nutrition_info": {
                            "calories": None,
                            "protein": None,
                            "fat": None,
                            "carbs": None
                        }
                    })
            return results
        except Exception as e:
            logger.error(f"Error querying LLM for nutrition: {e}")
            return self._mock_nutrition_extraction(food_items)

    def _call_llm_api(self, prompt: str) -> str:
        """Call LLM API with a prompt"""
        try:
            headers = {
                "Authorization": f"Bearer {self.llm_api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "prompt": prompt,
                "max_tokens": 500,
                "temperature": 0.2
            }
            response = requests.post(
                "https://api.anthropic.com/v1/complete",
                headers=headers,
                json=data,
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                return result.get("completion", "")
            else:
                logger.error(f"API error: {response.status_code}, {response.text}")
                return ""
        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            return ""

    def _mock_nutrition_extraction(self, food_items: List[str]) -> List[Dict]:
        """Generate mock nutritional data for testing"""
        results = []
        for item in food_items:
            name = item.split(',')[0] if ',' in item else item[:50]
            calories = np.random.randint(200, 800)
            protein = np.random.randint(5, 40)
            fat = np.random.randint(5, 30)
            carbs = np.random.randint(10, 80)
            potential_ingredients = [
                'chicken', 'beef', 'pork', 'fish', 'tofu', 'tempeh',
                'rice', 'pasta', 'bread', 'potato', 'quinoa',
                'lettuce', 'spinach', 'kale', 'arugula',
                'tomato', 'onion', 'garlic', 'bell pepper',
                'cheese', 'yogurt', 'milk', 'cream',
                'olive oil', 'butter', 'avocado'
            ]
            ingredients = []
            for ingredient in potential_ingredients:
                if ingredient in item.lower():
                    ingredients.append(ingredient)
            while len(ingredients) < 2:
                random_ingredient = np.random.choice(potential_ingredients)
                if random_ingredient not in ingredients:
                    ingredients.append(random_ingredient)
            results.append({
                "name": name,
                "description": item,
                "ingredients": ingredients,
                "nutrition_info": {
                    "calories": calories,
                    "protein": protein,
                    "fat": fat,
                    "carbs": carbs
                }
            })
        return results

    def add_foods_to_database(self, food_data: List[Dict]) -> bool:
        """Add new foods to the database"""
        try:
            new_foods_df = pd.DataFrame(food_data)
            if 'nutrition_info' in new_foods_df.columns:
                for nutrient in ['calories', 'protein', 'fat', 'carbs']:
                    if nutrient not in new_foods_df.columns:
                        new_foods_df[nutrient] = new_foods_df['nutrition_info'].apply(
                            lambda x: x.get(nutrient) if isinstance(x, dict) else None
                        )
            if 'food_id' not in new_foods_df.columns:
                max_id = 0
                if 'food_id' in self.food_db.columns and not self.food_db.empty:
                    existing_ids = self.food_db['food_id'].str.extract(r'(\d+)').astype(float)
                    if not existing_ids.empty:
                        max_id = int(existing_ids.max().iloc[0])
                new_foods_df['food_id'] = [
                    f"F{i + max_id + 1:03d}" for i in range(len(new_foods_df))
                ]
            self.food_db = pd.concat([self.food_db, new_foods_df], ignore_index=True)
            logger.info(f"Added {len(new_foods_df)} new food items to database")
            return True
        except Exception as e:
            logger.error(f"Error adding foods to database: {e}")
            return False

    def categorize_foods(self, food_items: Union[List[str], List[Dict], pd.DataFrame] = None) -> List[Dict]:
        """Categorize foods using Naive Bayes classifier"""
        if not self.food_categorizer:
            self.initialize_categorizer()
        try:
            if food_items is None:
                items_to_categorize = self.food_db
            elif isinstance(food_items, pd.DataFrame):
                items_to_categorize = food_items
            elif isinstance(food_items, list):
                if food_items and isinstance(food_items[0], dict):
                    items_to_categorize = pd.DataFrame(food_items)
                else:
                    items_to_categorize = pd.DataFrame({'full_description': food_items})
            else:
                logger.error(f"Unsupported type for categorization: {type(food_items)}")
                return []
            if not hasattr(self.food_categorizer, 'trained') or not self.food_categorizer.trained:
                if len(self.food_db) >= 5:
                    logger.info("Training food categorizer on current database")
                    self.food_categorizer.train(self.food_db)
                else:
                    logger.warning("Not enough data to train categorizer, using rule-based categorization")
            logger.info(f"Categorizing {len(items_to_categorize)} food items")
            categories = self.food_categorizer.predict(items_to_categorize)
            results = []
            for i, (_, item) in enumerate(items_to_categorize.iterrows()):
                item_dict = item.to_dict()
                if i < len(categories):
                    item_dict['predicted_category'] = categories[i]
                else:
                    item_dict['predicted_category'] = ['balanced']
                if 'category' not in item_dict or not item_dict['category']:
                    item_dict['category'] = item_dict['predicted_category']
                results.append(item_dict)
            return results
        except Exception as e:
            logger.error(f"Error categorizing foods: {e}")
            results = []
            for _, item in items_to_categorize.iterrows():
                item_dict = item.to_dict()
                name = item_dict.get('name', '')
                if isinstance(name, str):
                    name = name.lower()
                    if any(term in name for term in ['egg', 'bacon', 'sausage', 'pancake']):
                        item_dict['predicted_category'] = ['breakfast']
                    elif any(term in name for term in ['salad', 'vegetable']):
                        item_dict['predicted_category'] = ['healthy']
                    elif any(term in name for term in ['chicken', 'beef', 'protein']):
                        item_dict['predicted_category'] = ['high-protein']
                    else:
                        item_dict['predicted_category'] = ['balanced']
                    if 'category' not in item_dict or not item_dict['category']:
                        item_dict['category'] = item_dict['predicted_category']
                results.append(item_dict)
            return results

    def update_food_categories(self, categorized_foods: List[Dict]) -> bool:
        """Update food categories in the database based on NBC predictions"""
        try:
            count = 0
            for food in categorized_foods:
                if 'food_id' in food:
                    food_id = food['food_id']
                    new_category = food.get('predicted_category')
                    if not new_category:
                        continue
                    mask = self.food_db['food_id'] == food_id
                    if mask.any():
                        self.food_db.loc[mask, 'category'] = new_category
                        count += 1
                        if self.meal_planner and hasattr(self.meal_planner, 'modify_food_category'):
                            category_str = new_category[0] if isinstance(new_category, list) and new_category else 'balanced'
                            self.meal_planner.modify_food_category(food_id, category_str)
            logger.info(f"Updated categories for {count} foods")
            if count > 5 and self.food_categorizer:
                logger.info("Retraining classifier with updated categories")
                self.food_categorizer.train(self.food_db)
            return True
        except Exception as e:
            logger.error(f"Error updating food categories: {e}")
            return False

    def recommend_meals(self, days: int = 1, meal_types: List[str] = None, 
                        user_preferences: Dict = None) -> List[List[Dict]]:
        """Generate meal recommendations using the MDP planner"""
        if not self.meal_planner:
            self.initialize_meal_planner(user_preferences)
        elif user_preferences:
            self.user_preferences.update(user_preferences)
            self.meal_planner.user_preferences = self.user_preferences
        if not meal_types:
            meal_types = ['breakfast', 'lunch', 'dinner']
        try:
            logger.info(f"Generating {days}-day meal plan for {', '.join(meal_types)}")
            meal_plan = self.meal_planner.plan_meals(days=days, meal_types=meal_types)
            timestamp = datetime.now()
            for day_index, daily_meals in enumerate(meal_plan):
                for meal_index, meal in enumerate(daily_meals):
                    meal_type = meal_types[meal_index % len(meal_types)]
                    self.user_history.append({
                        'food_id': meal.get('food_id', ''),
                        'name': meal.get('name', ''),
                        'meal_type': meal_type,
                        'day': day_index + 1,
                        'timestamp': timestamp,
                        'recommended': True,
                        'consumed': False
                    })
            return meal_plan
        except Exception as e:
            logger.error(f"Error generating meal recommendations: {e}")
            return []

    def process_user_feedback(self, food_id: str, rating: int, consumed: bool = True) -> bool:
        """Process user feedback on recommendations"""
        try:
            for entry in self.user_history:
                if entry.get('food_id') == food_id and entry.get('recommended', False):
                    entry['rating'] = rating
                    entry['consumed'] = consumed
                    entry['feedback_timestamp'] = datetime.now()
                    if self.meal_planner:
                        self.meal_planner.update_from_feedback(food_id, rating)
                    logger.info(f"Processed feedback for food {food_id}: rating={rating}, consumed={consumed}")
                    return True
            logger.warning(f"Food {food_id} not found in recommendation history")
            return False
        except Exception as e:
            logger.error(f"Error processing user feedback: {e}")
            return False

    def run_full_workflow(self, food_items: List[str], user_preferences: Dict = None) -> Dict[str, Any]:
        """Run the complete workflow from food extraction to meal recommendation"""
        results = {
            'success': True,
            'errors': [],
            'new_foods': 0,
            'categorized_foods': 0,
            'meal_plan': None
        }
        try:
            logger.info(f"Step 1: Extracting nutrition from {len(food_items)} food items")
            nutrition_data = self.query_llm_for_nutrition(food_items)
            if not nutrition_data:
                results['errors'].append("Failed to extract nutrition information")
                results['success'] = False
                return results
            logger.info("Step 2: Adding foods to database")
            if not self.add_foods_to_database(nutrition_data):
                results['errors'].append("Failed to add foods to database")
                results['success'] = False
                return results
            results['new_foods'] = len(nutrition_data)
            logger.info("Step 3: Initializing food categorizer")
            if not self.food_categorizer:
                if not self.initialize_categorizer():
                    results['errors'].append("Failed to initialize food categorizer")
                    results['success'] = False
                    return results
            logger.info("Step 4: Categorizing new foods")
            categorized_foods = self.categorize_foods(nutrition_data)
            if not categorized_foods:
                results['errors'].append("Failed to categorize foods")
                results['success'] = False
                return results
            results['categorized_foods'] = len(categorized_foods)
            logger.info("Step 5: Updating food categories in database")
            if not self.update_food_categories(categorized_foods):
                results['errors'].append("Failed to update food categories")
            logger.info("Step 6: Initializing meal planner")
            if not self.meal_planner:
                if not self.initialize_meal_planner(user_preferences):
                    results['errors'].append("Failed to initialize meal planner")
                    results['success'] = False
                    return results
            elif user_preferences:
                self.user_preferences.update(user_preferences)
                self.meal_planner.user_preferences = self.user_preferences
            logger.info("Step 7: Integrating NBC categories with MDP")
            if hasattr(self.food_categorizer, 'categories'):
                nbc_categories = self.food_categorizer.categories
                self.meal_planner.integrate_nbc_categories(nbc_categories)
            logger.info("Step 8: Generating meal recommendations")
            meal_plan = self.recommend_meals(days=3)
            if not meal_plan:
                results['errors'].append("Failed to generate meal recommendations")
                results['success'] = False
                return results
            results['meal_plan'] = meal_plan
            logger.info("Full workflow completed successfully")
            return results
        except Exception as e:
            logger.error(f"Error in workflow: {e}")
            results['errors'].append(f"Workflow error: {str(e)}")
            results['success'] = False
            return results

    def save_database(self, filepath: str) -> bool:
        """Save the food database to a CSV file"""
        try:
            self.food_db.to_csv(filepath, index=False)
            logger.info(f"Saved food database with {len(self.food_db)} items to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving database: {e}")
            return False

    def load_database(self, filepath: str) -> bool:
        """Load the food database from a CSV file"""
        try:
            if os.path.exists(filepath):
                self.food_db = pd.read_csv(filepath)
                logger.info(f"Loaded food database with {len(self.food_db)} items from {filepath}")
                if self.food_categorizer and hasattr(self.food_categorizer, 'trained') and self.food_categorizer.trained:
                    logger.info("Retraining categorizer with loaded database")
                    self.food_categorizer.train(self.food_db)
                if self.meal_planner:
                    logger.info("Updating meal planner with loaded database")
                    self.meal_planner.food_db = self.food_db
                    self.meal_planner.actions = self.meal_planner._define_actions()
                    self.meal_planner.rewards = self.meal_planner._define_rewards()
                    self.meal_planner.transitions = self.meal_planner._define_transitions()
                    self.meal_planner.value_iteration()
                    self.meal_planner.extract_policy()
                return True
            else:
                logger.error(f"Database file not found: {filepath}")
                return False
        except Exception as e:
            logger.error(f"Error loading database: {e}")
            return False

def demo_meal_recommendation_system():
    """Demonstrate the meal recommendation system"""
    sample_foods = [
        "Grilled salmon with steamed broccoli and brown rice, high in omega-3 and protein",
        "Quinoa bowl with roasted vegetables, chickpeas, and tahini dressing, vegan meal",
        "Greek yogurt with berries, honey, and granola, quick breakfast option",
        "Chicken Caesar salad with romaine lettuce, parmesan cheese, and croutons",
        "Vegetable stir-fry with tofu, bell peppers, snap peas, and brown rice",
        "Beef and vegetable soup with whole grain bread, comforting dinner option",
        "Avocado toast with poached eggs and cherry tomatoes on whole grain bread",
        "Spinach and feta omelette with a side of fruit, high protein breakfast"
    ]
    user_preferences = {
        'dietary_restrictions': [],
        'budget': 12.0,
        'nutrition_goals': {
            'calories': 2200,
            'protein': 130,
            'carbs': 240,
            'fat': 70
        },
        'meal_preferences': {
            'breakfast': ['quick', 'high-protein'],
            'lunch': ['balanced', 'vegetarian'],
            'dinner': ['high-protein', 'low-carb']
        },
        'height': 175,
        'weight': 75,
        'age': 30,
        'gender': 'neutral',
        'activity_level': 'active'
    }
    system = MealRecommendationSystem()
    results = system.run_full_workflow(sample_foods, user_preferences)
    print("\nMeal Recommendation System Demo")
    print("===============================")
    if results['success']:
        print(f"Successfully processed {results['new_foods']} new food items")
        print(f"Categorized {results['categorized_foods']} foods")
        if results['meal_plan']:
            print("\n3-Day Meal Plan:")
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
                    print(f"    Nutrition: {meal.get('calories', 'N/A')} cal, {meal.get('protein', 'N/A')}g protein")
    else:
        print("Workflow had errors:")
        for error in results['errors']:
            print(f"- {error}")
    return system

if __name__ == "__main__":
    demo_meal_recommendation_system()