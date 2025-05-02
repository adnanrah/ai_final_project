import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Union, Optional
import json
import logging
import random
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MealPlannerMDP')

class MealPlannerMDP:
    """
    MDP for meal planning that optimizes based on nutrition, budget, variety, and preferences.
    Handles multi-day planning, nutritional balance, variety enforcement, and user feedback.
    """
    def __init__(self, food_db: pd.DataFrame, user_preferences: Dict):
        """
        Initialize the MDP for meal planning
        """
        self.food_db = food_db
        self.user_preferences = self._normalize_preferences(user_preferences)
        self.daily_nutrition_targets = self._calculate_nutrition_targets()
        self.meal_distribution = {
            'breakfast': {'calories': 0.25, 'protein': 0.2, 'carbs': 0.3, 'fat': 0.25},
            'lunch': {'calories': 0.35, 'protein': 0.35, 'carbs': 0.35, 'fat': 0.35},
            'dinner': {'calories': 0.4, 'protein': 0.45, 'carbs': 0.35, 'fat': 0.4}
        }
        if 'meal_distribution' in self.user_preferences:
            for meal, distribution in self.user_preferences['meal_distribution'].items():
                if meal in self.meal_distribution:
                    self.meal_distribution[meal].update(distribution)
        self.states = self._define_states()
        self.actions = self._define_actions()
        self.rewards = self._define_rewards()
        self.transitions = self._define_transitions()
        self.gamma = 0.9  # Discount factor
        self.theta = 0.01  # Convergence threshold
        self.V = {s: 0 for s in self.states}
        self.policy = {}
        self.current_state = 'initial'
        self.meal_history = []
        self.daily_nutrition_consumed = {
            'calories': 0, 'protein': 0, 'fat': 0, 'carbs': 0
        }
        self.nbc_categories = []
        self.category_weights = {}
        logger.info(f"MDP initialized with {len(self.food_db)} food items and {len(self.states)} states")
    
    def _normalize_preferences(self, preferences: Dict) -> Dict:
        """
        Normalize and validate user preferences
        """
        normalized = {
            'dietary_restrictions': [],
            'budget': float('inf'),
            'nutrition_goals': {},
            'meal_preferences': {},
            'previous_meals': [],
            'disliked_foods': [],
            'favorite_foods': [],
            'height': 170,
            'weight': 70,
            'age': 30,
            'gender': 'neutral',
            'activity_level': 'moderate'
        }
        normalized.update(preferences)
        if 'budget' in preferences and preferences['budget'] is not None:
            try:
                normalized['budget'] = float(normalized['budget'])
            except (ValueError, TypeError):
                normalized['budget'] = float('inf')
        for key in ['dietary_restrictions', 'previous_meals', 'disliked_foods', 'favorite_foods']:
            if key in normalized and not isinstance(normalized[key], list):
                normalized[key] = [normalized[key]]
        return normalized
    
    def _calculate_nutrition_targets(self) -> Dict[str, float]:
        """
        Calculate daily nutrition targets based on user profile
        """
        height = self.user_preferences.get('height', 170)
        weight = self.user_preferences.get('weight', 70)
        age = self.user_preferences.get('age', 30)
        gender = self.user_preferences.get('gender', 'neutral')
        activity_level = self.user_preferences.get('activity_level', 'moderate')
        activity_multipliers = {
            'sedentary': 1.2,
            'light': 1.375,
            'moderate': 1.55,
            'active': 1.725,
            'very_active': 1.9
        }
        activity_factor = activity_multipliers.get(activity_level, 1.55)
        if gender.lower() in ['male', 'm']:
            bmr = 10 * weight + 6.25 * height - 5 * age + 5
        elif gender.lower() in ['female', 'f']:
            bmr = 10 * weight + 6.25 * height - 5 * age - 161
        else:
            bmr = 10 * weight + 6.25 * height - 5 * age - 78
        tdee = bmr * activity_factor
        protein_ratio = 0.3
        fat_ratio = 0.3
        carb_ratio = 0.4
        if 'nutrition_goals' in self.user_preferences:
            nutrition_goals = self.user_preferences['nutrition_goals']
            if 'calories' in nutrition_goals:
                tdee = nutrition_goals['calories']
            for nutrient, value in nutrition_goals.items():
                if nutrient == 'protein_ratio' and 0 <= value <= 1:
                    protein_ratio = value
                elif nutrient == 'fat_ratio' and 0 <= value <= 1:
                    fat_ratio = value
                elif nutrient == 'carb_ratio' and 0 <= value <= 1:
                    carb_ratio = value
        protein_target = (tdee * protein_ratio) / 4
        fat_target = (tdee * fat_ratio) / 9
        carb_target = (tdee * carb_ratio) / 4
        targets = {
            'calories': tdee,
            'protein': protein_target,
            'fat': fat_target,
            'carbs': carb_target
        }
        logger.info(f"Calculated nutrition targets: {targets}")
        return targets
        
    def _define_states(self) -> List[str]:
        """
        Define states representing meal history, nutritional balance, and context
        """
        if 'category' in self.food_db.columns:
            categories = []
            for cat in self.food_db['category']:
                if isinstance(cat, list):
                    categories.extend(cat)
                else:
                    categories.append(cat)
            unique_categories = list(set(categories))
            self.nbc_categories = unique_categories
        else:
            unique_categories = [
                'healthy', 'high-protein', 'low-carb', 'vegetarian', 
                'vegan', 'gluten-free', 'budget-friendly', 'keto'
            ]
            self.nbc_categories = unique_categories
        meal_types = ['breakfast', 'lunch', 'dinner']
        states = []
        for meal_type in meal_types:
            for category in unique_categories:
                states.append(f"{meal_type}_{category}")
        nutrition_states = [
            'deficit_protein',
            'excess_protein',
            'deficit_calories',
            'excess_calories',
            'balanced'
        ]
        states.extend(nutrition_states)
        states.append('initial')
        for category in unique_categories:
            states.append(f"recent_{category}")
        return states
    
    def _define_actions(self) -> List[str]:
        """
        Define actions (food items) that can be recommended
        """
        valid_foods = self.food_db.copy()
        for restriction in self.user_preferences.get('dietary_restrictions', []):
            if restriction.lower() == 'vegetarian':
                meat_ingredients = ['chicken', 'beef', 'pork', 'fish', 'meat', 'turkey']
                valid_foods = valid_foods[~valid_foods['ingredients'].apply(
                    lambda x: any(ingredient in str(x).lower() for ingredient in meat_ingredients)
                )]
            elif restriction.lower() == 'vegan':
                animal_products = [
                    'chicken', 'beef', 'pork', 'fish', 'meat', 'turkey',
                    'milk', 'cheese', 'yogurt', 'butter', 'cream', 'egg'
                ]
                valid_foods = valid_foods[~valid_foods['ingredients'].apply(
                    lambda x: any(ingredient in str(x).lower() for ingredient in animal_products)
                )]
            elif restriction.lower() == 'gluten-free':
                gluten_ingredients = ['wheat', 'barley', 'rye', 'flour', 'bread', 'pasta']
                valid_foods = valid_foods[~valid_foods['ingredients'].apply(
                    lambda x: any(ingredient in str(x).lower() for ingredient in gluten_ingredients)
                )]
        if 'budget' in self.user_preferences and self.user_preferences['budget'] < float('inf'):
            budget = self.user_preferences['budget']
            if 'price' in valid_foods.columns:
                valid_foods = valid_foods[valid_foods['price'] <= budget]
        disliked_foods = self.user_preferences.get('disliked_foods', [])
        if disliked_foods and 'food_id' in valid_foods.columns:
            valid_foods = valid_foods[~valid_foods['food_id'].isin(disliked_foods)]
        if 'food_id' in valid_foods.columns:
            actions = valid_foods['food_id'].tolist()
        else:
            actions = [str(idx) for idx in valid_foods.index]
        if not actions:
            logger.warning("No valid foods found after applying filters. Using all foods.")
            if 'food_id' in self.food_db.columns:
                actions = self.food_db['food_id'].tolist()
            else:
                actions = [str(idx) for idx in self.food_db.index]
        logger.info(f"Defined {len(actions)} possible actions (food items)")
        return actions
    
    def _define_rewards(self) -> Dict[Tuple[str, str], float]:
        """
        Define rewards based on nutritional value, budget, and variety
        """
        rewards = {}
        food_lookup = {}
        for idx, row in self.food_db.iterrows():
            food_id = row.get('food_id', str(idx))
            food_lookup[food_id] = row.to_dict()
        for state in self.states:
            for action in self.actions:
                reward = 0.0
                food = food_lookup.get(action, {})
                if not food:
                    continue
                meal_type = None
                if state.startswith(('breakfast_', 'lunch_', 'dinner_')):
                    meal_type = state.split('_')[0]
                food_category = food.get('category', [])
                if isinstance(food_category, str):
                    food_category = [food_category]
                if meal_type in ['lunch', 'dinner'] and 'breakfast' in food_category:
                    reward -= 15.0
                if 'vegetarian' in food_category or 'vegan' in food_category:
                    if 'vegetarian' in self.user_preferences.get('dietary_restrictions', []) or 'vegan' in self.user_preferences.get('dietary_restrictions', []):
                        reward += 1.0
                    else:
                        reward -= 2.0
                if meal_type and meal_type in self.meal_distribution:
                    meal_targets = {}
                    for nutrient, ratio in self.meal_distribution[meal_type].items():
                        if nutrient in self.daily_nutrition_targets:
                            meal_targets[nutrient] = self.daily_nutrition_targets[nutrient] * ratio
                    for nutrient, target in meal_targets.items():
                        if nutrient in food and not pd.isna(food[nutrient]):
                            value = float(food[nutrient])
                            if nutrient == 'calories':
                                if value > target * 1.2:
                                    reward -= 2.0 * (value / target - 1.2)
                                elif value > target * 0.8:
                                    reward += 3.0 * (1.0 - abs(value / target - 1.0))
                                else:
                                    reward -= 1.0 * (target * 0.8 - value) / target
                            elif nutrient == 'protein':
                                if value >= target * 0.8:
                                    reward += 4.0 * min(value / target, 1.2)
                                else:
                                    reward += 2.0 * (value / target)
                            else:
                                reward += 2.0 * (1.0 - min(abs(value / target - 1.0), 0.5) / 0.5)
                if meal_type == 'breakfast' and 'breakfast' in food_category:
                    reward += 5.0
                elif meal_type == 'lunch' and any(cat in ['balanced', 'high-protein'] for cat in food_category):
                    reward += 3.0
                elif meal_type == 'dinner' and any(cat in ['high-protein', 'low-carb'] for cat in food_category):
                    reward += 3.0
                if state.startswith('deficit_'):
                    nutrient = state.split('_')[1]
                    if nutrient in food and not pd.isna(food[nutrient]):
                        value = float(food[nutrient])
                        if nutrient == 'protein':
                            reward += 3.0 * min(value / 20.0, 1.5)
                        elif nutrient == 'calories':
                            reward += 2.0 * min(value / 300.0, 1.5)
                elif state.startswith('excess_'):
                    nutrient = state.split('_')[1]
                    if nutrient in food and not pd.isna(food[nutrient]):
                        value = float(food[nutrient])
                        reward -= 2.0 * min(value / 100.0, 2.0)
                elif state == 'balanced':
                    if all(nutrient in food for nutrient in ['calories', 'protein', 'carbs', 'fat']):
                        if not any(pd.isna(food[nutrient]) for nutrient in ['calories', 'protein', 'carbs', 'fat']):
                            total_cals = float(food['calories'])
                            protein_cals = float(food['protein']) * 4
                            carb_cals = float(food['carbs']) * 4
                            fat_cals = float(food['fat']) * 9
                            protein_ratio = protein_cals / total_cals if total_cals > 0 else 0
                            carb_ratio = carb_cals / total_cals if total_cals > 0 else 0
                            fat_ratio = fat_cals / total_cals if total_cals > 0 else 0
                            balance_score = (
                                (1.0 - abs(protein_ratio - 0.3)) +
                                (1.0 - abs(carb_ratio - 0.4)) +
                                (1.0 - abs(fat_ratio - 0.3))
                            ) / 3.0
                            reward += 3.0 * balance_score
                if state.startswith('recent_'):
                    recent_category = state.split('_')[1]
                    food_category = food.get('category', '')
                    if isinstance(food_category, str):
                        food_category = [food_category]
                    if recent_category in food_category:
                        reward -= 3.0
                    else:
                        reward += 1.0
                if 'food_id' in food and food['food_id'] in self.user_preferences.get('favorite_foods', []):
                    reward += 4.0
                if meal_type and 'meal_preferences' in self.user_preferences:
                    meal_prefs = self.user_preferences['meal_preferences'].get(meal_type, [])
                    food_category = food.get('category', '')
                    if isinstance(food_category, str):
                        food_category = [food_category]
                    if any(pref in food_category for pref in meal_prefs):
                        reward += 2.5
                if 'price' in food and 'budget' in self.user_preferences:
                    price = float(food['price'])
                    budget = self.user_preferences['budget']
                    if price > budget:
                        reward -= 5.0 * (price / budget)
                    else:
                        reward += 1.0 * (1.0 - max(0.0, budget - price) / budget)
                rewards[(state, action)] = reward
        return rewards
    
    def _define_transitions(self) -> Dict[Tuple[str, str], str]:
        """
        Define transitions between states based on actions
        """
        transitions = {}
        food_lookup = {}
        for idx, row in self.food_db.iterrows():
            food_id = row.get('food_id', str(idx))
            food_lookup[food_id] = row.to_dict()
        for state in self.states:
            for action in self.actions:
                food = food_lookup.get(action, {})
                if not food:
                    continue
                food_category = food.get('category', 'healthy')
                if isinstance(food_category, list) and food_category:
                    food_category = food_category[0]
                calories = float(food.get('calories', 0))
                protein = float(food.get('protein', 0))
                if state.startswith(('breakfast_', 'lunch_', 'dinner_')):
                    meal_type = state.split('_')[0]
                    next_meal_map = {
                        'breakfast': 'lunch',
                        'lunch': 'dinner',
                        'dinner': 'breakfast'
                    }
                    if protein > self.daily_nutrition_targets['protein'] * 0.3:
                        next_state = f"{next_meal_map[meal_type]}_high-protein"
                    elif calories < self.daily_nutrition_targets['calories'] * 0.2:
                        next_state = 'deficit_calories'
                    elif calories > self.daily_nutrition_targets['calories'] * 0.4:
                        next_state = 'excess_calories'
                    else:
                        next_state = 'balanced'
                elif state.startswith('deficit_'):
                    nutrient = state.split('_')[1]
                    if nutrient in food and not pd.isna(food[nutrient]):
                        value = float(food[nutrient])
                        if nutrient == 'protein' and value > 20:
                            next_state = 'balanced'
                        elif nutrient == 'calories' and value > 300:
                            next_state = 'balanced'
                        else:
                            next_state = state
                    else:
                        next_state = state
                elif state.startswith('excess_'):
                    nutrient = state.split('_')[1]
                    if nutrient in food and not pd.isna(food[nutrient]):
                        value = float(food[nutrient])
                        if nutrient == 'protein' and value < 10:
                            next_state = 'balanced'
                        elif nutrient == 'calories' and value < 200:
                            next_state = 'balanced'
                        else:
                            next_state = state
                    else:
                        next_state = state
                elif state == 'balanced':
                    next_state = f"breakfast_{food_category}"
                elif state == 'initial':
                    next_state = f"breakfast_{food_category}"
                elif state.startswith('recent_'):
                    next_state = f"breakfast_{food_category}"
                else:
                    next_state = 'balanced'
                transitions[(state, action)] = next_state
        return transitions
    
    def value_iteration(self) -> Dict[str, float]:
        """
        Run value iteration to find optimal value function
        """
        logger.info("Starting value iteration algorithm")
        V = {s: 0 for s in self.states}
        iteration = 0
        while True:
            iteration += 1
            delta = 0
            for s in self.states:
                v = V[s]
                action_values = []
                for a in self.actions:
                    if (s, a) in self.rewards and (s, a) in self.transitions:
                        reward = self.rewards[(s, a)]
                        next_s = self.transitions[(s, a)]
                        value = reward + self.gamma * V[next_s]
                        action_values.append(value)
                if action_values:
                    V[s] = max(action_values)
                    delta = max(delta, abs(v - V[s]))
            if delta < self.theta:
                logger.info(f"Value iteration converged after {iteration} iterations")
                break
            if iteration > 100:
                logger.warning("Value iteration reached max iterations without convergence")
                break
        self.V = V
        return V
    
    def extract_policy(self) -> Dict[str, str]:
        """
        Extract optimal policy from value function
        """
        logger.info("Extracting optimal policy from value function")
        policy = {}
        for s in self.states:
            best_action = None
            best_value = float('-inf')
            for a in self.actions:
                if (s, a) in self.rewards and (s, a) in self.transitions:
                    reward = self.rewards[(s, a)]
                    next_s = self.transitions[(s, a)]
                    value = reward + self.gamma * self.V[next_s]
                    if value > best_value:
                        best_value = value
                        best_action = a
            if best_action:
                policy[s] = best_action
            else:
                if self.actions:
                    policy[s] = self.actions[0]
                    logger.warning(f"No valid action found for state {s}, using default")
        self.policy = policy
        return policy
    
    def recommend_meal(self, current_state: str = None, meal_type: str = None) -> Dict:
        """
        Recommend a meal based on current state and optimal policy
        """
        if not self.policy:
            logger.warning("Policy not computed yet. Running value iteration and extracting policy.")
            self.value_iteration()
            self.extract_policy()
        state = current_state or self.current_state
        if meal_type:
            meal_states = [s for s in self.states if s.startswith(f"{meal_type}_")]
            if meal_states:
                state = meal_states[0]
        if state not in self.policy:
            logger.warning(f"State {state} not in policy. Using initial state.")
            state = 'initial'
        if state not in self.policy:
            logger.error("Initial state not in policy. Using first action.")
            if self.actions:
                action = self.actions[0]
            else:
                logger.error("No actions available.")
                return None
        else:
            action = self.policy[state]
        if meal_type:
            food_item = None
            for idx, row in self.food_db.iterrows():
                food_id = row.get('food_id', str(idx))
                if food_id == action:
                    food_item = row.to_dict()
                    break
            if food_item:
                categories = food_item.get('category', [])
                if isinstance(categories, str):
                    categories = [categories]
                if meal_type in ['lunch', 'dinner'] and 'breakfast' in categories:
                    for _ in range(5):
                        for alt_state, alt_action in self.policy.items():
                            if alt_state.startswith(f"{meal_type}_"):
                                alt_food = None
                                for idx, row in self.food_db.iterrows():
                                    alt_food_id = row.get('food_id', str(idx))
                                    if alt_food_id == alt_action:
                                        alt_food = row.to_dict()
                                        break
                                if alt_food:
                                    alt_categories = alt_food.get('category', [])
                                    if isinstance(alt_categories, str):
                                        alt_categories = [alt_categories]
                                    if 'breakfast' not in alt_categories:
                                        action = alt_action
                                        break
        for idx, row in self.food_db.iterrows():
            food_id = row.get('food_id', str(idx))
            if food_id == action:
                if (state, action) in self.transitions:
                    self.current_state = self.transitions[(state, action)]
                self.meal_history.append({
                    'food_id': food_id,
                    'state': state,
                    'meal_type': meal_type or state.split('_')[0] if state != 'initial' else 'breakfast',
                    'timestamp': datetime.now()
                })
                return row.to_dict()
        logger.error(f"Food item for action {action} not found in database.")
        return None
    
    def plan_meals(self, days: int = 1, meal_types: List[str] = None) -> List[List[Dict]]:
        """
        Plan meals for multiple days
        """
        if not meal_types:
            meal_types = ['breakfast', 'lunch', 'dinner']
        if not self.policy:
            logger.info("Computing policy for meal planning")
            self.value_iteration()
            self.extract_policy()
        self.current_state = 'initial'
        self.daily_nutrition_consumed = {
            'calories': 0, 'protein': 0, 'fat': 0, 'carbs': 0
        }
        recent_meals = set()
        meal_type_history = {'breakfast': [], 'lunch': [], 'dinner': []}
        category_counts = defaultdict(int)
        meal_plan = []
        for day in range(days):
            daily_meals = []
            self.daily_nutrition_consumed = {
                'calories': 0, 'protein': 0, 'fat': 0, 'carbs': 0
            }
            for meal_type in meal_types:
                max_attempts = 8
                best_meal = None
                best_score = float('-inf')
                for attempt in range(max_attempts):
                    meal = self.recommend_meal(meal_type=meal_type)
                    if not meal:
                        break
                    meal_id = meal.get('food_id', str(meal.get('name', '')))
                    variety_score = 0
                    if meal_id in recent_meals:
                        variety_score -= 10
                    if meal_id in meal_type_history[meal_type]:
                        recency_index = meal_type_history[meal_type].index(meal_id)
                        recency_penalty = len(meal_type_history[meal_type]) - recency_index
                        variety_score -= recency_penalty
                    categories = meal.get('category', [])
                    if isinstance(categories, str):
                        categories = [categories]
                    for category in categories:
                        if category_counts[category] > 0:
                            variety_score -= category_counts[category]
                    if meal_type in ['lunch', 'dinner'] and 'breakfast' in categories:
                        variety_score -= 15
                    if meal_type == 'breakfast' and 'breakfast' not in categories:
                        has_breakfast_foods = False
                        for _, row in self.food_db.iterrows():
                            row_categories = row.get('category', [])
                            if isinstance(row_categories, str):
                                row_categories = [row_categories]
                            if 'breakfast' in row_categories:
                                has_breakfast_foods = True
                                break
                        if not has_breakfast_foods:
                            variety_score += 5
                    if variety_score > best_score or best_meal is None:
                        best_meal = meal
                        best_score = variety_score
                if best_meal:
                    meal_id = best_meal.get('food_id', str(best_meal.get('name', '')))
                    daily_meals.append(best_meal)
                    recent_meals.add(meal_id)
                    if len(meal_type_history[meal_type]) >= 10:
                        meal_type_history[meal_type].pop(0)
                    meal_type_history[meal_type].append(meal_id)
                    categories = best_meal.get('category', [])
                    if isinstance(categories, str):
                        categories = [categories]
                    for category in categories:
                        category_counts[category] += 1
                    for nutrient in ['calories', 'protein', 'fat', 'carbs']:
                        if nutrient in best_meal and not pd.isna(best_meal[nutrient]):
                            self.daily_nutrition_consumed[nutrient] += float(best_meal[nutrient])
                    self._update_state_based_on_nutrition()
                else:
                    daily_meals.append({'name': f"No suitable {meal_type} found", 'calories': 0, 'protein': 0})
            meal_plan.append(daily_meals)
        return meal_plan
    
    def _update_state_based_on_nutrition(self):
        """Update current state based on nutrition consumption"""
        protein_consumed = self.daily_nutrition_consumed['protein']
        protein_target = self.daily_nutrition_targets['protein']
        if protein_consumed < protein_target * 0.5:
            self.current_state = 'deficit_protein'
        elif protein_consumed > protein_target * 1.3:
            self.current_state = 'excess_protein'
        calories_consumed = self.daily_nutrition_consumed['calories']
        calories_target = self.daily_nutrition_targets['calories']
        if calories_consumed < calories_target * 0.6:
            self.current_state = 'deficit_calories'
        elif calories_consumed > calories_target * 1.2:
            self.current_state = 'excess_calories'
        if protein_consumed >= protein_target * 0.5 and protein_consumed <= protein_target * 1.3 and \
           calories_consumed >= calories_target * 0.6 and calories_consumed <= calories_target * 1.2:
            self.current_state = 'balanced'
    
    def update_from_feedback(self, meal_id: str, rating: int, food_id: str = None):
        """
        Update model based on user feedback (1-5 rating)
        """
        meal_entry = None
        for entry in self.meal_history:
            if entry.get('food_id') == meal_id or entry.get('food_id') == food_id:
                meal_entry = entry
                break
        if not meal_entry:
            logger.warning(f"Meal {meal_id} not found in history, feedback ignored")
            return
        state = meal_entry['state']
        action = meal_entry['food_id']
        if (state, action) not in self.rewards:
            logger.warning(f"State-action pair ({state}, {action}) not in rewards, feedback ignored")
            return
        feedback_reward = (rating - 3) * 2.5
        current_reward = self.rewards[(state, action)]
        self.rewards[(state, action)] = 0.7 * current_reward + 0.3 * feedback_reward
        logger.info(f"Updated reward for ({state}, {action}) based on feedback: {current_reward} -> {self.rewards[(state, action)]}")
        if abs(current_reward - self.rewards[(state, action)]) > 1.0:
            logger.info("Significant reward change, recomputing policy")
            self.value_iteration()
            self.extract_policy()
        if rating >= 4:
            if food_id and food_id not in self.user_preferences.get('favorite_foods', []):
                if 'favorite_foods' not in self.user_preferences:
                    self.user_preferences['favorite_foods'] = []
                self.user_preferences['favorite_foods'].append(food_id)
        if rating <= 2:
            if food_id and food_id not in self.user_preferences.get('disliked_foods', []):
                if 'disliked_foods' not in self.user_preferences:
                    self.user_preferences['disliked_foods'] = []
                self.user_preferences['disliked_foods'].append(food_id)
    
    def integrate_nbc_categories(self, nbc_categories: List[str], category_weights: Dict[str, float] = None):
        """
        Integrate NBC classifications into the MDP
        """
        self.nbc_categories = nbc_categories
        if not category_weights:
            category_weights = {cat: 1.0 for cat in nbc_categories}
        self.category_weights = category_weights
        new_states = []
        for cat in nbc_categories:
            for meal_type in ['breakfast', 'lunch', 'dinner']:
                new_states.append(f"{meal_type}_{cat}")
            new_states.append(f"recent_{cat}")
        for state in new_states:
            if state not in self.states:
                self.states.append(state)
        logger.info(f"Integrated {len(nbc_categories)} NBC categories into MDP")
        logger.info(f"MDP now has {len(self.states)} states")
        self.rewards = self._define_rewards()
        self.transitions = self._define_transitions()
        self.value_iteration()
        self.extract_policy()
    
    def modify_food_category(self, food_id: str, new_category: str):
        """
        Modify category of a food item based on NBC classification
        """
        for idx, row in self.food_db.iterrows():
            if row.get('food_id', str(idx)) == food_id:
                self.food_db.at[idx, 'category'] = new_category
                logger.info(f"Updated category for food {food_id} to {new_category}")
                for state in self.states:
                    if (state, food_id) in self.rewards:
                        food = self.food_db.loc[idx].to_dict()
                        self._recompute_reward(state, food_id, food)
                    if (state, food_id) in self.transitions:
                        self._recompute_transition(state, food_id, new_category)
                return True
        logger.warning(f"Food {food_id} not found in database, category not modified")
        return False
    
    def _recompute_reward(self, state: str, action: str, food: Dict):
        """
        Recompute reward for a state-action pair
        """
        reward = 0.0
        meal_type = None
        if state.startswith(('breakfast_', 'lunch_', 'dinner_')):
            meal_type = state.split('_')[0]
        if meal_type and meal_type in self.meal_distribution:
            meal_targets = {}
            for nutrient, ratio in self.meal_distribution[meal_type].items():
                if nutrient in self.daily_nutrition_targets:
                    meal_targets[nutrient] = self.daily_nutrition_targets[nutrient] * ratio
            for nutrient, target in meal_targets.items():
                if nutrient in food and not pd.isna(food[nutrient]):
                    value = float(food[nutrient])
                    if nutrient == 'calories':
                        if value > target * 1.2:
                            reward -= 2.0 * (value / target - 1.2)
                        elif value > target * 0.8:
                            reward += 3.0 * (1.0 - abs(value / target - 1.0))
                        else:
                            reward -= 1.0 * (target * 0.8 - value) / target
                    elif nutrient == 'protein':
                        if value >= target * 0.8:
                            reward += 4.0 * min(value / target, 1.2)
                        else:
                            reward += 2.0 * (value / target)
                    else:
                        reward += 2.0 * (1.0 - min(abs(value / target - 1.0), 0.5) / 0.5)
        if (state, action) in self.rewards:
            self.rewards[(state, action)] = reward
    
    def _recompute_transition(self, state: str, action: str, category: str):
        """
        Recompute transition for a state-action pair
        """
        if state.startswith(('breakfast_', 'lunch_', 'dinner_')):
            meal_type = state.split('_')[0]
            next_meal_map = {
                'breakfast': 'lunch',
                'lunch': 'dinner',
                'dinner': 'breakfast'
            }
            next_state = f"{next_meal_map[meal_type]}_{category}"
        elif state == 'initial' or state == 'balanced':
            next_state = f"breakfast_{category}"
        elif state.startswith('recent_'):
            next_state = f"breakfast_{category}"
        else:
            next_state = 'balanced'
        if (state, action) in self.transitions:
            self.transitions[(state, action)] = next_state

# Example usage function
def demo_meal_planner(food_db: pd.DataFrame = None, user_preferences: Dict = None):
    """
    Demo of the meal planner with sample data
    """
    if food_db is None:
        food_data = []
        food_data.append({
            'food_id': 'B001',
            'name': 'Oatmeal with Berries',
            'price': 4.50,
            'calories': 320,
            'protein': 12,
            'fat': 6,
            'carbs': 54,
            'category': ['healthy', 'balanced'],
            'ingredients': ['oats', 'milk', 'berries', 'honey'],
            'meal_type': 'breakfast'
        })
        food_data.append({
            'food_id': 'B002',
            'name': 'Protein Pancakes',
            'price': 6.75,
            'calories': 450,
            'protein': 25,
            'fat': 12,
            'carbs': 45,
            'category': ['high-protein', 'balanced'],
            'ingredients': ['flour', 'protein powder', 'eggs', 'milk'],
            'meal_type': 'breakfast'
        })
        food_data.append({
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
        })
        food_data.append({
            'food_id': 'L002',
            'name': 'Vegetarian Wrap',
            'price': 7.25,
            'calories': 420,
            'protein': 18,
            'fat': 14,
            'carbs': 52,
            'category': ['vegetarian', 'balanced'],
            'ingredients': ['tortilla', 'hummus', 'lettuce', 'tomato', 'cucumber', 'avocado'],
            'meal_type': 'lunch'
        })
        food_data.append({
            'food_id': 'D001',
            'name': 'Salmon with Roasted Vegetables',
            'price': 12.50,
            'calories': 480,
            'protein': 38,
            'fat': 22,
            'carbs': 25,
            'category': ['high-protein', 'low-carb'],
            'ingredients': ['salmon', 'broccoli', 'carrots', 'olive oil', 'garlic'],
            'meal_type': 'dinner'
        })
        food_data.append({
            'food_id': 'D002',
            'name': 'Spaghetti with Meatballs',
            'price': 9.75,
            'calories': 720,
            'protein': 35,
            'fat': 28,
            'carbs': 75,
            'category': ['high-protein', 'balanced'],
            'ingredients': ['pasta', 'beef', 'tomato sauce', 'garlic', 'onion'],
            'meal_type': 'dinner'
        })
        food_db = pd.DataFrame(food_data)
    if user_preferences is None:
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
            },
            'height': 175,
            'weight': 75,
            'age': 30,
            'gender': 'neutral',
            'activity_level': 'moderate'
        }
    planner = MealPlannerMDP(food_db, user_preferences)
    planner.value_iteration()
    planner.extract_policy()
    meal_plan = planner.plan_meals(days=3)
    print("\n3-Day Meal Plan:")
    for day, meals in enumerate(meal_plan, 1):
        print(f"\nDay {day}:")
        for i, meal in enumerate(meals):
            meal_type = ["Breakfast", "Lunch", "Dinner"][i]
            print(f"  {meal_type}: {meal['name']} (${meal['price']:.2f})")
            print(f"    Category: {meal['category']}")
            print(f"    Nutrition: {meal['calories']} cal, {meal['protein']}g protein, {meal['carbs']}g carbs, {meal['fat']}g fat")
    return planner