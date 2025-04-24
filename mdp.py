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
    Enhanced Markov Decision Process for meal planning that optimizes meal choices
    based on nutrition, budget, variety, and user preferences.
    
    This implementation includes:
    - Multi-day planning for all meals (breakfast, lunch, dinner)
    - Nutritional balance across meals and days
    - Variety enforcement to avoid repetition
    - Learning from user feedback
    - Integration with Naive Bayes food categorization
    """
    
    def __init__(self, food_db: pd.DataFrame, user_preferences: Dict):
        """
        Initialize the MDP for meal planning
        
        Args:
            food_db (pd.DataFrame): Food database with nutritional information
            user_preferences (dict): User preferences including:
                - dietary_restrictions (list): e.g., ['vegetarian', 'gluten-free']
                - budget (float): Maximum budget per meal
                - nutrition_goals (dict): e.g., {'protein': 25, 'calories': 500}
                - meal_preferences (dict): e.g., {'breakfast': ['quick', 'high-protein']}
                - previous_meals (list): List of previously eaten meal IDs
                - disliked_foods (list): List of disliked food IDs
                - favorite_foods (list): List of favorite food IDs
                - height (float): User's height in cm
                - weight (float): User's weight in kg
                - age (int): User's age
                - gender (str): User's gender
                - activity_level (str): User's activity level
        """
        self.food_db = food_db
        self.user_preferences = self._normalize_preferences(user_preferences)
        
        # Calculate daily nutrition targets based on user profile
        self.daily_nutrition_targets = self._calculate_nutrition_targets()
        
        # Meal type distribution (defaults)
        self.meal_distribution = {
            'breakfast': {'calories': 0.25, 'protein': 0.2, 'carbs': 0.3, 'fat': 0.25},
            'lunch': {'calories': 0.35, 'protein': 0.35, 'carbs': 0.35, 'fat': 0.35},
            'dinner': {'calories': 0.4, 'protein': 0.45, 'carbs': 0.35, 'fat': 0.4}
        }
        
        # Adjust based on user preferences if provided
        if 'meal_distribution' in self.user_preferences:
            for meal, distribution in self.user_preferences['meal_distribution'].items():
                if meal in self.meal_distribution:
                    self.meal_distribution[meal].update(distribution)
        
        # Define MDP components
        self.states = self._define_states()
        self.actions = self._define_actions()
        self.rewards = self._define_rewards()
        self.transitions = self._define_transitions()
        
        # Parameters for value iteration
        self.gamma = 0.9  # Discount factor for future rewards
        self.theta = 0.01  # Convergence threshold
        
        # Policy and values
        self.V = {s: 0 for s in self.states}
        self.policy = {}
        
        # Tracking for planning
        self.current_state = 'initial'
        self.meal_history = []
        self.daily_nutrition_consumed = {
            'calories': 0, 'protein': 0, 'fat': 0, 'carbs': 0
        }
        
        # Add NBC integration attributes
        self.nbc_categories = []
        self.category_weights = {}
        
        logger.info(f"MDP initialized with {len(self.food_db)} food items and {len(self.states)} states")
    
    def _normalize_preferences(self, preferences: Dict) -> Dict:
        """
        Normalize and validate user preferences
        
        Args:
            preferences (Dict): Raw user preferences
            
        Returns:
            Dict: Normalized preferences with default values
        """
        normalized = {
            'dietary_restrictions': [],
            'budget': float('inf'),
            'nutrition_goals': {},
            'meal_preferences': {},
            'previous_meals': [],
            'disliked_foods': [],
            'favorite_foods': [],
            'height': 170,  # Default height in cm
            'weight': 70,   # Default weight in kg
            'age': 30,      # Default age
            'gender': 'neutral',  # Default gender
            'activity_level': 'moderate'  # Default activity level
        }
        
        # Update with provided preferences
        normalized.update(preferences)
        
        # Convert budget to float if it's a string or int
        if 'budget' in preferences and preferences['budget'] is not None:
            try:
                normalized['budget'] = float(normalized['budget'])
            except (ValueError, TypeError):
                normalized['budget'] = float('inf')
        
        # Ensure lists are lists
        for key in ['dietary_restrictions', 'previous_meals', 'disliked_foods', 'favorite_foods']:
            if key in normalized and not isinstance(normalized[key], list):
                normalized[key] = [normalized[key]]
        
        return normalized
    
    def _calculate_nutrition_targets(self) -> Dict[str, float]:
        """
        Calculate daily nutrition targets based on user profile
        
        Returns:
            Dict: Daily nutrition targets
        """
        # Extract user profile
        height = self.user_preferences.get('height', 170)  # cm
        weight = self.user_preferences.get('weight', 70)   # kg
        age = self.user_preferences.get('age', 30)
        gender = self.user_preferences.get('gender', 'neutral')
        activity_level = self.user_preferences.get('activity_level', 'moderate')
        
        # Activity level multipliers
        activity_multipliers = {
            'sedentary': 1.2,
            'light': 1.375,
            'moderate': 1.55,
            'active': 1.725,
            'very_active': 1.9
        }
        activity_factor = activity_multipliers.get(activity_level, 1.55)
        
        # Calculate BMR (Basal Metabolic Rate) using Mifflin-St Jeor Equation
        if gender.lower() in ['male', 'm']:
            bmr = 10 * weight + 6.25 * height - 5 * age + 5
        elif gender.lower() in ['female', 'f']:
            bmr = 10 * weight + 6.25 * height - 5 * age - 161
        else:
            # Use average for non-binary or unspecified
            bmr = 10 * weight + 6.25 * height - 5 * age - 78
        
        # Calculate TDEE (Total Daily Energy Expenditure)
        tdee = bmr * activity_factor
        
        # Define macronutrient distribution
        # Default: 30% protein, 30% fat, 40% carbs for balanced diet
        protein_ratio = 0.3
        fat_ratio = 0.3
        carb_ratio = 0.4
        
        # Override with user-specified nutrition goals if provided
        if 'nutrition_goals' in self.user_preferences:
            nutrition_goals = self.user_preferences['nutrition_goals']
            
            # If explicit calorie goal is provided
            if 'calories' in nutrition_goals:
                tdee = nutrition_goals['calories']
            
            # If macronutrient ratios are provided
            for nutrient, value in nutrition_goals.items():
                if nutrient == 'protein_ratio' and 0 <= value <= 1:
                    protein_ratio = value
                elif nutrient == 'fat_ratio' and 0 <= value <= 1:
                    fat_ratio = value
                elif nutrient == 'carb_ratio' and 0 <= value <= 1:
                    carb_ratio = value
        
        # Calculate macronutrient targets in grams
        # 1g protein = 4 calories, 1g carbs = 4 calories, 1g fat = 9 calories
        protein_target = (tdee * protein_ratio) / 4  # grams
        fat_target = (tdee * fat_ratio) / 9  # grams
        carb_target = (tdee * carb_ratio) / 4  # grams
        
        # Return nutrition targets
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
        Define the state space for the MDP.
        States represent the user's meal history, nutritional balance,
        and other context.
        
        Returns:
            List[str]: List of state identifiers
        """
        # Extract food categories from the database
        if 'category' in self.food_db.columns:
            # Handle both single categories and list categories
            categories = []
            for cat in self.food_db['category']:
                if isinstance(cat, list):
                    categories.extend(cat)
                else:
                    categories.append(cat)
            
            # Get unique categories
            unique_categories = list(set(categories))
            
            # Store for NBC integration
            self.nbc_categories = unique_categories
        else:
            # Default categories if none in database
            unique_categories = [
                'healthy', 'high-protein', 'low-carb', 'vegetarian', 
                'vegan', 'gluten-free', 'budget-friendly', 'keto'
            ]
            self.nbc_categories = unique_categories
        
        # Create meal type-specific states
        meal_types = ['breakfast', 'lunch', 'dinner']
        
        # For each meal type, we have states representing what category was chosen
        states = []
        for meal_type in meal_types:
            for category in unique_categories:
                states.append(f"{meal_type}_{category}")
        
        # Add nutritional balance states
        nutrition_states = [
            'deficit_protein',   # Low on protein for the day
            'excess_protein',    # Too much protein for the day
            'deficit_calories',  # Under calorie target
            'excess_calories',   # Over calorie target
            'balanced'           # Balanced nutrition
        ]
        states.extend(nutrition_states)
        
        # Add an initial state
        states.append('initial')
        
        # Add variety states (recent meals)
        for category in unique_categories:
            states.append(f"recent_{category}")  # Recently had this category
        
        return states
    
    def _define_actions(self) -> List[str]:
        """
        Define the action space for the MDP.
        Actions are the food items that can be recommended.
        
        Returns:
            List[str]: List of action identifiers (food IDs)
        """
        # Start with all foods
        valid_foods = self.food_db.copy()
        
        # Apply dietary restrictions
        for restriction in self.user_preferences.get('dietary_restrictions', []):
            if restriction.lower() == 'vegetarian':
                # Exclude foods with meat ingredients
                meat_ingredients = ['chicken', 'beef', 'pork', 'fish', 'meat', 'turkey']
                valid_foods = valid_foods[~valid_foods['ingredients'].apply(
                    lambda x: any(ingredient in str(x).lower() for ingredient in meat_ingredients)
                )]
            
            elif restriction.lower() == 'vegan':
                # Exclude foods with animal products
                animal_products = [
                    'chicken', 'beef', 'pork', 'fish', 'meat', 'turkey',
                    'milk', 'cheese', 'yogurt', 'butter', 'cream', 'egg'
                ]
                valid_foods = valid_foods[~valid_foods['ingredients'].apply(
                    lambda x: any(ingredient in str(x).lower() for ingredient in animal_products)
                )]
            
            elif restriction.lower() == 'gluten-free':
                # Exclude foods with gluten
                gluten_ingredients = ['wheat', 'barley', 'rye', 'flour', 'bread', 'pasta']
                valid_foods = valid_foods[~valid_foods['ingredients'].apply(
                    lambda x: any(ingredient in str(x).lower() for ingredient in gluten_ingredients)
                )]
        
        # Apply budget constraint if specified
        if 'budget' in self.user_preferences and self.user_preferences['budget'] < float('inf'):
            budget = self.user_preferences['budget']
            if 'price' in valid_foods.columns:
                valid_foods = valid_foods[valid_foods['price'] <= budget]
        
        # Remove disliked foods
        disliked_foods = self.user_preferences.get('disliked_foods', [])
        if disliked_foods and 'food_id' in valid_foods.columns:
            valid_foods = valid_foods[~valid_foods['food_id'].isin(disliked_foods)]
        
        # Get all valid food IDs
        if 'food_id' in valid_foods.columns:
            actions = valid_foods['food_id'].tolist()
        else:
            # If no food_id column, use index as identifier
            actions = [str(idx) for idx in valid_foods.index]
        
        # If no valid foods found, log warning
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
        Define the reward function for the MDP.
        Rewards are based on nutritional value, budget, and variety.
        
        Returns:
            Dict: Mapping from (state, action) to reward
        """
        rewards = {}
        
        # Build a lookup for food items by ID
        food_lookup = {}
        for idx, row in self.food_db.iterrows():
            food_id = row.get('food_id', str(idx))
            food_lookup[food_id] = row.to_dict()
        
        # Process each state-action pair
        for state in self.states:
            for action in self.actions:
                # Default reward
                reward = 0.0
                
                # Get the food item for this action
                food = food_lookup.get(action, {})
                if not food:
                    continue  # Skip if food not found
                
                # Extract meal type from state if applicable
                meal_type = None
                if state.startswith(('breakfast_', 'lunch_', 'dinner_')):
                    meal_type = state.split('_')[0]
                
                # 1. Reward for meeting nutrition goals based on meal type
                if meal_type and meal_type in self.meal_distribution:
                    meal_targets = {}
                    for nutrient, ratio in self.meal_distribution[meal_type].items():
                        if nutrient in self.daily_nutrition_targets:
                            meal_targets[nutrient] = self.daily_nutrition_targets[nutrient] * ratio
                    
                    # Check how well the food item meets the meal targets
                    for nutrient, target in meal_targets.items():
                        if nutrient in food and not pd.isna(food[nutrient]):
                            # Calculate how close we are to the target
                            value = float(food[nutrient])
                            if nutrient == 'calories':
                                # Penalty for exceeding calorie target, reward for getting close
                                if value > target * 1.2:  # More than 20% over
                                    reward -= 2.0 * (value / target - 1.2)
                                elif value > target * 0.8:  # Within 20% of target
                                    reward += 3.0 * (1.0 - abs(value / target - 1.0))
                                else:  # More than 20% under
                                    reward -= 1.0 * (target * 0.8 - value) / target
                            elif nutrient == 'protein':
                                # Higher reward for meeting protein targets
                                if value >= target * 0.8:  # At least 80% of target
                                    reward += 4.0 * min(value / target, 1.2)
                                else:
                                    reward += 2.0 * (value / target)
                            else:  # carbs and fat
                                # Reward for being close to target
                                reward += 2.0 * (1.0 - min(abs(value / target - 1.0), 0.5) / 0.5)
                
                # 2. Adjust rewards based on state context
                if state.startswith('deficit_'):
                    # Reward for addressing nutritional deficits
                    nutrient = state.split('_')[1]
                    if nutrient in food and not pd.isna(food[nutrient]):
                        value = float(food[nutrient])
                        if nutrient == 'protein':
                            reward += 3.0 * min(value / 20.0, 1.5)  # Extra reward for protein when deficient
                        elif nutrient == 'calories':
                            reward += 2.0 * min(value / 300.0, 1.5)  # Reward for calories when deficient
                
                elif state.startswith('excess_'):
                    # Penalty for adding to excess
                    nutrient = state.split('_')[1]
                    if nutrient in food and not pd.isna(food[nutrient]):
                        value = float(food[nutrient])
                        reward -= 2.0 * min(value / 100.0, 2.0)  # Penalty proportional to excess
                
                elif state == 'balanced':
                    # Reward foods that maintain balance
                    if all(nutrient in food for nutrient in ['calories', 'protein', 'carbs', 'fat']):
                        if not any(pd.isna(food[nutrient]) for nutrient in ['calories', 'protein', 'carbs', 'fat']):
                            # Check if macronutrient ratios are balanced
                            total_cals = float(food['calories'])
                            protein_cals = float(food['protein']) * 4
                            carb_cals = float(food['carbs']) * 4
                            fat_cals = float(food['fat']) * 9
                            
                            # Balanced is roughly 30% protein, 40% carbs, 30% fat
                            protein_ratio = protein_cals / total_cals if total_cals > 0 else 0
                            carb_ratio = carb_cals / total_cals if total_cals > 0 else 0
                            fat_ratio = fat_cals / total_cals if total_cals > 0 else 0
                            
                            # Reward for balanced macros
                            balance_score = (
                                (1.0 - abs(protein_ratio - 0.3)) +
                                (1.0 - abs(carb_ratio - 0.4)) +
                                (1.0 - abs(fat_ratio - 0.3))
                            ) / 3.0
                            
                            reward += 3.0 * balance_score
                
                # 3. Reward for variety
                if state.startswith('recent_'):
                    recent_category = state.split('_')[1]
                    food_category = food.get('category', '')
                    
                    # Convert to list if string
                    if isinstance(food_category, str):
                        food_category = [food_category]
                    
                    # Penalty if the same category was recently consumed
                    if recent_category in food_category:
                        reward -= 3.0
                    else:
                        reward += 1.0  # Bonus for variety
                
                # 4. Reward for user preferences
                # Bonus for favorite foods
                if 'food_id' in food and food['food_id'] in self.user_preferences.get('favorite_foods', []):
                    reward += 4.0
                
                # Bonus for matching meal preferences
                if meal_type and 'meal_preferences' in self.user_preferences:
                    meal_prefs = self.user_preferences['meal_preferences'].get(meal_type, [])
                    food_category = food.get('category', '')
                    
                    # Convert to list if string
                    if isinstance(food_category, str):
                        food_category = [food_category]
                    
                    # Reward if food matches preferred categories for this meal
                    if any(pref in food_category for pref in meal_prefs):
                        reward += 2.5
                
                # 5. Budget considerations
                if 'price' in food and 'budget' in self.user_preferences:
                    price = float(food['price'])
                    budget = self.user_preferences['budget']
                    
                    # Proportional penalty if price exceeds budget
                    if price > budget:
                        reward -= 5.0 * (price / budget)
                    # Reward for being under budget (but not too much)
                    else:
                        reward += 1.0 * (1.0 - max(0.0, budget - price) / budget)
                
                # Store the computed reward
                rewards[(state, action)] = reward
        
        return rewards
    
    def _define_transitions(self) -> Dict[Tuple[str, str], str]:
        """
        Define the transition function for the MDP.
        Transitions model how states change based on actions.
        
        Returns:
            Dict: Mapping from (state, action) to next state
        """
        transitions = {}
        
        # Build a lookup for food items by ID
        food_lookup = {}
        for idx, row in self.food_db.iterrows():
            food_id = row.get('food_id', str(idx))
            food_lookup[food_id] = row.to_dict()
        
        # For each state-action pair, define the transition
        for state in self.states:
            for action in self.actions:
                # Get the food item for this action
                food = food_lookup.get(action, {})
                if not food:
                    continue  # Skip if food not found
                
                # Get food category
                food_category = food.get('category', 'healthy')
                if isinstance(food_category, list) and food_category:
                    # If it's a list, take the first category
                    food_category = food_category[0]
                
                # Get nutritional info
                calories = float(food.get('calories', 0))
                protein = float(food.get('protein', 0))
                
                # Determine next state based on current state and action
                if state.startswith(('breakfast_', 'lunch_', 'dinner_')):
                    # After a meal, transition to a nutrition state based on the meal's nutrients
                    meal_type = state.split('_')[0]
                    next_meal_map = {
                        'breakfast': 'lunch',
                        'lunch': 'dinner',
                        'dinner': 'breakfast'
                    }
                    
                    # If high protein meal, transition to high protein state for next meal
                    if protein > self.daily_nutrition_targets['protein'] * 0.3:  # 30% of daily protein
                        next_state = f"{next_meal_map[meal_type]}_high-protein"
                    # If low calorie meal, transition to deficit calories state
                    elif calories < self.daily_nutrition_targets['calories'] * 0.2:  # 20% of daily calories
                        next_state = 'deficit_calories'
                    # If high calorie meal, transition to excess calories state
                    elif calories > self.daily_nutrition_targets['calories'] * 0.4:  # 40% of daily calories
                        next_state = 'excess_calories'
                    else:
                        # Otherwise go to the balanced state
                        next_state = 'balanced'
                
                elif state.startswith('deficit_'):
                    # If we're deficient in a nutrient and choose a food high in that nutrient,
                    # transition to balanced state
                    nutrient = state.split('_')[1]
                    if nutrient in food and not pd.isna(food[nutrient]):
                        value = float(food[nutrient])
                        if nutrient == 'protein' and value > 20:
                            next_state = 'balanced'
                        elif nutrient == 'calories' and value > 300:
                            next_state = 'balanced'
                        else:
                            # Stay in deficit state
                            next_state = state
                    else:
                        # Stay in deficit state
                        next_state = state
                
                elif state.startswith('excess_'):
                    # If we have excess of a nutrient and choose a food low in that nutrient,
                    # transition to balanced state
                    nutrient = state.split('_')[1]
                    if nutrient in food and not pd.isna(food[nutrient]):
                        value = float(food[nutrient])
                        if nutrient == 'protein' and value < 10:
                            next_state = 'balanced'
                        elif nutrient == 'calories' and value < 200:
                            next_state = 'balanced'
                        else:
                            # Stay in excess state
                            next_state = state
                    else:
                        # Stay in excess state
                        next_state = state
                
                elif state == 'balanced':
                    # From balanced state, transition to a meal state based on time of day
                    # In practice, we'd use current time to determine this
                    # For simplicity, we'll assume breakfast
                    next_state = f"breakfast_{food_category}"
                
                elif state == 'initial':
                    # From initial state, always transition to breakfast state
                    next_state = f"breakfast_{food_category}"
                
                elif state.startswith('recent_'):
                    # If we recently had a food category, transition to the appropriate meal state
                    # For simplicity, we'll use breakfast
                    next_state = f"breakfast_{food_category}"
                
                else:
                    # Default transition to balanced state
                    next_state = 'balanced'
                
                # Store the transition
                transitions[(state, action)] = next_state
        
        return transitions
    
    def value_iteration(self) -> Dict[str, float]:
        """
        Perform value iteration algorithm to find optimal value function
        
        Returns:
            Dict: Optimal value function (state -> value)
        """
        logger.info("Starting value iteration algorithm")
        
        # Initialize value function
        V = {s: 0 for s in self.states}
        
        # Value iteration
        iteration = 0
        while True:
            iteration += 1
            delta = 0
            
            # Update each state
            for s in self.states:
                v = V[s]
                
                # Find the maximum value over all actions
                action_values = []
                for a in self.actions:
                    if (s, a) in self.rewards and (s, a) in self.transitions:
                        # Calculate value: R(s,a) + gamma * V(s')
                        reward = self.rewards[(s, a)]
                        next_s = self.transitions[(s, a)]
                        value = reward + self.gamma * V[next_s]
                        action_values.append(value)
                
                # Update state value if we found valid actions
                if action_values:
                    V[s] = max(action_values)
                    delta = max(delta, abs(v - V[s]))
            
            # Check convergence
            if delta < self.theta:
                logger.info(f"Value iteration converged after {iteration} iterations")
                break
            
            # Prevent infinite loops
            if iteration > 100:
                logger.warning("Value iteration reached max iterations without convergence")
                break
        
        # Store the value function
        self.V = V
        return V
    
    def extract_policy(self) -> Dict[str, str]:
        """
        Extract the optimal policy from the value function
        
        Returns:
            Dict: Mapping from state to optimal action
        """
        logger.info("Extracting optimal policy from value function")
        
        # Initialize policy
        policy = {}
        
        # For each state, find the best action
        for s in self.states:
            best_action = None
            best_value = float('-inf')
            
            # Try all actions
            for a in self.actions:
                if (s, a) in self.rewards and (s, a) in self.transitions:
                    # Calculate value: R(s,a) + gamma * V(s')
                    reward = self.rewards[(s, a)]
                    next_s = self.transitions[(s, a)]
                    value = reward + self.gamma * self.V[next_s]
                    
                    # Update best action if needed
                    if value > best_value:
                        best_value = value
                        best_action = a
            
            # Store the best action for this state
            if best_action:
                policy[s] = best_action
            else:
                # If no valid action found, use first available action
                if self.actions:
                    policy[s] = self.actions[0]
                    logger.warning(f"No valid action found for state {s}, using default")
        
        # Store the policy
        self.policy = policy
        return policy
    
    def recommend_meal(self, current_state: str = None, meal_type: str = None) -> Dict:
        """
        Recommend a meal based on the current state and the optimal policy
        
        Args:
            current_state (str, optional): Current state. If None, uses self.current_state
            meal_type (str, optional): Type of meal to recommend. Used to override state.
            
        Returns:
            Dict: Recommended food item
        """
        if not self.policy:
            logger.warning("Policy not computed yet. Running value iteration and extracting policy.")
            self.value_iteration()
            self.extract_policy()
        
        # Use provided state or current tracked state
        state = current_state or self.current_state
        
        # If meal type is specified, override the state
        if meal_type:
            # Find a compatible state
            meal_states = [s for s in self.states if s.startswith(f"{meal_type}_")]
            if meal_states:
                # Use first compatible state
                state = meal_states[0]
        
        # Check if state is in policy
        if state not in self.policy:
            logger.warning(f"State {state} not in policy. Using initial state.")
            state = 'initial'
        
        # If still not in policy, use fallback
        if state not in self.policy:
            logger.error("Initial state not in policy. Using first action.")
            if self.actions:
                action = self.actions[0]
            else:
                logger.error("No actions available.")
                return None
        else:
            # Get the recommended action
            action = self.policy[state]
        
        # Get the food item
        for idx, row in self.food_db.iterrows():
            food_id = row.get('food_id', str(idx))
            if food_id == action:
                # Update current state based on transition
                if (state, action) in self.transitions:
                    self.current_state = self.transitions[(state, action)]
                
                # Update meal history
                self.meal_history.append({
                    'food_id': food_id,
                    'state': state,
                    'meal_type': meal_type or state.split('_')[0] if state != 'initial' else 'breakfast',
                    'timestamp': datetime.now()
                })
                
                # Return the food item
                return row.to_dict()
        
        logger.error(f"Food item for action {action} not found in database.")
        return None
    
    def plan_meals(self, days: int = 1, meal_types: List[str] = None) -> List[List[Dict]]:
        """
        Plan meals for multiple days
        
        Args:
            days (int): Number of days to plan
            meal_types (List[str]): Types of meals to plan. Defaults to ['breakfast', 'lunch', 'dinner']
            
        Returns:
            List[List[Dict]]: List of meal plans for each day
        """
        if not meal_types:
            meal_types = ['breakfast', 'lunch', 'dinner']
        
        # Make sure policy is computed
        if not self.policy:
            logger.info("Computing policy for meal planning")
            self.value_iteration()
            self.extract_policy()
        
        # Reset state and nutrition tracking
        self.current_state = 'initial'
        self.daily_nutrition_consumed = {
            'calories': 0, 'protein': 0, 'fat': 0, 'carbs': 0
        }
        
        # Plan meals
        meal_plan = []
        
        for day in range(days):
            daily_meals = []
            
            # Reset daily nutrition for the new day
            self.daily_nutrition_consumed = {
                'calories': 0, 'protein': 0, 'fat': 0, 'carbs': 0
            }
            
            for meal_type in meal_types:
                # Recommend a meal for this meal type
                meal = self.recommend_meal(meal_type=meal_type)
                
                if meal:
                    daily_meals.append(meal)
                    
                    # Update daily nutrition consumed
                    for nutrient in ['calories', 'protein', 'fat', 'carbs']:
                        if nutrient in meal and not pd.isna(meal[nutrient]):
                            self.daily_nutrition_consumed[nutrient] += float(meal[nutrient])
                    
                    # Update current state based on nutrition balance
                    self._update_state_based_on_nutrition()
            
            meal_plan.append(daily_meals)
        
        return meal_plan
    
    def _update_state_based_on_nutrition(self):
        """Update current state based on nutrition consumption"""
        # Check protein balance
        protein_consumed = self.daily_nutrition_consumed['protein']
        protein_target = self.daily_nutrition_targets['protein']
        
        if protein_consumed < protein_target * 0.5:  # Less than 50% of target
            self.current_state = 'deficit_protein'
        elif protein_consumed > protein_target * 1.3:  # More than 130% of target
            self.current_state = 'excess_protein'
        
        # Check calorie balance
        calories_consumed = self.daily_nutrition_consumed['calories']
        calories_target = self.daily_nutrition_targets['calories']
        
        if calories_consumed < calories_target * 0.6:  # Less than 60% of target
            self.current_state = 'deficit_calories'
        elif calories_consumed > calories_target * 1.2:  # More than 120% of target
            self.current_state = 'excess_calories'
        
        # If balanced, keep the current state
        if protein_consumed >= protein_target * 0.5 and protein_consumed <= protein_target * 1.3 and \
           calories_consumed >= calories_target * 0.6 and calories_consumed <= calories_target * 1.2:
            self.current_state = 'balanced'
    
    def update_from_feedback(self, meal_id: str, rating: int, food_id: str = None):
        """
        Update the model based on user feedback
        
        Args:
            meal_id (str): ID of the meal that received feedback
            rating (int): User rating (1-5)
            food_id (str, optional): Food ID if different from meal_id
        """
        # Find the meal in history
        meal_entry = None
        for entry in self.meal_history:
            if entry.get('food_id') == meal_id or entry.get('food_id') == food_id:
                meal_entry = entry
                break
        
        if not meal_entry:
            logger.warning(f"Meal {meal_id} not found in history, feedback ignored")
            return
        
        # Get state and action from the meal entry
        state = meal_entry['state']
        action = meal_entry['food_id']
        
        if (state, action) not in self.rewards:
            logger.warning(f"State-action pair ({state}, {action}) not in rewards, feedback ignored")
            return
        
        # Update reward based on rating (1-5)
        # Scale to our reward system (-5 to +5)
        feedback_reward = (rating - 3) * 2.5
        
        # Update the reward
        current_reward = self.rewards[(state, action)]
        # Blend with existing reward (30% feedback, 70% existing)
        self.rewards[(state, action)] = 0.7 * current_reward + 0.3 * feedback_reward
        
        logger.info(f"Updated reward for ({state}, {action}) based on feedback: {current_reward} -> {self.rewards[(state, action)]}")
        
        # Recompute policy if significantly different
        if abs(current_reward - self.rewards[(state, action)]) > 1.0:
            logger.info("Significant reward change, recomputing policy")
            self.value_iteration()
            self.extract_policy()
        
        # Add to favorite foods if highly rated
        if rating >= 4:
            if food_id and food_id not in self.user_preferences.get('favorite_foods', []):
                if 'favorite_foods' not in self.user_preferences:
                    self.user_preferences['favorite_foods'] = []
                self.user_preferences['favorite_foods'].append(food_id)
        
        # Add to disliked foods if poorly rated
        if rating <= 2:
            if food_id and food_id not in self.user_preferences.get('disliked_foods', []):
                if 'disliked_foods' not in self.user_preferences:
                    self.user_preferences['disliked_foods'] = []
                self.user_preferences['disliked_foods'].append(food_id)
    
    def integrate_nbc_categories(self, nbc_categories: List[str], category_weights: Dict[str, float] = None):
        """
        Integrate NBC classifications into the MDP
        
        Args:
            nbc_categories (List[str]): Food categories from NBC
            category_weights (Dict[str, float]): Weights for each category
        """
        # Store NBC categories
        self.nbc_categories = nbc_categories
        
        # Initialize weights if not provided
        if not category_weights:
            category_weights = {cat: 1.0 for cat in nbc_categories}
        
        self.category_weights = category_weights
        
        # Update states to include NBC categories
        new_states = []
        for cat in nbc_categories:
            for meal_type in ['breakfast', 'lunch', 'dinner']:
                new_states.append(f"{meal_type}_{cat}")
            new_states.append(f"recent_{cat}")
        
        # Add to existing states (avoiding duplicates)
        for state in new_states:
            if state not in self.states:
                self.states.append(state)
        
        logger.info(f"Integrated {len(nbc_categories)} NBC categories into MDP")
        logger.info(f"MDP now has {len(self.states)} states")
        
        # Recompute transitions and rewards with new states
        self.rewards = self._define_rewards()
        self.transitions = self._define_transitions()
        
        # Recompute policy
        self.value_iteration()
        self.extract_policy()
    
    def modify_food_category(self, food_id: str, new_category: str):
        """
        Modify the category of a food item based on NBC classification
        
        Args:
            food_id (str): ID of the food to modify
            new_category (str): New category for the food
        """
        # Find the food in the database
        for idx, row in self.food_db.iterrows():
            if row.get('food_id', str(idx)) == food_id:
                # Update the category
                self.food_db.at[idx, 'category'] = new_category
                logger.info(f"Updated category for food {food_id} to {new_category}")
                
                # Recompute rewards and transitions for this food
                for state in self.states:
                    if (state, food_id) in self.rewards:
                        # Recompute reward
                        food = self.food_db.loc[idx].to_dict()
                        self._recompute_reward(state, food_id, food)
                    
                    if (state, food_id) in self.transitions:
                        # Recompute transition
                        self._recompute_transition(state, food_id, new_category)
                
                return True
        
        logger.warning(f"Food {food_id} not found in database, category not modified")
        return False
    
    def _recompute_reward(self, state: str, action: str, food: Dict):
        """
        Recompute reward for a state-action pair
        
        Args:
            state (str): State
            action (str): Action
            food (Dict): Food item data
        """
        # Similar logic to _define_rewards but for a single food
        reward = 0.0
        
        # Extract meal type from state if applicable
        meal_type = None
        if state.startswith(('breakfast_', 'lunch_', 'dinner_')):
            meal_type = state.split('_')[0]
        
        # 1. Reward for meeting nutrition goals based on meal type
        if meal_type and meal_type in self.meal_distribution:
            meal_targets = {}
            for nutrient, ratio in self.meal_distribution[meal_type].items():
                if nutrient in self.daily_nutrition_targets:
                    meal_targets[nutrient] = self.daily_nutrition_targets[nutrient] * ratio
            
            # Check how well the food item meets the meal targets
            for nutrient, target in meal_targets.items():
                if nutrient in food and not pd.isna(food[nutrient]):
                    # Calculate how close we are to the target
                    value = float(food[nutrient])
                    if nutrient == 'calories':
                        # Penalty for exceeding calorie target, reward for getting close
                        if value > target * 1.2:  # More than 20% over
                            reward -= 2.0 * (value / target - 1.2)
                        elif value > target * 0.8:  # Within 20% of target
                            reward += 3.0 * (1.0 - abs(value / target - 1.0))
                        else:  # More than 20% under
                            reward -= 1.0 * (target * 0.8 - value) / target
                    elif nutrient == 'protein':
                        # Higher reward for meeting protein targets
                        if value >= target * 0.8:  # At least 80% of target
                            reward += 4.0 * min(value / target, 1.2)
                        else:
                            reward += 2.0 * (value / target)
                    else:  # carbs and fat
                        # Reward for being close to target
                        reward += 2.0 * (1.0 - min(abs(value / target - 1.0), 0.5) / 0.5)
        
        # Update the reward
        if (state, action) in self.rewards:
            self.rewards[(state, action)] = reward
    
    def _recompute_transition(self, state: str, action: str, category: str):
        """
        Recompute transition for a state-action pair
        
        Args:
            state (str): State
            action (str): Action
            category (str): Food category
        """
        # Similar logic to _define_transitions but for a single food
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
        
        # Update the transition
        if (state, action) in self.transitions:
            self.transitions[(state, action)] = next_state

# Example usage function
def demo_meal_planner(food_db: pd.DataFrame = None, user_preferences: Dict = None):
    """
    Demonstrate the meal planner
    
    Args:
        food_db (pd.DataFrame, optional): Food database
        user_preferences (Dict, optional): User preferences
    
    Returns:
        MealPlannerMDP: Configured meal planner
    """
    # Create dummy food database if not provided
    if food_db is None:
        # Create sample food items
        food_data = []
        
        # Breakfast items
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
        
        # Lunch items
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
        
        # Dinner items
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
        
        # Create DataFrame
        food_db = pd.DataFrame(food_data)
    
    # Create default user preferences if not provided
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
            'height': 175,  # cm
            'weight': 75,   # kg
            'age': 30,
            'gender': 'neutral',
            'activity_level': 'moderate'
        }
    
    # Create and configure the planner
    planner = MealPlannerMDP(food_db, user_preferences)
    
    # Train the model
    planner.value_iteration()
    planner.extract_policy()
    
    # Generate meal plan
    meal_plan = planner.plan_meals(days=3)
    
    # Print the plan
    print("\n3-Day Meal Plan:")
    for day, meals in enumerate(meal_plan, 1):
        print(f"\nDay {day}:")
        for i, meal in enumerate(meals):
            meal_type = ["Breakfast", "Lunch", "Dinner"][i]
            print(f"  {meal_type}: {meal['name']} (${meal['price']:.2f})")
            print(f"    Category: {meal['category']}")
            print(f"    Nutrition: {meal['calories']} cal, {meal['protein']}g protein, {meal['carbs']}g carbs, {meal['fat']}g fat")
    
    return planner