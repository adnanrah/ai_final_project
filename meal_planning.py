# import numpy as np
# import pandas as pd
# from collections import defaultdict

# class MealPlannerMDP:
#     """
#     Implements a Markov Decision Process for meal planning.
#     Optimizes meal choices based on nutrition, budget, and variety.
#     """
    
#     def __init__(self, food_db, user_preferences):
#         """
#         Initializes the MDP for meal planning
        
#         Args:
#             food_db (pandas.DataFrame): Food database
#             user_preferences (dict): User preferences including:
#                 - dietary_restrictions (list): e.g., ['vegetarian', 'gluten-free']
#                 - budget (float): Maximum budget per meal
#                 - nutrition_goals (dict): e.g., {'protein': 25, 'calories': 500}
#                 - previous_meals (list): List of previous meal IDs
#         """
#         self.food_db = food_db
#         self.user_preferences = user_preferences
        
#         # Define MDP components
#         self.states = self._define_states()
#         self.actions = self._define_actions()
#         self.rewards = self._define_rewards()
#         self.transitions = self._define_transitions()
        
#         # Parameters for value iteration
#         self.gamma = 0.9  # Discount factor
#         self.theta = 0.01  # Convergence threshold
        
#         # Policy and values
#         self.V = {s: 0 for s in self.states}
#         self.policy = {}
    
#     def _define_states(self):
#         """
#         Defines the state space for the MDP.
#         States represent the user's meal history for the last 3 meals.
        
#         Returns:
#             list: List of state identifiers
#         """
#         # In a real implementation, this would be more complex
#         # For simplicity, we'll use the categories of the last 3 meals
        
#         # Get all possible food categories
#         categories = list(self.food_db['category'].unique())
        
#         # Create states from combinations of last 3 meals
#         # For simplicity, we'll just use a single previous meal as the state
#         states = categories.copy()
        
#         # Add an initial state for new users
#         states.append('initial')
        
#         return states
    
#     def _define_actions(self):
#         """
#         Defines the action space for the MDP.
#         Actions are the food items that can be recommended.
        
#         Returns:
#             list: List of action identifiers (food IDs)
#         """
#         # Filter foods based on user preferences
#         filtered_db = self.food_db.copy()
        
#         # Apply dietary restrictions
#         for restriction in self.user_preferences.get('dietary_restrictions', []):
#             if restriction == 'vegetarian':
#                 filtered_db = filtered_db[
#                     ~filtered_db['ingredients'].apply(
#                         lambda x: any(item in ['chicken', 'beef', 'pork', 'fish'] for item in x)
#                     )
#                 ]
#             elif restriction == 'gluten-free':
#                 filtered_db = filtered_db[
#                     ~filtered_db['ingredients'].apply(
#                         lambda x: any(item in ['wheat', 'bread', 'pasta', 'flour'] for item in x)
#                     )
#                 ]
        
#         # Apply budget constraint
#         if 'budget' in self.user_preferences:
#             filtered_db = filtered_db[filtered_db['price'] <= self.user_preferences['budget']]
        
#         # Return food IDs as actions
#         return list(filtered_db['food_id'])
    
#     def _define_rewards(self):
#         """
#         Defines the reward function for the MDP.
#         Rewards are based on nutritional value, budget, and variety.
        
#         Returns:
#             dict: Mapping from (state, action) to reward
#         """
#         rewards = {}
        
#         for state in self.states:
#             for action in self.actions:
#                 # Get the food item
#                 food = self.food_db[self.food_db['food_id'] == action].iloc[0]
                
#                 # Base reward
#                 reward = 0
                
#                 # Reward for meeting nutrition goals
#                 nutrition_goals = self.user_preferences.get('nutrition_goals', {})
#                 for nutrient, goal in nutrition_goals.items():
#                     if nutrient in food and food[nutrient] >= goal * 0.8:
#                         reward += 2
                    
#                 # Penalty for exceeding budget
#                 if 'budget' in self.user_preferences and food['price'] > self.user_preferences['budget']:
#                     reward -= 5
                
#                 # Reward for variety (higher if different from current state)
#                 if state != 'initial' and food['category'] != state:
#                     reward += 3
                
#                 rewards[(state, action)] = reward
        
#         return rewards
    
#     def _define_transitions(self):
#         """
#         Defines the transition probabilities for the MDP.
#         Transitions are deterministic: choosing a meal leads to a state
#         representing that meal category.
        
#         Returns:
#             dict: Mapping from (state, action) to next state
#         """
#         transitions = {}
        
#         for state in self.states:
#             for action in self.actions:
#                 # Get the food category as the next state
#                 food = self.food_db[self.food_db['food_id'] == action].iloc[0]
#                 next_state = food['category']
                
#                 transitions[(state, action)] = next_state
        
#         return transitions
    
#     def value_iteration(self):
#         """
#         Performs value iteration to find the optimal value function.
#         """
#         # Initialize value function
#         V = {s: 0 for s in self.states}
        
#         # Value iteration
#         while True:
#             delta = 0
#             for s in self.states:
#                 v = V[s]
                
#                 # Find the max value over all actions
#                 action_values = []
#                 for a in self.actions:
#                     if (s, a) in self.rewards and (s, a) in self.transitions:
#                         # R(s,a) + gamma * V(s')
#                         next_s = self.transitions[(s, a)]
#                         value = self.rewards[(s, a)] + self.gamma * V[next_s]
#                         action_values.append(value)
                
#                 if action_values:
#                     V[s] = max(action_values)
#                     delta = max(delta, abs(v - V[s]))
            
#             # Check convergence
#             if delta < self.theta:
#                 break
        
#         self.V = V
#         return V
    
#     def extract_policy(self):
#         """
#         Extracts the optimal policy from the value function.
        
#         Returns:
#             dict: Mapping from state to optimal action
#         """
#         policy = {}
        
#         for s in self.states:
#             best_action = None
#             best_value = float('-inf')
            
#             for a in self.actions:
#                 if (s, a) in self.rewards and (s, a) in self.transitions:
#                     next_s = self.transitions[(s, a)]
#                     value = self.rewards[(s, a)] + self.gamma * self.V[next_s]
                    
#                     if value > best_value:
#                         best_value = value
#                         best_action = a
            
#             if best_action:
#                 policy[s] = best_action
        
#         self.policy = policy
#         return policy
    
#     def recommend_meal(self, current_state):
#         """
#         Recommends a meal based on the current state using the optimal policy.
        
#         Args:
#             current_state (str): Current state (food category of last meal)
            
#         Returns:
#             dict: Recommended food item
#         """
#         if current_state not in self.policy:
#             current_state = 'initial'
        
#         if current_state in self.policy:
#             action = self.policy[current_state]
#             food = self.food_db[self.food_db['food_id'] == action].iloc[0].to_dict()
#             return food
#         else:
#             # Fallback recommendation
#             return self.food_db.iloc[0].to_dict()
    
#     def plan_meals(self, days=3, meals_per_day=3):
#         """
#         Plans meals for multiple days.
        
#         Args:
#             days (int): Number of days to plan
#             meals_per_day (int): Number of meals per day
            
#         Returns:
#             list: List of meal plans for each meal
#         """
#         meal_plan = []
#         current_state = 'initial'
        
#         for day in range(days):
#             daily_meals = []
            
#             for meal in range(meals_per_day):
#                 # Get recommendation
#                 recommended_food = self.recommend_meal(current_state)
#                 daily_meals.append(recommended_food)
                
#                 # Update state
#                 current_state = recommended_food['category']
            
#             meal_plan.append(daily_meals)
        
#         return meal_plan

# # Example usage
# def create_meal_planner(food_db):
#     """
#     Creates a meal planner based on user preferences
    
#     Args:
#         food_db (pandas.DataFrame): Food database
        
#     Returns:
#         MealPlannerMDP: Configured meal planner
#     """
#     # Sample user preferences
#     user_preferences = {
#         'dietary_restrictions': ['vegetarian'],
#         'budget': 10.0,
#         'nutrition_goals': {
#             'protein': 20,
#             'calories': 600
#         },
#         'previous_meals': []  # Would be populated with user's meal history
#     }
    
#     # Create and train the meal planner
#     planner = MealPlannerMDP(food_db, user_preferences)
#     planner.value_iteration()
#     planner.extract_policy()
    
#     return planner

# # Demonstrate meal planning
# def demo_meal_planning(planner):
#     """
#     Demonstrates meal planning
    
#     Args:
#         planner (MealPlannerMDP): Trained meal planner
#     """
#     meal_plan = planner.plan_meals(days=3, meals_per_day=3)
    
#     print("3-Day Meal Plan:")
#     for day, meals in enumerate(meal_plan, 1):
#         print(f"\nDay {day}:")
#         for i, meal in enumerate(meals):
#             meal_type = ["Breakfast", "Lunch", "Dinner"][i]
#             print(f"  {meal_type}: {meal['name']} (${meal['price']:.2f})")
#             print(f"    Category: {meal['category']}")
#             print(f"    Nutrition: {meal['calories']} cal, {meal['protein']}g protein")