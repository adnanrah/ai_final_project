def modify_mdp_planner():
    """
    Modify the MealPlannerMDP class to improve learning and variety
    
    This function is intended to be imported and run to patch the existing MDP code
    """
    from mdp import MealPlannerMDP
    import numpy as np
    from datetime import datetime, timedelta
    
    # Add a meal history tracking attribute to the class initialization
    original_init = MealPlannerMDP.__init__
    
    def enhanced_init(self, food_db, user_preferences):
        # Set up new attributes first
        self.recommended_items = set()  # Track all recommended items to avoid repetition
        self.meal_type_history = {
            'breakfast': [],
            'lunch': [],
            'dinner': []
        }  # Track meal history by type
        self.item_cooldown = {}  # Track when items were last recommended
        self.variety_weight = 5.0  # Increased weight for variety in recommendations
        self.learning_rate = 0.4  # How quickly the model learns from feedback
        self.explore_rate = 0.3  # Increased probability of exploring new options
        self.cooldown_days = 3  # Number of days before an item can be recommended again
        
        # Then call the original init
        original_init(self, food_db, user_preferences)
    
    # Replace the original init
    MealPlannerMDP.__init__ = enhanced_init
    
    # Enhance the _define_rewards method to better account for variety
    original_rewards = MealPlannerMDP._define_rewards
    
    def enhanced_rewards(self):
        """Enhanced reward function that puts more emphasis on variety"""
        # Get base rewards from original function
        rewards = original_rewards(self)
        
        # Add variety factor to rewards
        for state, action in list(rewards.keys()):  # Create a copy of keys to iterate over
            # Check cooldown period
            if action in self.item_cooldown:
                last_recommended = self.item_cooldown[action]
                days_since_last = (datetime.now() - last_recommended).days
                if days_since_last < self.cooldown_days:
                    # Strong penalty for items still in cooldown
                    cooldown_penalty = self.variety_weight * (self.cooldown_days - days_since_last) / self.cooldown_days
                    rewards[(state, action)] -= cooldown_penalty
            
            # Decrease reward for recently recommended items
            if action in self.recommended_items:
                rewards[(state, action)] -= self.variety_weight
            
            # Check if this is a meal-specific state
            if state.startswith(('breakfast_', 'lunch_', 'dinner_')):
                meal_type = state.split('_')[0]
                # Penalize items recently recommended for this meal type
                if meal_type in self.meal_type_history and action in self.meal_type_history[meal_type]:
                    # More penalty for more recent recommendations
                    recency_penalty = 0
                    if self.meal_type_history[meal_type]:
                        try:
                            recency_index = self.meal_type_history[meal_type].index(action)
                            # Items recommended more recently get more penalty
                            recency_penalty = self.variety_weight * (len(self.meal_type_history[meal_type]) - recency_index) / len(self.meal_type_history[meal_type])
                        except ValueError:
                            pass
                    rewards[(state, action)] -= recency_penalty
        
        return rewards
    
    # Replace the rewards function
    MealPlannerMDP._define_rewards = enhanced_rewards
    
    # Enhance the recommend_meal method
    original_recommend = MealPlannerMDP.recommend_meal
    
    def enhanced_recommend(self, current_state=None, meal_type=None):
        """Enhanced recommendation method with exploration and variety"""
        # Check if we should explore (try something new) based on explore_rate
        explore = np.random.random() < self.explore_rate
        
        if explore:
            # Get all valid actions for the current state
            valid_actions = [a for s, a in self.policy.items() if s == current_state or s == 'initial']
            
            if not valid_actions:
                valid_actions = self.actions.copy()
            
            # Filter out items in cooldown
            current_time = datetime.now()
            valid_actions = [a for a in valid_actions 
                           if a not in self.item_cooldown or 
                           (current_time - self.item_cooldown[a]).days >= self.cooldown_days]
            
            # Filter out recently recommended items if possible
            fresh_actions = [a for a in valid_actions if a not in self.recommended_items]
            
            if fresh_actions and len(fresh_actions) > 1:
                action = np.random.choice(fresh_actions)
            else:
                # If all actions have been recommended, choose randomly from valid actions
                action = np.random.choice(valid_actions) if valid_actions else np.random.choice(self.actions)
            
            # Find the item
            for idx, row in self.food_db.iterrows():
                food_id = row.get('food_id', str(idx))
                if food_id == action:
                    # Update tracking
                    self.recommended_items.add(action)
                    self.item_cooldown[action] = datetime.now()
                    if meal_type:
                        if meal_type not in self.meal_type_history:
                            self.meal_type_history[meal_type] = []
                        self.meal_type_history[meal_type].append(action)
                    
                    # Update meal history
                    self.meal_history.append({
                        'food_id': food_id,
                        'state': current_state or 'initial',
                        'meal_type': meal_type or 'breakfast',
                        'timestamp': datetime.now(),
                        'explored': True
                    })
                    
                    # Return the food item
                    return row.to_dict()
        
        # If not exploring, use the original recommendation method
        return original_recommend(self, current_state, meal_type)
    
    # Replace the recommend method
    MealPlannerMDP.recommend_meal = enhanced_recommend
    
    # Enhance the feedback method
    original_feedback = MealPlannerMDP.update_from_feedback
    
    def enhanced_feedback(self, meal_id: str, rating: int, food_id: str = None):
        """Enhanced feedback processing with stronger variety enforcement"""
        # Scale rating to our reward system (-5 to +5)
        scaled_rating = (rating - 3) * 2.5
        
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
        meal_type = meal_entry.get('meal_type', '')
        
        # If negative rating, add to avoid list with higher weight
        if scaled_rating < 0:
            # Make it less likely to recommend this item for this meal type again
            if meal_type in self.meal_type_history:
                # Add it multiple times based on how negative the rating was
                times_to_add = max(1, int(-scaled_rating * 5))
                for _ in range(times_to_add):
                    self.meal_type_history[meal_type].append(action)
            
            # Update the reward more aggressively
            if (state, action) in self.rewards:
                self.rewards[(state, action)] += scaled_rating * 5.0
            
            # Add to cooldown with extended period
            self.item_cooldown[action] = datetime.now() - timedelta(days=self.cooldown_days - 1)
        elif scaled_rating > 0:
            # For positive ratings, make similar items more likely
            try:
                # Find this food in the database
                food_item = None
                for idx, row in self.food_db.iterrows():
                    if row.get('food_id', str(idx)) == action:
                        food_item = row.to_dict()
                        break
                
                if food_item:
                    # Find similar items and boost their rewards
                    category = food_item.get('category', '')
                    ingredients = food_item.get('ingredients', [])
                    
                    # Boost items with similar category or ingredients
                    for s, a in list(self.rewards.keys()):  # Create a copy of keys to iterate over
                        if s.startswith(f"{meal_type}_"):
                            # Find this action's food
                            for idx, row in self.food_db.iterrows():
                                if row.get('food_id', str(idx)) == a:
                                    compare_item = row.to_dict()
                                    
                                    # Check for similarity
                                    if compare_item.get('category') == category:
                                        # Same category, small boost
                                        self.rewards[(s, a)] += 0.5 * scaled_rating
                                    
                                    # Check ingredients overlap
                                    if isinstance(ingredients, list) and isinstance(compare_item.get('ingredients', []), list):
                                        common_ingredients = set(ingredients).intersection(set(compare_item.get('ingredients', [])))
                                        if common_ingredients:
                                            # Boost based on percentage of common ingredients
                                            similarity = len(common_ingredients) / max(1, min(len(ingredients), len(compare_item.get('ingredients', []))))
                                            self.rewards[(s, a)] += similarity * scaled_rating
            except Exception as e:
                print(f"Error processing positive feedback boost: {e}")
        
        # Always recompute policy after feedback
        self.value_iteration()
        self.extract_policy()
    
    # Replace the feedback method
    MealPlannerMDP.update_from_feedback = enhanced_feedback
    
    # Add a new method to reset meal history for testing
    def reset_meal_history(self):
        """Reset the meal history tracking to start fresh"""
        self.recommended_items = set()
        self.meal_type_history = {
            'breakfast': [],
            'lunch': [],
            'dinner': []
        }
        self.meal_history = []
        self.current_state = 'initial'
        self.daily_nutrition_consumed = {
            'calories': 0, 'protein': 0, 'fat': 0, 'carbs': 0
        }
    
    # Add the new method to the class
    MealPlannerMDP.reset_meal_history = reset_meal_history
    
    # Return the modified class
    return MealPlannerMDP