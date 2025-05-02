def modify_mdp_planner():
    """
    Modify MealPlannerMDP to improve learning and variety
    """
    from mdp import MealPlannerMDP
    import numpy as np
    from datetime import datetime, timedelta
    
    original_init = MealPlannerMDP.__init__
    
    def enhanced_init(self, food_db, user_preferences):
        self.recommended_items = set()
        self.meal_type_history = {
            'breakfast': [],
            'lunch': [],
            'dinner': []}
        self.item_cooldown = {}
        self.variety_weight = 5.0
        self.learning_rate = 0.4
        self.explore_rate = 0.3
        self.cooldown_days = 3
        original_init(self, food_db, user_preferences)
    MealPlannerMDP.__init__ = enhanced_init
    original_rewards = MealPlannerMDP._define_rewards
    
    def enhanced_rewards(self):
        """Add variety factors to the reward function"""
        rewards = original_rewards(self)
        for state, action in list(rewards.keys()):
            if action in self.item_cooldown:
                last_recommended = self.item_cooldown[action]
                days_since_last = (datetime.now() - last_recommended).days
                if days_since_last < self.cooldown_days:
                    cooldown_penalty = self.variety_weight * (self.cooldown_days - days_since_last) / self.cooldown_days
                    rewards[(state, action)] -= cooldown_penalty
            if action in self.recommended_items:
                rewards[(state, action)] -= self.variety_weight
            if state.startswith(('breakfast_', 'lunch_', 'dinner_')):
                meal_type = state.split('_')[0]
                if meal_type in self.meal_type_history and action in self.meal_type_history[meal_type]:
                    recency_penalty = 0
                    if self.meal_type_history[meal_type]:
                        try:
                            recency_index = self.meal_type_history[meal_type].index(action)
                            recency_penalty = self.variety_weight * (len(self.meal_type_history[meal_type]) - recency_index) / len(self.meal_type_history[meal_type])
                        except ValueError:
                            pass
                    rewards[(state, action)] -= recency_penalty
        return rewards
    
    MealPlannerMDP._define_rewards = enhanced_rewards
    original_recommend = MealPlannerMDP.recommend_meal
    
    def enhanced_recommend(self, current_state=None, meal_type=None):
        """Add exploration to recommendation process"""
        explore = np.random.random() < self.explore_rate
        if explore:
            valid_actions = [a for s, a in self.policy.items() if s == current_state or s == 'initial'] 
            if not valid_actions:
                valid_actions = self.actions.copy()
            current_time = datetime.now()
            valid_actions = [a for a in valid_actions 
                           if a not in self.item_cooldown or 
                           (current_time - self.item_cooldown[a]).days >= self.cooldown_days]
            fresh_actions = [a for a in valid_actions if a not in self.recommended_items]
            if fresh_actions and len(fresh_actions) > 1:
                action = np.random.choice(fresh_actions)
            else:
                action = np.random.choice(valid_actions) if valid_actions else np.random.choice(self.actions)
            for idx, row in self.food_db.iterrows():
                food_id = row.get('food_id', str(idx))
                if food_id == action:
                    self.recommended_items.add(action)
                    self.item_cooldown[action] = datetime.now()
                    if meal_type:
                        if meal_type not in self.meal_type_history:
                            self.meal_type_history[meal_type] = []
                        self.meal_type_history[meal_type].append(action)
                    self.meal_history.append({
                        'food_id': food_id,
                        'state': current_state or 'initial',
                        'meal_type': meal_type or 'breakfast',
                        'timestamp': datetime.now(),
                        'explored': True
                    })
                    return row.to_dict()
        return original_recommend(self, current_state, meal_type)
    MealPlannerMDP.recommend_meal = enhanced_recommend
    original_feedback = MealPlannerMDP.update_from_feedback
    
    def enhanced_feedback(self, meal_id: str, rating: int, food_id: str = None):
        """Process feedback with stronger learning effects"""
        scaled_rating = (rating - 3) * 2.5
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
        meal_type = meal_entry.get('meal_type', '')
        if scaled_rating < 0:
            if meal_type in self.meal_type_history:
                times_to_add = max(1, int(-scaled_rating * 5))
                for _ in range(times_to_add):
                    self.meal_type_history[meal_type].append(action)
            if (state, action) in self.rewards:
                self.rewards[(state, action)] += scaled_rating * 5.0
            self.item_cooldown[action] = datetime.now() - timedelta(days=self.cooldown_days - 1)
        elif scaled_rating > 0:
            try:
                food_item = None
                for idx, row in self.food_db.iterrows():
                    if row.get('food_id', str(idx)) == action:
                        food_item = row.to_dict()
                        break
                if food_item:
                    category = food_item.get('category', '')
                    ingredients = food_item.get('ingredients', [])
                    for s, a in list(self.rewards.keys()):
                        if s.startswith(f"{meal_type}_"):
                            for idx, row in self.food_db.iterrows():
                                if row.get('food_id', str(idx)) == a:
                                    compare_item = row.to_dict()
                                    if compare_item.get('category') == category:
                                        self.rewards[(s, a)] += 0.5 * scaled_rating
                                    if isinstance(ingredients, list) and isinstance(compare_item.get('ingredients', []), list):
                                        common_ingredients = set(ingredients).intersection(set(compare_item.get('ingredients', [])))
                                        if common_ingredients:
                                            similarity = len(common_ingredients) / max(1, min(len(ingredients), len(compare_item.get('ingredients', []))))
                                            self.rewards[(s, a)] += similarity * scaled_rating
            except Exception as e:
                print(f"Error processing positive feedback boost: {e}")
        self.value_iteration()
        self.extract_policy()
    MealPlannerMDP.update_from_feedback = enhanced_feedback
    
    def reset_meal_history(self):
        """Reset all meal tracking data"""
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
    MealPlannerMDP.reset_meal_history = reset_meal_history
    return MealPlannerMDP