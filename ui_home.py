import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import random

# Import our modules
# In a real app, you would import the modules we defined earlier
# from food_data_schema import build_food_database
# from naive_bayes_classifier import FoodCategorizer, train_food_categorizer
# from mdp_meal_planner import MealPlannerMDP, create_meal_planner

# Mock functions for demonstration
def generate_mock_food_db():
    """Generate mock food database for demonstration"""
    categories = ['healthy', 'high-protein', 'low-carb', 'vegetarian', 'vegan', 'gluten-free', 'budget-friendly', 'keto']
    locations = ['Runk', 'Newcomb', 'O\'Hill', 'Roots', 'Corner Juice', 'Bodo\'s', 'Got Dumplings']
    
    mock_db = []
    for i in range(100):
        food_item = {
            "food_id": f"F{i:03d}",
            "name": f"Food Item {i}",
            "location": random.choice(locations),
            "price": round(random.uniform(3.0, 15.0), 2),
            "description": f"Delicious food item number {i}",
            "ingredients": random.sample(["chicken", "rice", "beans", "lettuce", "tomato", "cheese", "bread", "pasta"], k=random.randint(2, 5)),
            "calories": random.randint(200, 800),
            "protein": random.randint(5, 40),
            "fat": random.randint(2, 30),
            "carbs": random.randint(10, 80),
            "category": random.choice(categories),
            "tags": random.sample(categories, k=random.randint(1, 3))
        }
        mock_db.append(food_item)
    
    return pd.DataFrame(mock_db)

def recommend_meals(preferences):
    """Generate meal recommendations based on user preferences"""
    # In a real app, this would use the MDP planner
    food_db = generate_mock_food_db()
    
    # Filter based on preferences
    filtered_db = food_db.copy()
    
    # Apply dietary restrictions
    if 'vegetarian' in preferences.get('dietary_restrictions', []):
        filtered_db = filtered_db[filtered_db['category'].isin(['vegetarian', 'vegan'])]
    
    if 'gluten-free' in preferences.get('dietary_restrictions', []):
        filtered_db = filtered_db[filtered_db['category'].isin(['gluten-free'])]
    
    # Apply budget constraint
    if preferences.get('budget'):
        filtered_db = filtered_db[filtered_db['price'] <= preferences['budget']]
    
    # Get 9 recommended meals (3 days x 3 meals)
    recommendations = []
    if len(filtered_db) >= 9:
        recommendations = filtered_db.sample(9).to_dict('records')
    else:
        # If we don't have enough items, duplicate some
        recommendations = (filtered_db.to_dict('records') * 3)[:9]
    
    # Organize into days
    meal_plan = []
    for i in range(0, 9, 3):
        daily_meals = recommendations[i:i+3]
        meal_plan.append(daily_meals)
    
    return meal_plan

def main():
    """Main Streamlit application"""
    st.title("UVA Smart Meal Planner")
    st.subheader("Find affordable, nutritious meals on and around grounds")
    
    # Sidebar for user preferences
    st.sidebar.header("Your Preferences")
    
    # Dietary restrictions
    st.sidebar.subheader("Dietary Restrictions")
    vegetarian = st.sidebar.checkbox("Vegetarian")
    gluten_free = st.sidebar.checkbox("Gluten Free")
    vegan = st.sidebar.checkbox("Vegan")
    
    # Budget
    st.sidebar.subheader("Budget")
    budget = st.sidebar.slider("Maximum price per meal ($)", 5.0, 20.0, 10.0, 0.5)
    
    # Nutrition goals
    st.sidebar.subheader("Nutrition Goals")
    min_protein = st.sidebar.slider("Minimum protein (g)", 0, 50, 20)
    max_calories = st.sidebar.slider("Maximum calories", 300, 1000, 600)
    
    # Location preferences
    st.sidebar.subheader("Locations")
    on_grounds = st.sidebar.checkbox("On Grounds Dining Halls", True)
    off_grounds = st.sidebar.checkbox("Off Grounds Restaurants", True)
    
    # Meal history
    st.sidebar.subheader("Recent Meals")
    st.sidebar.text("Your recent meals will be shown here")
    
    # Main area - tabs for different features
    tab1, tab2, tab3 = st.tabs(["Meal Recommendations", "Food Explorer", "Nutrition Tracker"])
    
    with tab1:
        st.header("Your Personalized Meal Plan")
        
        # Create user preferences dict
        dietary_restrictions = []
        if vegetarian:
            dietary_restrictions.append("vegetarian")
        if gluten_free:
            dietary_restrictions.append("gluten-free")
        if vegan:
            dietary_restrictions.append("vegan")
        
        user_preferences = {
            "dietary_restrictions": dietary_restrictions,
            "budget": budget,
            "nutrition_goals": {
                "protein": min_protein,
                "calories": max_calories
            }
        }
        
        # Get meal recommendations
        if st.button("Generate Meal Plan"):
            meal_plan = recommend_meals(user_preferences)
            
            # Display meal plan
            for day, meals in enumerate(meal_plan, 1):
                st.subheader(f"Day {day}")
                
                # Create three columns for breakfast, lunch, dinner
                cols = st.columns(3)
                
                for i, meal in enumerate(meals):
                    meal_type = ["Breakfast", "Lunch", "Dinner"][i]
                    with cols[i]:
                        st.markdown(f"**{meal_type}**")
                        st.markdown(f"**{meal['name']}**")
                        st.markdown(f"Location: {meal['location']}")
                        st.markdown(f"Price: ${meal['price']:.2f}")
                        st.markdown(f"Calories: {meal['calories']} | Protein: {meal['protein']}g")
                        
                        # Add a "Like" button for feedback
                        if st.button(f"👍 Like this meal", key=f"like_{day}_{i}"):
                            st.success("Thanks for your feedback! We'll improve your recommendations.")
        else:
            st.info("Click 'Generate Meal Plan' to get personalized recommendations")
    
    with tab2:
        st.header("Explore Foods Around UVA")
        
        # Filter options
        col1, col2 = st.columns(2)
        
        with col1:
            category_filter = st.selectbox(
                "Filter by Category",
                ["All", "Healthy", "High-Protein", "Low-Carb", "Vegetarian", "Budget-Friendly"]
            )
        
        with col2:
            location_filter = st.selectbox(
                "Filter by Location",
                ["All", "Runk", "Newcomb", "O'Hill", "Roots", "Corner Juice", "Bodo's"]
            )
        
        # Generate mock data
        food_db = generate_mock_food_db()
        
        # Apply filters
        filtered_db = food_db.copy()
        if category_filter != "All":
            filtered_db = filtered_db[filtered_db]