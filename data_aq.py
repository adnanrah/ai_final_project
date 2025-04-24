import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("adnan adnan")
print("Akshay Kulkarni")
print("hello")
print("test")

# Sample data structure for food items
food_data = {
    "food_id": [],           # Unique identifier
    "name": [],              # Name of the food item
    "location": [],          # Dining hall or restaurant name
    "price": [],             # Price in dollars
    "description": [],       # Text description of the food
    "ingredients": [],       # List of ingredients
    "calories": [],          # Caloric content
    "protein": [],           # Protein content in grams
    "fat": [],               # Fat content in grams
    "carbs": [],             # Carbohydrate content in grams
    "tags": []               # Additional tags like 'vegetarian', 'gluten-free', etc.
}

# Function to scrape UVA dining hall menus
def scrape_uva_dining(dining_hall):
    """
    Scrapes menu information from a UVA dining hall website
    
    Args:
        dining_hall (str): Name of the dining hall ('runk', 'newcomb', or 'ohill')
        
    Returns:
        list: List of dictionaries containing food items
    """
    # Placeholder for actual web scraping code
    # In a real implementation, you would use libraries like requests and BeautifulSoup
    
    print(f"Scraping data from {dining_hall}...")
    
    # Mock data for development
    mock_data = []
    
    if dining_hall == 'runk':
        mock_data = [
            {
                "food_id": "R001",
                "name": "Grilled Chicken Sandwich",
                "location": "Runk",
                "price": 7.99,
                "description": "Grilled chicken breast with lettuce and tomato on a whole wheat bun",
                "ingredients": ["chicken breast", "whole wheat bun", "lettuce", "tomato", "mayo"],
                "calories": 450,
                "protein": 32,
                "fat": 12,
                "carbs": 45,
                "tags": ["high-protein"]
            },
            # Add more mock items...
        ]
    
    # Similar mock data for other dining halls
    
    return mock_data

# Function to get nutritional information from USDA database
def get_nutrition_data(food_name):
    """
    Gets nutritional information for a food item from USDA database
    
    Args:
        food_name (str): Name of the food item
        
    Returns:
        dict: Dictionary containing nutritional information
    """
    # Placeholder for actual API call to USDA database
    # In a real implementation, you would use the USDA API
    
    # Mock nutrition data for development
    mock_nutrition = {
        "calories": np.random.randint(100, 800),
        "protein": np.random.randint(1, 40),
        "fat": np.random.randint(1, 30),
        "carbs": np.random.randint(10, 80)
    }
    
    return mock_nutrition

# Function to combine all data sources
def build_food_database():
    """
    Builds a comprehensive food database from all data sources
    
    Returns:
        pandas.DataFrame: DataFrame containing all food items
    """
    all_food_items = []
    
    # Collect from UVA dining halls
    for hall in ['runk', 'newcomb', 'ohill']:
        all_food_items.extend(scrape_uva_dining(hall))
    
    # Add local restaurant data (placeholder)
    # You would implement similar functions for local restaurants
    
    # Convert to DataFrame
    df = pd.DataFrame(all_food_items)
    
    # Fill missing nutritional information
    for idx, row in df.iterrows():
        if pd.isna(row['calories']):
            nutrition = get_nutrition_data(row['name'])
            df.loc[idx, 'calories'] = nutrition['calories']
            df.loc[idx, 'protein'] = nutrition['protein']
            df.loc[idx, 'fat'] = nutrition['fat']
            df.loc[idx, 'carbs'] = nutrition['carbs']
    
    return df

# Example usage
food_db = build_food_database()
print(f"Database contains {len(food_db)} food items")