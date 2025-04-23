
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

class FoodCategorizer:
    """
    Uses Naive Bayes to categorize food items based on their descriptions
    and nutritional information.
    """
    
    def __init__(self):
        self.categories = [
            'healthy', 'high-protein', 'low-carb', 'vegetarian', 
            'vegan', 'gluten-free', 'budget-friendly', 'keto'
        ]
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.classifier = MultinomialNB()
        self.label_encoder = LabelEncoder()
        
    def preprocess_data(self, food_db):
        """
        Prepares food data for classification
        
        Args:
            food_db (pandas.DataFrame): DataFrame of food items
            
        Returns:
            tuple: X (features) and y (labels) for training
        """
        # Creates a combined text description for each food item
        food_db['full_description'] = food_db.apply(
            lambda row: f"{row['name']} {row['description']} {' '.join(row['ingredients'])} "
                       f"calories:{row['calories']} protein:{row['protein']} "
                       f"fat:{row['fat']} carbs:{row['carbs']}",
            axis=1
        )
        
        def assign_mock_categories(row):
            categories = []
            if row['protein'] > 25:
                categories.append('high-protein')
            if row['carbs'] < 20:
                categories.append('low-carb')
            if 'chicken' not in row['ingredients'] and 'beef' not in row['ingredients']:
                categories.append('vegetarian')
            if row['price'] < 5.0:
                categories.append('budget-friendly')
            if not categories:
                categories.append('healthy')  # Default category
            return np.random.choice(categories)  # Select one category for simplicity
        
        food_db['category'] = food_db.apply(assign_mock_categories, axis=1)
        
        X = food_db['full_description']
        y = food_db['category']
        
        # Encode category labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        return X, y_encoded
    
    def train(self, X, y):
        """
        Trains the Naive Bayes classifier
        
        Args:
            X: Feature matrix (text descriptions)
            y: Target labels (encoded categories)
        """
        # Transform text data into feature vectors
        X_vectorized = self.vectorizer.fit_transform(X)
        
        # Train the model
        self.classifier.fit(X_vectorized, y)
        print("Model trained successfully")
    
    def predict(self, food_descriptions):
        """
        Predicts categories for new food descriptions
        
        Args:
            food_descriptions (list): List of food descriptions
            
        Returns:
            list: Predicted categories
        """
        X_vectorized = self.vectorizer.transform(food_descriptions)
        y_pred = self.classifier.predict(X_vectorized)
        
        # Decode labels back to category names
        return self.label_encoder.inverse_transform(y_pred)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluates the classifier performance
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Performance metrics
        """
        X_test_vectorized = self.vectorizer.transform(X_test)
        y_pred = self.classifier.predict(X_test_vectorized)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, 
                                     target_names=self.label_encoder.classes_)
        
        print(f"Accuracy: {accuracy:.2f}")
        print("Classification Report:")
        print(report)
        
        return {"accuracy": accuracy, "report": report}

# Example usage:
def train_food_categorizer(food_db):
    """
    Trains and evaluates the food categorizer
    
    Args:
        food_db (pandas.DataFrame): Food database
        
    Returns:
        FoodCategorizer: Trained food categorizer
    """
    categorizer = FoodCategorizer()
    
    # Preprocess data
    X, y = categorizer.preprocess_data(food_db)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    categorizer.train(X_train, y_train)
    
    # Evaluate the model
    categorizer.evaluate(X_test, y_test)
    
    return categorizer

# Sample prediction
def demo_prediction(categorizer):
    """
    Demonstrates prediction on new food items
    
    Args:
        categorizer (FoodCategorizer): Trained food categorizer
    """
    new_foods = [
        "Grilled salmon with steamed broccoli and brown rice, 350 calories, 30g protein, 10g fat, 25g carbs",
        "Cheeseburger with french fries, 850 calories, 25g protein, 45g fat, 80g carbs",
        "Garden salad with tofu, quinoa, and vinaigrette, 300 calories, 15g protein, 12g fat, 30g carbs"
    ]
    
    predictions = categorizer.predict(new_foods)
    
    for food, category in zip(new_foods, predictions):
        print(f"Food: {food[:50]}...")
        print(f"Predicted category: {category}")
        print()