import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, hamming_loss
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import logging
import json
from typing import List, Dict, Union, Tuple, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('FoodCategorizer')

class NutritionFeatureExtractor(BaseEstimator, TransformerMixin):
    """Custom transformer to extract nutritional features from food data"""
    
    def __init__(self, nutrition_cols=None, ranges=None):
        self.nutrition_cols = nutrition_cols or [
            'calories', 'protein', 'fat', 'carbs', 'fiber', 
            'sugar', 'sodium', 'calcium', 'iron', 'potassium'
        ]
        # Define standard ranges for normalization
        self.ranges = ranges or {
            'calories': (0, 1000),
            'protein': (0, 50),
            'fat': (0, 50),
            'carbs': (0, 100),
            'fiber': (0, 15),
            'sugar': (0, 50),
            'sodium': (0, 2000),
            'calcium': (0, 100),
            'iron': (0, 20),
            'potassium': (0, 1000)
        }
        self.scaler = StandardScaler()
        
    def fit(self, X, y=None):
        """Fit the scaler to the data"""
        # Extract available nutrition columns
        nutrition_data = self._extract_nutrition_values(X)
        
        # Fit the scaler
        if nutrition_data.shape[1] > 0:
            self.scaler.fit(nutrition_data)
        return self
        
    def transform(self, X):
        """Transform nutritional data into scaled features"""
        # Extract available nutrition columns
        nutrition_data = self._extract_nutrition_values(X)
        
        # Transform the data if we have columns
        if nutrition_data.shape[1] > 0:
            return self.scaler.transform(nutrition_data)
        return np.array([]).reshape(X.shape[0], 0)
    
    def _extract_nutrition_values(self, X):
        """Extract and normalize nutrition values from DataFrame"""
        # Create a new DataFrame for nutritional features
        nutrition_data = pd.DataFrame(index=X.index)
        
        # Process each nutrition column if it exists
        for col in self.nutrition_cols:
            if col in X.columns:
                # Normalize values to a 0-1 range based on predefined ranges
                min_val, max_val = self.ranges.get(col, (0, 100))
                nutrition_data[col] = X[col].fillna(0).clip(min_val, max_val) / max_val
            else:
                # Try to extract from nutrition_info if it exists
                if 'nutrition_info' in X.columns:
                    nutrition_data[col] = X['nutrition_info'].apply(
                        lambda x: self._extract_from_dict(x, col)
                    ).fillna(0)
        
        return nutrition_data
    
    def _extract_from_dict(self, nutrition_info, key):
        """Extract a value from a nutrition dictionary"""
        if pd.isna(nutrition_info):
            return 0
        
        if isinstance(nutrition_info, str):
            try:
                # Try to parse as JSON
                data = json.loads(nutrition_info)
                return data.get(key, 0)
            except:
                return 0
        elif isinstance(nutrition_info, dict):
            return nutrition_info.get(key, 0)
        return 0

class EnhancedFoodCategorizer:
    """
    Advanced Naive Bayes classifier for categorizing food items based on
    both text descriptions and nutritional information.
    
    This classifier can handle:
    - Multi-label classification (foods can belong to multiple categories)
    - Numerical nutritional features
    - Text-based features from descriptions and ingredients
    - Integration with LLM-generated nutrition information
    """
    
    def __init__(self, multi_label=True):
        """
        Initialize the food categorizer
        
        Args:
            multi_label (bool): Whether to use multi-label classification
        """
        self.multi_label = multi_label
        
        # Define possible food categories
        self.categories = [
            'healthy', 'high-protein', 'low-carb', 'vegetarian', 
            'vegan', 'gluten-free', 'dairy-free', 'breakfast',
            'keto', 'balanced'
        ]
        
        # Define nutrition thresholds for category assignment
        self.nutrition_thresholds = {
            'high-protein': {'protein': 25},  # >25g protein
            'low-carb': {'carbs': 20},        # <20g carbs
            'keto': {'carbs': 15, 'fat': 30}, # <15g carbs, >30g fat
            'balanced': {'protein': 15, 'carbs': 50, 'fat': 15}  # balanced macros
        }
        
        # Set up encoders
        if multi_label:
            self.label_encoder = MultiLabelBinarizer()
        else:
            self.label_encoder = LabelEncoder()
        
        # Feature extractors
        self.text_vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000,
            min_df=2
        )
        
        self.nutrition_extractor = NutritionFeatureExtractor()
        
        # Models for text and nutrition
        self.text_classifier = MultinomialNB(alpha=0.5)
        self.nutrition_classifier = GaussianNB()
        
        # Final ensemble classifier
        self.ensemble_classifier = None
        self.trained = False
        
    def preprocess_data(self, food_db: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare food data for classification
        
        Args:
            food_db (pd.DataFrame): DataFrame containing food items
            
        Returns:
            Tuple: Processed DataFrame and encoded labels
        """
        logger.info(f"Preprocessing data with {len(food_db)} food items")
        
        # Process text features
        if 'full_description' not in food_db.columns:
            food_db['full_description'] = self._create_text_features(food_db)
        
        # Process nutrition features
        food_db = self._standardize_nutrition_features(food_db)
        
        # Assign categories
        if 'category' not in food_db.columns or isinstance(food_db['category'].iloc[0], str):
            categories = self._assign_categories(food_db)
            
            if self.multi_label:
                food_db['category'] = categories
            else:
                # For single-label, take the first category or 'healthy' as default
                food_db['category'] = [cats[0] if cats else 'cilly' for cats in categories]
        
        # Encode labels
        y_encoded = self._encode_labels(food_db['category'])
        
        return food_db, y_encoded
    
    def _create_text_features(self, food_db: pd.DataFrame) -> pd.Series:
        """Create text features by combining relevant text columns"""
        # Combine name, description, and ingredients
        text_features = []
        
        for _, row in food_db.iterrows():
            feature_parts = []
            
            # Add name
            if 'name' in row and not pd.isna(row['name']):
                feature_parts.append(str(row['name']))
            
            # Add description
            if 'description' in row and not pd.isna(row['description']):
                feature_parts.append(str(row['description']))
            
            # Add ingredients
            if 'ingredients' in row:
                if isinstance(row['ingredients'], list):
                    feature_parts.append(' '.join(row['ingredients']))
                elif not pd.isna(row['ingredients']):
                    feature_parts.append(str(row['ingredients']))
            
            # Add nutrition as keywords
            for nutrient in ['calories', 'protein', 'fat', 'carbs']:
                if nutrient in row and not pd.isna(row[nutrient]):
                    feature_parts.append(f"{nutrient}:{row[nutrient]}")
            
            text_features.append(' '.join(feature_parts))
        
        return pd.Series(text_features, index=food_db.index)
    
    def _standardize_nutrition_features(self, food_db: pd.DataFrame) -> pd.DataFrame:
        """Standardize nutrition features, handle missing values"""
        # Make a copy to avoid modifying the original
        df = food_db.copy()
        
        # If nutrition_info exists as JSON or dict, extract into individual columns
        if 'nutrition_info' in df.columns:
            nutrition_keys = ['calories', 'protein', 'fat', 'carbs', 
                              'fiber', 'sugar', 'sodium']
            
            for key in nutrition_keys:
                if key not in df.columns:
                    df[key] = df['nutrition_info'].apply(
                        lambda x: self._extract_nutrition_value(x, key)
                    )
        
        # Fill missing values
        for nutrient in ['calories', 'protein', 'fat', 'carbs']:
            if nutrient in df.columns:
                df[nutrient] = df[nutrient].fillna(df[nutrient].median())
        
        return df
    
    def _extract_nutrition_value(self, nutrition_info, key):
        """Extract a nutrition value from various formats"""
        if pd.isna(nutrition_info):
            return np.nan
            
        # Handle different formats
        if isinstance(nutrition_info, dict):
            return nutrition_info.get(key, np.nan)
        elif isinstance(nutrition_info, str):
            try:
                data = json.loads(nutrition_info)
                return data.get(key, np.nan)
            except:
                return np.nan
        
        return np.nan
    
    def _assign_categories(self, food_db: pd.DataFrame) -> List[List[str]]:
        """
        Assign food categories based on nutrition and ingredients
        
        Args:
            food_db (pd.DataFrame): DataFrame with food items
            
        Returns:
            List[List[str]]: List of category lists for each food item
        """
        all_categories = []
        
        for _, row in food_db.iterrows():
            categories = []
            
            # Get the food name
            name = str(row.get('name', '')).lower()
            
            # Check for explicit breakfast foods by name
            breakfast_items = ['egg', 'bacon', 'sausage', 'pancake', 'waffle', 
                            'toast', 'bagel', 'breakfast', 'cereal', 'oatmeal',
                            'syrup', 'yogurt', 'grits', 'biscuit', 'muffin']
            
            # Is this clearly a breakfast food by name?
            is_breakfast_food = any(item in name for item in breakfast_items)
            
            # Check ingredients for breakfast indicators
            ingredients = []
            if 'ingredients' in row:
                if isinstance(row['ingredients'], list):
                    ingredients = row['ingredients']
                elif isinstance(row['ingredients'], str):
                    try:
                        # Try to parse as JSON list
                        ingredients = json.loads(row['ingredients'])
                    except:
                        # Treat as comma-separated string
                        ingredients = [i.strip() for i in row['ingredients'].split(',')]
            
            # Check if ingredients indicate breakfast
            ingredients_str = str(ingredients).lower()
            if not is_breakfast_food:
                is_breakfast_food = any(item in ingredients_str for item in breakfast_items)
            
            # Add breakfast category if applicable
            if is_breakfast_food:
                categories.append('breakfast')
            
            # Check nutrition-based categories
            if 'protein' in row and not pd.isna(row['protein']):
                # High protein foods
                if row['protein'] > self.nutrition_thresholds['high-protein']['protein']:
                    categories.append('high-protein')
                # Medium protein foods
                elif row['protein'] > 10:  # More lenient threshold for balanced
                    if 'balanced' not in categories:
                        categories.append('balanced')
            
            if 'carbs' in row and not pd.isna(row['carbs']):
                # Low carb foods
                if row['carbs'] < self.nutrition_thresholds['low-carb']['carbs']:
                    categories.append('low-carb')
                
                # Keto foods
                if ('fat' in row and not pd.isna(row['fat']) and 
                    row['carbs'] < self.nutrition_thresholds['keto']['carbs'] and
                    row['fat'] > self.nutrition_thresholds['keto']['fat']):
                    categories.append('keto')
            
            # Check for vegetarian and vegan
            # List of terms that indicate meat content
            meat_terms = ['chicken', 'beef', 'pork', 'fish', 'meat', 'turkey', 'bacon', 'sausage']
            dairy_terms = ['milk', 'cheese', 'yogurt', 'butter', 'cream']
            
            # More careful assessment - only label as vegetarian if we're fairly confident
            has_meat = any(term in name for term in meat_terms) or any(term in ingredients_str for term in meat_terms)
            
            if not has_meat:
                has_dairy = any(term in name for term in dairy_terms) or any(term in ingredients_str for term in dairy_terms)
                
                # Only label vegetarian if we have some evidence (not just by default)
                if not has_meat and ('veg' in name or 'veg' in ingredients_str or 'tofu' in name or 'tofu' in ingredients_str):
                    categories.append('vegetarian')
                    
                    # Only vegan if vegetarian and no dairy
                    if not has_dairy:
                        categories.append('vegan')
            
            # Check for specific food types to differentiate lunch/dinner more
            lunch_terms = ['salad', 'sandwich', 'wrap', 'soup', 'bowl', 'lunch']
            dinner_terms = ['dinner', 'entrée', 'entree', 'steak', 'roast', 'fillet', 'pasta']
            
            # Add lunch/dinner categories to help with meal planning
            if any(term in name for term in lunch_terms):
                categories.append('lunch')
            
            if any(term in name for term in dinner_terms):
                categories.append('dinner')
            
            # Detect healthy foods
            if ('calories' in row and not pd.isna(row['calories']) and
                'fat' in row and not pd.isna(row['fat'])):
                # Lower calorie and lower fat items are generally healthier
                if row['calories'] < 350 and row['fat'] < 15:
                    if 'fruit' in name or 'vegetable' in name or 'salad' in name:
                        categories.append('healthy')
            
            # If no categories assigned and it's not breakfast, add balanced as fallback
            if not categories:
                categories.append('balanced')
            
            all_categories.append(categories)
        
        return all_categories
        
    def _encode_labels(self, categories):
        """Encode categories for model training"""
        if self.multi_label:
            # For multi-label, ensure categories is a list of lists
            if isinstance(categories.iloc[0], str):
                # Convert string categories to list of lists
                categories = [[cat] for cat in categories]
            elif isinstance(categories.iloc[0], list):
                # Ensure each category is a list
                categories = [cat if isinstance(cat, list) else [cat] for cat in categories]
            
            # Fit the encoder and transform
            self.label_encoder.fit(self.categories)  # Fit on all possible categories
            encoded_matrix = self.label_encoder.transform(categories)
            
            # For MultinomialNB, we need a 1D array
            # Create a single category index for each item (choose the first category)
            single_category_indices = []
            for i, row in enumerate(encoded_matrix):
                if np.any(row):  # If any category is present
                    first_index = np.where(row)[0][0]  # Get the first category index
                    single_category_indices.append(first_index)
                else:
                    # Default to the last category index (balanced) if none present
                    balanced_index = np.where(self.label_encoder.classes_ == 'balanced')[0]
                    if len(balanced_index) > 0:
                        single_category_indices.append(balanced_index[0])
                    else:
                        # Just use the first category as fallback
                        single_category_indices.append(0)
            
            return np.array(single_category_indices)
        else:
            # For single-label, ensure categories is a list of strings
            if isinstance(categories.iloc[0], list):
                # Take first category from each list
                categories = [cat[0] if cat and len(cat) > 0 else 'balanced' for cat in categories]
            return self.label_encoder.fit_transform(categories)
    
    def train(self, food_db: pd.DataFrame, params: Dict = None):
        """
        Train the classifier on food data
        
        Args:
            food_db (pd.DataFrame): Food database
            params (Dict, optional): Hyperparameters for model training
        
        Returns:
            EnhancedFoodCategorizer: Self instance for method chaining
        """
        logger.info("Starting classifier training")
        
        # Preprocess data
        processed_db, y_encoded = self.preprocess_data(food_db)
        
        # Extract features
        X_text = processed_db['full_description']
        X_nutrition = processed_db[['calories', 'protein', 'fat', 'carbs']]
        
        # Train text classifier
        logger.info("Training text classifier")
        X_text_vectorized = self.text_vectorizer.fit_transform(X_text)
        self.text_classifier.fit(X_text_vectorized, y_encoded)
        
        # Train nutrition classifier if nutrition data is available
        if not X_nutrition.empty and not X_nutrition.isna().all().all():
            logger.info("Training nutrition classifier")
            X_nutrition_scaled = self.nutrition_extractor.fit_transform(X_nutrition)
            
            if X_nutrition_scaled.shape[1] > 0:
                self.nutrition_classifier.fit(X_nutrition_scaled, y_encoded)
                
                # Train ensemble using a weighted average of both classifiers
                # (implemented directly in the predict method)
                self.ensemble_classifier = True
        
        self.trained = True
        logger.info("Training completed successfully")
        return self
    
    def predict(self, food_items: Union[pd.DataFrame, List[Dict], List[str]], 
            return_proba: bool = False) -> Union[List[str], List[List[str]], np.ndarray]:
        """
        Predict categories for new food items
        
        Args:
            food_items: Food items to categorize (DataFrame, dict list, or text descriptions)
            return_proba: Whether to return probabilities instead of labels
            
        Returns:
            Predicted categories or probability scores
        """
        if not self.trained:
            logger.error("Classifier not trained yet")
            raise RuntimeError("Classifier must be trained before prediction")
        
        # Convert input to correct format
        if isinstance(food_items, list):
            if food_items and isinstance(food_items[0], dict):
                # Convert list of dicts to DataFrame
                food_df = pd.DataFrame(food_items)
            else:
                # List of strings (descriptions)
                food_df = pd.DataFrame({'full_description': food_items})
        else:
            # Already a DataFrame
            food_df = food_items.copy()
        
        # Ensure we have text features
        if 'full_description' not in food_df.columns:
            food_df['full_description'] = self._create_text_features(food_df)
        
        # If there are no training examples, directly assign categories based on rules
        if not self.trained or not hasattr(self, 'text_classifier') or not hasattr(self.text_classifier, 'classes_'):
            logger.warning("Classifier not fully trained, using rule-based categorization")
            return self._rule_based_categorization(food_df)
        
        # Extract text features
        X_text_vectorized = self.text_vectorizer.transform(food_df['full_description'])
        
        # Get predictions from text classifier
        if return_proba:
            text_preds = self.text_classifier.predict_proba(X_text_vectorized)
        else:
            text_preds = self.text_classifier.predict(X_text_vectorized)
        
        # If we have a nutrition classifier, combine predictions
        if self.ensemble_classifier:
            # Check if we have nutrition data
            nutrition_cols = ['calories', 'protein', 'fat', 'carbs']
            if all(col in food_df.columns for col in nutrition_cols):
                # Extract nutrition features
                X_nutrition = food_df[nutrition_cols]
                X_nutrition_scaled = self.nutrition_extractor.transform(X_nutrition)
                
                if X_nutrition_scaled.shape[1] > 0:
                    # Get predictions from nutrition classifier
                    if return_proba:
                        nutrition_preds = self.nutrition_classifier.predict_proba(X_nutrition_scaled)
                        # Combine predictions with weights (0.7 text, 0.3 nutrition)
                        combined_preds = 0.7 * text_preds + 0.3 * nutrition_preds
                        return combined_preds
                    else:
                        nutrition_preds = self.nutrition_classifier.predict(X_nutrition_scaled)
                        # For hard predictions, we'll still use text_preds as primary
                        # but could implement a more sophisticated ensemble approach
        
        # Decode predictions back to category names
        if not return_proba:
            try:
                if self.multi_label:
                    # For multi-label, convert predictions to list of category names
                    if isinstance(text_preds, np.ndarray) and text_preds.ndim == 2:
                        # Convert binary predictions to category names
                        return [self.label_encoder.classes_[np.where(pred == 1)[0]].tolist() 
                                for pred in text_preds]
                    else:
                        # Handle single prediction case
                        return [self.label_encoder.classes_[np.where(pred == 1)[0]].tolist() 
                                for pred in [text_preds]]
                else:
                    # For single-label, directly inverse transform
                    categories = self.label_encoder.inverse_transform(text_preds)
                    
                    # Ensure we return lists for consistency
                    return [[cat] for cat in categories]
            except Exception as e:
                logger.error(f"Error decoding predictions: {e}")
                # Fallback to rule-based categorization
                return self._rule_based_categorization(food_df)
        
        return text_preds

    def _rule_based_categorization(self, food_df: pd.DataFrame) -> List[List[str]]:
        """
        Categorize food items based on rules when the classifier isn't trained
        
        Args:
            food_df: DataFrame with food items
        
        Returns:
            List[List[str]]: Predicted categories for each item
        """
        categories = []
        
        for _, row in food_df.iterrows():
            item_categories = []
            
            # Extract name and ingredients
            name = ''
            if 'name' in row:
                name = str(row['name']).lower()
            elif 'full_description' in row:
                name = str(row['full_description']).lower()
            
            ingredients = []
            if 'ingredients' in row:
                if isinstance(row['ingredients'], list):
                    ingredients = row['ingredients']
                elif isinstance(row['ingredients'], str):
                    try:
                        ingredients = json.loads(row['ingredients'])
                    except:
                        ingredients = row['ingredients'].split(',')
            
            ingredients_str = str(ingredients).lower()
            
            # Check for breakfast items
            breakfast_terms = ['egg', 'bacon', 'sausage', 'pancake', 'waffle', 
                            'toast', 'bagel', 'breakfast', 'cereal', 'oatmeal']
            if any(term in name for term in breakfast_terms) or \
            any(term in ingredients_str for term in breakfast_terms):
                item_categories.append('breakfast')
            
            # Check for protein-rich items
            protein_terms = ['chicken', 'beef', 'pork', 'fish', 'tofu', 'egg', 
                        'turkey', 'protein', 'meat', 'cheese']
            if any(term in name for term in protein_terms) or \
            any(term in ingredients_str for term in protein_terms):
                # Check protein value if available
                if 'protein' in row and not pd.isna(row['protein']) and float(row['protein']) > 15:
                    item_categories.append('high-protein')
            
            # Check for low-carb items
            if 'carbs' in row and not pd.isna(row['carbs']) and float(row['carbs']) < 20:
                item_categories.append('low-carb')
            
            # Check for healthy items
            healthy_terms = ['salad', 'vegetable', 'fruit', 'smoothie', 'yogurt', 'grilled']
            if any(term in name for term in healthy_terms) or \
            any(term in ingredients_str for term in healthy_terms):
                item_categories.append('healthy')
            
            # Check for vegetarian items
            meat_terms = ['chicken', 'beef', 'pork', 'fish', 'meat', 'bacon', 'sausage']
            if not any(term in name for term in meat_terms) and \
            not any(term in ingredients_str for term in meat_terms):
                item_categories.append('vegetarian')
                
                # Check for vegan items
                dairy_terms = ['milk', 'cheese', 'yogurt', 'butter', 'cream']
                if not any(term in name for term in dairy_terms) and \
                not any(term in ingredients_str for term in dairy_terms):
                    item_categories.append('vegan')
            
            # Add balanced as default if no categories assigned
            if not item_categories:
                item_categories.append('balanced')
            
            categories.append(item_categories)
        
        return categories
    
    def save_model(self, filepath: str):
        """Save the trained model to a file"""
        if not self.trained:
            logger.warning("Saving untrained model")
        
        joblib.dump(self, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'EnhancedFoodCategorizer':
        """Load a trained model from a file"""
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model
    
    def process_llm_nutrition_data(self, llm_output: Union[str, Dict, List]) -> pd.DataFrame:
        """
        Process nutrition data from LLM into a format usable by the classifier
        
        Args:
            llm_output: LLM output containing nutrition information
                Can be JSON string, dict, or list of food items
                
        Returns:
            pd.DataFrame: Processed food data
        """
        # Handle different input formats
        if isinstance(llm_output, str):
            try:
                # Try to parse as JSON
                data = json.loads(llm_output)
            except:
                # Treat as a single food description
                return pd.DataFrame({
                    'full_description': [llm_output]
                })
        else:
            data = llm_output
        
        # Convert to DataFrame
        if isinstance(data, list):
            # List of food items
            if data and isinstance(data[0], dict):
                df = pd.DataFrame(data)
            else:
                # List of strings
                df = pd.DataFrame({'full_description': data})
        elif isinstance(data, dict):
            # Single food item
            df = pd.DataFrame([data])
        else:
            raise ValueError("Unsupported LLM output format")
        
        # Standardize column names if they're not in our expected format
        column_mapping = {
            'name': 'name',
            'food_name': 'name',
            'title': 'name',
            'desc': 'description',
            'description': 'description',
            'ingredients_list': 'ingredients',
            'ingredients': 'ingredients',
            'ingredient_list': 'ingredients',
            'nutritional_info': 'nutrition_info',
            'nutrition': 'nutrition_info',
            'nutrition_info': 'nutrition_info'
        }
        
        df = df.rename(columns={old: new for old, new in column_mapping.items() 
                                if old in df.columns and old != new})
        
        # Extract nutrition values if they're in a nested structure
        if 'nutrition_info' in df.columns:
            for col in ['calories', 'protein', 'fat', 'carbs']:
                if col not in df.columns:
                    df[col] = df['nutrition_info'].apply(
                        lambda x: self._extract_nutrition_value(x, col)
                    )
        
        # Create text features if not already present
        if 'full_description' not in df.columns:
            df['full_description'] = self._create_text_features(df)
        
        return df