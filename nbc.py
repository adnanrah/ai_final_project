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
    """Transformer that extracts nutritional features from food data"""
    def __init__(self, nutrition_cols=None, ranges=None):
        self.nutrition_cols = nutrition_cols or [
            'calories', 'protein', 'fat', 'carbs', 'fiber', 
            'sugar', 'sodium', 'calcium', 'iron', 'potassium'
        ]
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
        nutrition_data = self._extract_nutrition_values(X)
        if nutrition_data.shape[1] > 0:
            self.scaler.fit(nutrition_data)
        return self
        
    def transform(self, X):
        """Transform nutritional data into scaled features"""
        nutrition_data = self._extract_nutrition_values(X)
        if nutrition_data.shape[1] > 0:
            return self.scaler.transform(nutrition_data)
        return np.array([]).reshape(X.shape[0], 0)
    
    def _extract_nutrition_values(self, X):
        """Extract and normalize nutrition values from DataFrame"""
        nutrition_data = pd.DataFrame(index=X.index)
        for col in self.nutrition_cols:
            if col in X.columns:
                min_val, max_val = self.ranges.get(col, (0, 100))
                nutrition_data[col] = X[col].fillna(0).clip(min_val, max_val) / max_val
            else:
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
                data = json.loads(nutrition_info)
                return data.get(key, 0)
            except:
                return 0
        elif isinstance(nutrition_info, dict):
            return nutrition_info.get(key, 0)
        return 0

class EnhancedFoodCategorizer:
    """Naive Bayes classifier for food categorization handling multi-label classification"""
    def __init__(self, multi_label=True):
        """Initialize the food categorizer"""
        self.multi_label = multi_label
        self.categories = [
            'healthy', 'high-protein', 'low-carb', 'vegetarian', 
            'vegan', 'gluten-free', 'dairy-free', 'breakfast',
            'keto', 'balanced'
        ]
        self.nutrition_thresholds = {
            'high-protein': {'protein': 25},
            'low-carb': {'carbs': 20},
            'keto': {'carbs': 15, 'fat': 30},
            'balanced': {'protein': 15, 'carbs': 50, 'fat': 15}
        }
        if multi_label:
            self.label_encoder = MultiLabelBinarizer()
        else:
            self.label_encoder = LabelEncoder()
        self.text_vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000,
            min_df=2
        )
        self.nutrition_extractor = NutritionFeatureExtractor()
        self.text_classifier = MultinomialNB(alpha=0.5)
        self.nutrition_classifier = GaussianNB()
        self.ensemble_classifier = None
        self.trained = False
        
    def preprocess_data(self, food_db: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare food data for classification"""
        logger.info(f"Preprocessing data with {len(food_db)} food items")
        if 'full_description' not in food_db.columns:
            food_db['full_description'] = self._create_text_features(food_db)
        food_db = self._standardize_nutrition_features(food_db)
        if 'category' not in food_db.columns or isinstance(food_db['category'].iloc[0], str):
            categories = self._assign_categories(food_db)
            if self.multi_label:
                food_db['category'] = categories
            else:
                food_db['category'] = [cats[0] if cats else 'cilly' for cats in categories]
        y_encoded = self._encode_labels(food_db['category'])
        return food_db, y_encoded
    
    def _create_text_features(self, food_db: pd.DataFrame) -> pd.Series:
        """Combine relevant text columns into features"""
        text_features = []
        for _, row in food_db.iterrows():
            feature_parts = []
            if 'name' in row and not pd.isna(row['name']):
                feature_parts.append(str(row['name']))
            if 'description' in row and not pd.isna(row['description']):
                feature_parts.append(str(row['description']))
            if 'ingredients' in row:
                if isinstance(row['ingredients'], list):
                    feature_parts.append(' '.join(row['ingredients']))
                elif not pd.isna(row['ingredients']):
                    feature_parts.append(str(row['ingredients']))
            for nutrient in ['calories', 'protein', 'fat', 'carbs']:
                if nutrient in row and not pd.isna(row[nutrient]):
                    feature_parts.append(f"{nutrient}:{row[nutrient]}")
            text_features.append(' '.join(feature_parts))
        return pd.Series(text_features, index=food_db.index)
    
    def _standardize_nutrition_features(self, food_db: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values and extract nutrition from JSON if needed"""
        df = food_db.copy()
        if 'nutrition_info' in df.columns:
            nutrition_keys = ['calories', 'protein', 'fat', 'carbs', 
                             'fiber', 'sugar', 'sodium']
            for key in nutrition_keys:
                if key not in df.columns:
                    df[key] = df['nutrition_info'].apply(
                        lambda x: self._extract_nutrition_value(x, key)
                    )
        for nutrient in ['calories', 'protein', 'fat', 'carbs']:
            if nutrient in df.columns:
                df[nutrient] = df[nutrient].fillna(df[nutrient].median())
        return df
    
    def _extract_nutrition_value(self, nutrition_info, key):
        """Get nutrition value from different possible formats"""
        if pd.isna(nutrition_info):
            return np.nan
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
        """Assign food categories based on nutrition and ingredients"""
        all_categories = []
        for _, row in food_db.iterrows():
            categories = []
            name = str(row.get('name', '')).lower()
            breakfast_items = ['egg', 'bacon', 'sausage', 'pancake', 'waffle', 
                            'toast', 'bagel', 'breakfast', 'cereal', 'oatmeal',
                            'syrup', 'yogurt', 'grits', 'biscuit', 'muffin']
            is_breakfast_food = any(item in name for item in breakfast_items)
            ingredients = []
            if 'ingredients' in row:
                if isinstance(row['ingredients'], list):
                    ingredients = row['ingredients']
                elif isinstance(row['ingredients'], str):
                    try:
                        ingredients = json.loads(row['ingredients'])
                    except:
                        ingredients = [i.strip() for i in row['ingredients'].split(',')]
            ingredients_str = str(ingredients).lower()
            if not is_breakfast_food:
                is_breakfast_food = any(item in ingredients_str for item in breakfast_items)
            if is_breakfast_food:
                categories.append('breakfast')
            if 'protein' in row and not pd.isna(row['protein']):
                if row['protein'] > self.nutrition_thresholds['high-protein']['protein']:
                    categories.append('high-protein')
                elif row['protein'] > 10:
                    if 'balanced' not in categories:
                        categories.append('balanced')
            if 'carbs' in row and not pd.isna(row['carbs']):
                if row['carbs'] < self.nutrition_thresholds['low-carb']['carbs']:
                    categories.append('low-carb')
                if ('fat' in row and not pd.isna(row['fat']) and 
                    row['carbs'] < self.nutrition_thresholds['keto']['carbs'] and
                    row['fat'] > self.nutrition_thresholds['keto']['fat']):
                    categories.append('keto')
            meat_terms = ['chicken', 'beef', 'pork', 'fish', 'meat', 'turkey', 'bacon', 'sausage']
            dairy_terms = ['milk', 'cheese', 'yogurt', 'butter', 'cream']
            has_meat = any(term in name for term in meat_terms) or any(term in ingredients_str for term in meat_terms)
            if not has_meat:
                has_dairy = any(term in name for term in dairy_terms) or any(term in ingredients_str for term in dairy_terms)
                if not has_meat and ('veg' in name or 'veg' in ingredients_str or 'tofu' in name or 'tofu' in ingredients_str):
                    categories.append('vegetarian')
                    if not has_dairy:
                        categories.append('vegan')
            lunch_terms = ['salad', 'sandwich', 'wrap', 'soup', 'bowl', 'lunch']
            dinner_terms = ['dinner', 'entrée', 'entree', 'steak', 'roast', 'fillet', 'pasta']
            if any(term in name for term in lunch_terms):
                categories.append('lunch')
            if any(term in name for term in dinner_terms):
                categories.append('dinner')
            if ('calories' in row and not pd.isna(row['calories']) and
                'fat' in row and not pd.isna(row['fat'])):
                if row['calories'] < 350 and row['fat'] < 15:
                    if 'fruit' in name or 'vegetable' in name or 'salad' in name:
                        categories.append('healthy')
            if not categories:
                categories.append('balanced')
            all_categories.append(categories)
        return all_categories
        
    def _encode_labels(self, categories):
        """Convert category lists/strings to numeric labels for the model"""
        if self.multi_label:
            if isinstance(categories.iloc[0], str):
                categories = [[cat] for cat in categories]
            elif isinstance(categories.iloc[0], list):
                categories = [cat if isinstance(cat, list) else [cat] for cat in categories]
            self.label_encoder.fit(self.categories)
            encoded_matrix = self.label_encoder.transform(categories)
            single_category_indices = []
            for i, row in enumerate(encoded_matrix):
                if np.any(row):
                    first_index = np.where(row)[0][0]
                    single_category_indices.append(first_index)
                else:
                    balanced_index = np.where(self.label_encoder.classes_ == 'balanced')[0]
                    if len(balanced_index) > 0:
                        single_category_indices.append(balanced_index[0])
                    else:
                        single_category_indices.append(0)
            return np.array(single_category_indices)
        else:
            if isinstance(categories.iloc[0], list):
                categories = [cat[0] if cat and len(cat) > 0 else 'balanced' for cat in categories]
            return self.label_encoder.fit_transform(categories)
    
    def train(self, food_db: pd.DataFrame, params: Dict = None):
        """Train the classifier on food data"""
        logger.info("Starting classifier training")
        processed_db, y_encoded = self.preprocess_data(food_db)
        X_text = processed_db['full_description']
        X_nutrition = processed_db[['calories', 'protein', 'fat', 'carbs']]
        logger.info("Training text classifier")
        X_text_vectorized = self.text_vectorizer.fit_transform(X_text)
        self.text_classifier.fit(X_text_vectorized, y_encoded)
        if not X_nutrition.empty and not X_nutrition.isna().all().all():
            logger.info("Training nutrition classifier")
            X_nutrition_scaled = self.nutrition_extractor.fit_transform(X_nutrition)
            if X_nutrition_scaled.shape[1] > 0:
                self.nutrition_classifier.fit(X_nutrition_scaled, y_encoded)
                self.ensemble_classifier = True
        self.trained = True
        logger.info("Training completed successfully")
        return self
    
    def predict(self, food_items: Union[pd.DataFrame, List[Dict], List[str]], 
            return_proba: bool = False) -> Union[List[str], List[List[str]], np.ndarray]:
        """Predict categories for new food items"""
        if not self.trained:
            logger.error("Classifier not trained yet")
            raise RuntimeError("Classifier must be trained before prediction")
        if isinstance(food_items, list):
            if food_items and isinstance(food_items[0], dict):
                food_df = pd.DataFrame(food_items)
            else:
                food_df = pd.DataFrame({'full_description': food_items})
        else:
            food_df = food_items.copy()
        if 'full_description' not in food_df.columns:
            food_df['full_description'] = self._create_text_features(food_df)
        if not self.trained or not hasattr(self, 'text_classifier') or not hasattr(self.text_classifier, 'classes_'):
            logger.warning("Classifier not fully trained, using rule-based categorization")
            return self._rule_based_categorization(food_df)
        X_text_vectorized = self.text_vectorizer.transform(food_df['full_description'])
        if return_proba:
            text_preds = self.text_classifier.predict_proba(X_text_vectorized)
        else:
            text_preds = self.text_classifier.predict(X_text_vectorized)
        if self.ensemble_classifier:
            nutrition_cols = ['calories', 'protein', 'fat', 'carbs']
            if all(col in food_df.columns for col in nutrition_cols):
                X_nutrition = food_df[nutrition_cols]
                X_nutrition_scaled = self.nutrition_extractor.transform(X_nutrition)
                if X_nutrition_scaled.shape[1] > 0:
                    if return_proba:
                        nutrition_preds = self.nutrition_classifier.predict_proba(X_nutrition_scaled)
                        combined_preds = 0.7 * text_preds + 0.3 * nutrition_preds
                        return combined_preds
                    else:
                        nutrition_preds = self.nutrition_classifier.predict(X_nutrition_scaled)
        if not return_proba:
            try:
                if self.multi_label:
                    if isinstance(text_preds, np.ndarray) and text_preds.ndim == 2:
                        return [self.label_encoder.classes_[np.where(pred == 1)[0]].tolist() 
                                for pred in text_preds]
                    else:
                        return [self.label_encoder.classes_[np.where(pred == 1)[0]].tolist() 
                                for pred in [text_preds]]
                else:
                    categories = self.label_encoder.inverse_transform(text_preds)
                    return [[cat] for cat in categories]
            except Exception as e:
                logger.error(f"Error decoding predictions: {e}")
                return self._rule_based_categorization(food_df)
        return text_preds

    def _rule_based_categorization(self, food_df: pd.DataFrame) -> List[List[str]]:
        """Fallback categorization when classifier isn't trained"""
        categories = []
        for _, row in food_df.iterrows():
            item_categories = []
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
            breakfast_terms = ['egg', 'bacon', 'sausage', 'pancake', 'waffle', 
                            'toast', 'bagel', 'breakfast', 'cereal', 'oatmeal']
            if any(term in name for term in breakfast_terms) or \
            any(term in ingredients_str for term in breakfast_terms):
                item_categories.append('breakfast')
            protein_terms = ['chicken', 'beef', 'pork', 'fish', 'tofu', 'egg', 
                        'turkey', 'protein', 'meat', 'cheese']
            if any(term in name for term in protein_terms) or \
            any(term in ingredients_str for term in protein_terms):
                if 'protein' in row and not pd.isna(row['protein']) and float(row['protein']) > 15:
                    item_categories.append('high-protein')
            if 'carbs' in row and not pd.isna(row['carbs']) and float(row['carbs']) < 20:
                item_categories.append('low-carb')
            healthy_terms = ['salad', 'vegetable', 'fruit', 'smoothie', 'yogurt', 'grilled']
            if any(term in name for term in healthy_terms) or \
            any(term in ingredients_str for term in healthy_terms):
                item_categories.append('healthy')
            meat_terms = ['chicken', 'beef', 'pork', 'fish', 'meat', 'bacon', 'sausage']
            if not any(term in name for term in meat_terms) and \
            not any(term in ingredients_str for term in meat_terms):
                item_categories.append('vegetarian')
                dairy_terms = ['milk', 'cheese', 'yogurt', 'butter', 'cream']
                if not any(term in name for term in dairy_terms) and \
                not any(term in ingredients_str for term in dairy_terms):
                    item_categories.append('vegan')
            if not item_categories:
                item_categories.append('balanced')
            categories.append(item_categories)
        return categories
    
    def save_model(self, filepath: str):
        """Save the model to a file"""
        if not self.trained:
            logger.warning("Saving untrained model")
        joblib.dump(self, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'EnhancedFoodCategorizer':
        """Load a model from a file"""
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model
    
    def process_llm_nutrition_data(self, llm_output: Union[str, Dict, List]) -> pd.DataFrame:
        """Process nutrition data from LLM output"""
        if isinstance(llm_output, str):
            try:
                data = json.loads(llm_output)
            except:
                return pd.DataFrame({
                    'full_description': [llm_output]
                })
        else:
            data = llm_output
        if isinstance(data, list):
            if data and isinstance(data[0], dict):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame({'full_description': data})
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            raise ValueError("Unsupported LLM output format")
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
        if 'nutrition_info' in df.columns:
            for col in ['calories', 'protein', 'fat', 'carbs']:
                if col not in df.columns:
                    df[col] = df['nutrition_info'].apply(
                        lambda x: self._extract_nutrition_value(x, col)
                    )
        if 'full_description' not in df.columns:
            df['full_description'] = self._create_text_features(df)
        return df