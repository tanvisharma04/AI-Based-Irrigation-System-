import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


class IrrigationModelEngine:
    """
    Machine Learning Engine for AI-Based Irrigation System.
    Loads 'irrigation_prediction.csv', trains a Random Forest Classifier pipeline,
    evaluates model metrics, saves the model artifact, and provides single-row predictions
    with calculated water volume and irrigation schedule recommendations.
    """

    def __init__(self, data_path: str = "irrigation_prediction.csv", model_path: str = "irrigation_model.pkl"):
        self.base_dir = Path(__file__).parent.resolve()
        self.data_path = self.base_dir / data_path
        self.model_path = self.base_dir / model_path
        self.pipeline = None
        self.feature_defaults = {}
        self.target_col = "Irrigation_Need"
        
        # Categorical and numerical column lists matching dataset schema
        self.cat_cols = [
            'Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season',
            'Irrigation_Type', 'Water_Source', 'Mulching_Used', 'Region'
        ]
        self.num_cols = [
            'Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 'Electrical_Conductivity',
            'Temperature_C', 'Humidity', 'Rainfall_mm', 'Sunlight_Hours',
            'Wind_Speed_kmh', 'Field_Area_hectare', 'Previous_Irrigation_mm'
        ]
        
        # Mapping frontend form input keys to dataset column names
        self.input_mapping = {
            'first_merging': 'Mulching_Used',
            'first_merging_used': 'Mulching_Used',
            'region': 'Region',
            'water_source': 'Water_Source',
            'irrigation_method': 'Irrigation_Type',
            'irrigation_type': 'Irrigation_Type',
            'season': 'Season',
            'growth_stage': 'Crop_Growth_Stage',
            'crop_growth_stage': 'Crop_Growth_Stage',
            'crop_type': 'Crop_Type',
            'soil_type': 'Soil_Type',
            'sunlight_hours': 'Sunlight_Hours',
            'temperature': 'Temperature_C',
            'temperature_c': 'Temperature_C'
        }

    def train_and_save(self) -> dict:
        """
        Loads CSV dataset, trains Random Forest model, computes feature defaults,
        and saves trained model bundle to disk.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset file not found at {self.data_path}")

        print(f"Loading dataset from {self.data_path}...")
        df = pd.read_csv(self.data_path)

        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        # Compute feature defaults for imputing unprovided UI parameters
        self.feature_defaults = {}
        for col in self.num_cols:
            self.feature_defaults[col] = float(df[col].median())
        for col in self.cat_cols:
            self.feature_defaults[col] = str(df[col].mode()[0])

        # Scikit-learn preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.cat_cols)
            ]
        )

        # Classifier with balanced class weights to handle class imbalance
        classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print("Training Random Forest Classifier model...")
        self.pipeline.fit(X_train, y_train)

        # Evaluate performance
        y_pred = self.pipeline.predict(X_test)
        accuracy = float(accuracy_score(y_test, y_pred))
        report = classification_report(y_test, y_pred)

        print(f"\nModel Training Complete!")
        print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print("\nClassification Report:\n", report)

        # Bundle pipeline and defaults
        model_bundle = {
            'pipeline': self.pipeline,
            'feature_defaults': self.feature_defaults,
            'cat_cols': self.cat_cols,
            'num_cols': self.num_cols,
            'accuracy': accuracy,
            'report': report
        }

        joblib.dump(model_bundle, self.model_path)
        print(f"Model saved successfully to {self.model_path}\n")

        return {'accuracy': accuracy, 'report': report}

    def load(self):
        """Loads trained model bundle from file if exists, otherwise trains a new one."""
        if self.model_path.exists():
            try:
                bundle = joblib.load(self.model_path)
                self.pipeline = bundle['pipeline']
                self.feature_defaults = bundle['feature_defaults']
                return
            except Exception as e:
                print(f"Error loading saved model ({e}), retraining model...")
        
        self.train_and_save()

    def _normalize_input(self, raw_input: dict) -> dict:
        """
        Maps user inputs (e.g. form fields) to official dataset schema features
        and applies sensible default values for unprovided fields.
        """
        cleaned = {}
        
        # Standardize key names
        for raw_key, raw_val in raw_input.items():
            key_lower = str(raw_key).strip().lower()
            mapped_key = self.input_mapping.get(key_lower, raw_key)
            cleaned[mapped_key] = raw_val

        # Prepare final feature row
        row = {}
        
        # Numerical features
        for num_col in self.num_cols:
            val = cleaned.get(num_col, self.feature_defaults.get(num_col, 0.0))
            try:
                # Handle numeric inputs with units (e.g., '30°C' or '8 hrs')
                if isinstance(val, str):
                    val = float(''.join([c for c in val if c.isdigit() or c == '.']))
                else:
                    val = float(val)
            except (ValueError, TypeError):
                val = float(self.feature_defaults.get(num_col, 0.0))
            row[num_col] = val

        # Categorical features
        for cat_col in self.cat_cols:
            val = str(cleaned.get(cat_col, self.feature_defaults.get(cat_col, ''))).strip()
            
            # Map common value variations
            if cat_col == 'Irrigation_Type':
                if val.lower() in ['sprinklers', 'sprinkler']:
                    val = 'Sprinkler'
                elif val.lower() in ['rain fed', 'rainfed']:
                    val = 'Rainfed'
            elif cat_col == 'Crop_Type':
                if val == 'Sugar cane':
                    val = 'Sugarcane'
                elif val == 'Rhode Island':
                    val = 'Wheat'  # Fallback mapping for unsupported UI crop
            
            if not val or val == 'None':
                val = self.feature_defaults.get(cat_col, '')
            
            row[cat_col] = val

        return row

    def predict(self, input_data: dict) -> dict:
        """
        Predicts Irrigation Need (Low, Medium, High) and computes water volume & schedule recommendations.
        """
        if self.pipeline is None:
            self.load()

        feature_row = self._normalize_input(input_data)
        input_df = pd.DataFrame([feature_row])

        # Predict target class and probabilities
        pred_need = str(self.pipeline.predict(input_df)[0])
        probas = self.pipeline.predict_proba(input_df)[0]
        classes = list(self.pipeline.classes_)
        proba_dict = {cls: round(float(prob), 4) for cls, prob in zip(classes, probas)}
        confidence = round(float(max(probas)) * 100, 1)

        # Dynamic Water Volume Recommendation (Liters per Acre)
        temp = feature_row.get('Temperature_C', 30.0)
        sunlight = feature_row.get('Sunlight_Hours', 6.0)

        # Base water requirement & frequency depending on predicted need category
        if pred_need == 'Low':
            base_water = 250
            frequency = "1x Daily (Every 24 hours)"
            best_time = "Early Morning (5:00 AM - 8:00 AM) to minimize evaporation"
            schedule = "1x Daily (Early Morning)"
        elif pred_need == 'Medium':
            base_water = 480
            frequency = "2x Daily (Every 12 hours)"
            best_time = "Early Morning (6:00 AM) & Late Evening (6:00 PM) to minimize evaporation"
            schedule = "2x Daily (Early Morning / Late Evening)"
        else:  # High
            base_water = 750
            frequency = "3x to 4x Daily (High Frequency)"
            best_time = "Early Morning, Early Afternoon & Late Evening to minimize evaporation"
            schedule = "3x - 4x Daily (Immediate / High Frequency)"

        # Temperature & sunlight adjustments
        temp_factor = 1.0 + max(0.0, (temp - 25.0) * 0.015)
        sun_factor = 1.0 + max(0.0, (sunlight - 6.0) * 0.02)
        
        calculated_water = int(round(base_water * temp_factor * sun_factor))
        
        # Convert liters to gallons (1 Liter ≈ 0.264172 Gallons)
        gallons = int(round(calculated_water * 0.264172))

        return {
            "status": pred_need,
            "irrigation_need": pred_need,
            "recommended_water_liters": calculated_water,
            "recommended_water_gallons": gallons,
            "water_volume_display": f"{calculated_water} Liters / Acre ({gallons} Gallons / Acre)",
            "irrigation_frequency": frequency,
            "best_time_of_day": best_time,
            "schedule": schedule,
            "confidence": f"{confidence}%",
            "probabilities": proba_dict
        }


# Singleton engine instance
_engine_instance = None


def predict_irrigation(input_data: dict) -> dict:
    """
    Convenience function for backend endpoints.
    Accepts raw input dictionary and returns model prediction dictionary.
    """
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = IrrigationModelEngine()
        _engine_instance.load()
    return _engine_instance.predict(input_data)


if __name__ == "__main__":
    print("=" * 60)
    print("AI Irrigation System - Machine Learning Backend Engine")
    print("=" * 60)
    
    engine = IrrigationModelEngine()
    metrics = engine.train_and_save()
    
    print("=" * 60)
    print("Testing Model Inference with Sample UI Inputs:")
    print("=" * 60)
    
    sample_input = {
        "first_merging": "Yes",
        "region": "South",
        "water_source": "River",
        "irrigation_method": "Drip",
        "season": "Kharif",
        "growth_stage": "Vegetative",
        "crop_type": "Wheat",
        "soil_type": "Clay",
        "sunlight_hours": "8",
        "temperature": "32"
    }
    
    result = engine.predict(sample_input)
    print(f"Sample Input: {sample_input}")
    print(f"Prediction Result: {result}")
    print("=" * 60)
