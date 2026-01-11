"""
src/models.py
-------------
Advanced Model Definitions.
UPDATED: Supports 'Sample Weights' to prioritize high-volatility events.
UPDATED: Supports both single and batch predictions for efficiency.
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️ XGBoost not found. Run 'pip install xgboost' to unlock it.")

class ModelOrchestrator:
    def __init__(self):
        self.scaler = StandardScaler()

        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced'),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced'),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, alpha=0.001, random_state=42)
        }

        if XGB_AVAILABLE:
            self.models['XGBoost'] = XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                eval_metric='mlogloss', # Multi-class logloss
                random_state=42
            )

        self.param_grids = {
            'Random Forest': {'n_estimators': [100, 200], 'max_depth': [3, 5, 8], 'min_samples_leaf': [1, 4]},
            'XGBoost': {'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 4, 5], 'n_estimators': [100, 200]},
            'Neural Network': {'hidden_layer_sizes': [(32,), (64, 32)], 'alpha': [0.0001, 0.001]}
        }

    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, sample_weight: Optional[np.ndarray] = None):
        """Runs RandomizedSearchCV with Sample Weights."""
        X_train_s = self.scaler.fit_transform(X_train)

        for name, grid in self.param_grids.items():
            if name not in self.models: continue

            if name in ['Logistic Regression', 'Neural Network']:
                X_use = X_train_s
            else:
                X_use = X_train

            search = RandomizedSearchCV(
                self.models[name],
                grid,
                n_iter=10,
                cv=3,
                scoring='accuracy',
                n_jobs=-1,
                random_state=42
            )
            try:
                # Direct fit for simplicity in this architecture
                if sample_weight is not None and name != 'Neural Network':
                    search.fit(X_use, y_train, sample_weight=sample_weight)
                else:
                    search.fit(X_use, y_train)

                self.models[name] = search.best_estimator_
            except: pass

    def train_and_evaluate(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, sample_weight: Optional[np.ndarray] = None) -> Tuple[Dict[str, int], Dict[str, float]]:
        """
        Trains models with Volatility Weighted Samples.
        Supports both single predictions and batch predictions.
        """
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        # Detect if we're doing batch prediction or single prediction
        is_batch = len(X_test) > 1

        preds = {}
        probs = {}

        for name, model in self.models.items():
            try:
                # 1. FIT THE MODEL
                if name in ['Logistic Regression', 'Neural Network']:
                    # Neural Net (MLP) in sklearn does NOT support sample_weight in fit
                    if name == 'Neural Network':
                        model.fit(X_train_s, y_train)
                    else:
                        model.fit(X_train_s, y_train, sample_weight=sample_weight)

                    # Predict
                    p_class = model.predict(X_test_s)
                    p_prob = model.predict_proba(X_test_s) # Array of probs for each sample
                else:
                    # Trees support sample weights
                    model.fit(X_train, y_train, sample_weight=sample_weight)
                    p_class = model.predict(X_test)
                    p_prob = model.predict_proba(X_test)

                # 2. STORE PREDICTIONS
                if is_batch:
                    # Return arrays for batch prediction
                    preds[name] = p_class.astype(int)
                    # Max probability for each sample
                    probs[name] = np.max(p_prob, axis=1).astype(float)
                else:
                    # Return single values for single prediction
                    preds[name] = int(p_class[0])
                    probs[name] = float(np.max(p_prob[0]))

            except Exception as e:
                # Handle errors gracefully
                if is_batch:
                    n_samples = len(X_test)
                    preds[name] = np.ones(n_samples, dtype=int)  # Neutral default
                    probs[name] = np.zeros(n_samples, dtype=float)
                else:
                    preds[name] = 1  # Neutral default
                    probs[name] = 0.0

        # Ensemble Logic (Voting)
        if is_batch:
            # Majority vote for each sample
            n_samples = len(X_test)
            ensemble_preds = []
            ensemble_probs = []

            for i in range(n_samples):
                # Get votes from all models for this sample
                votes = [preds[name][i] for name in self.models.keys() if name in preds]
                ensemble_pred = max(set(votes), key=votes.count)
                ensemble_preds.append(ensemble_pred)

                # Average probability across models for this sample
                sample_probs = [probs[name][i] for name in self.models.keys() if name in probs]
                ensemble_probs.append(np.mean(sample_probs))

            preds['Ensemble'] = np.array(ensemble_preds, dtype=int)
            probs['Ensemble'] = np.array(ensemble_probs, dtype=float)
        else:
            # Simple Majority Vote for single prediction
            votes = list(preds.values())
            ensemble_pred = max(set(votes), key=votes.count)

            # Ensemble Confidence
            ensemble_prob = np.mean(list(probs.values()))

            preds['Ensemble'] = ensemble_pred
            probs['Ensemble'] = ensemble_prob

        return preds, probs
