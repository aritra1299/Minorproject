import pandas as pd
import numpy as np
import os
import pickle  
import json
import logging
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from config import Config
from utils.data_loader import DataLoader
from utils.feature_extractor import AdvancedFeatureExtractor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedModelTrainer:
    def __init__(self):
        self.data_loader = DataLoader()
        self.scaler = StandardScaler()
        self.models = {}
        self.training_history = {}
        
    def train_ensemble_model(self):
        """Train optimized ensemble model with comprehensive evaluation"""
        logger.info("Starting optimized model training...")
        
        try:
            # Load and prepare data
            X, y, file_info = self.data_loader.load_training_data()
            logger.info(f"Dataset shape: {X.shape}")
            
            # Handle missing values
            X = self._handle_missing_values(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=Config.TEST_SIZE, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train models
            trained_models = self._train_individual_models(X_train_scaled, y_train)
            
            # Evaluate models
            evaluation_results = self._evaluate_models(trained_models, X_test_scaled, y_test)
            
            # Select best model
            best_model_name = self._select_best_model(evaluation_results)
            
            # Perform cross-validation
            cv_results = self._cross_validate_models(trained_models, X_train_scaled, y_train)
            
            # Save models and artifacts
            self._save_training_artifacts(trained_models, best_model_name, X.columns, 
                                        evaluation_results, cv_results, file_info)
            
            # Generate reports
            self._generate_training_reports(trained_models, X_test_scaled, y_test, evaluation_results)
            
            logger.info("✅ Model training completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
    
    def _handle_missing_values(self, X):
        """Handle missing values intelligently"""
        # For numeric columns, use median
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].median())
        
        return X
    
    def _train_individual_models(self, X_train, y_train):
        """Train individual models with optimized parameters"""
        trained_models = {}
        
        # Random Forest
        logger.info("Training Random Forest...")
        rf_model = RandomForestClassifier(**Config.MODELS['RandomForest'])
        rf_model.fit(X_train, y_train)
        trained_models['RandomForest'] = rf_model
        
        # LightGBM
        logger.info("Training LightGBM...")
        lgb_model = LGBMClassifier(**Config.MODELS['LightGBM'])
        lgb_model.fit(X_train, y_train)
        trained_models['LightGBM'] = lgb_model
        
        # XGBoost
        logger.info("Training XGBoost...")
        xgb_model = XGBClassifier(**Config.MODELS['XGBoost'])
        xgb_model.fit(X_train, y_train)
        trained_models['XGBoost'] = xgb_model
        
        return trained_models
    
    def _evaluate_models(self, models, X_test, y_test):
        """Comprehensive model evaluation"""
        evaluation_results = {}
        
        for name, model in models.items():
            logger.info(f"Evaluating {name}...")
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            accuracy = np.mean(y_pred == y_test)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # Precision-Recall analysis
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = auc(recall, precision)
            
            # Store results
            evaluation_results[name] = {
                'accuracy': accuracy,
                'auc_score': auc_score,
                'pr_auc': pr_auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'model': model
            }
            
            logger.info(f"  {name} - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}, PR-AUC: {pr_auc:.4f}")
        
        return evaluation_results
    
    def _select_best_model(self, evaluation_results):
        """Select the best model based on multiple metrics"""
        best_score = -1
        best_model = None
        
        for name, results in evaluation_results.items():
            # Combined score (weighted average)
            combined_score = (
                0.4 * results['accuracy'] +
                0.4 * results['auc_score'] + 
                0.2 * results['pr_auc']
            )
            
            if combined_score > best_score:
                best_score = combined_score
                best_model = name
        
        logger.info(f"🏆 Best model: {best_model} (score: {best_score:.4f})")
        return best_model
    
    def _cross_validate_models(self, models, X, y):
        """Perform cross-validation for robust evaluation"""
        cv_results = {}
        skf = StratifiedKFold(n_splits=Config.CROSS_VALIDATION, shuffle=True, random_state=42)
        
        for name, model in models.items():
            logger.info(f"Cross-validating {name}...")
            
            cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
            cv_auc_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
            
            cv_results[name] = {
                'accuracy_mean': cv_scores.mean(),
                'accuracy_std': cv_scores.std(),
                'auc_mean': cv_auc_scores.mean(),
                'auc_std': cv_auc_scores.std()
            }
            
            logger.info(f"  {name} CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            logger.info(f"  {name} CV AUC: {cv_auc_scores.mean():.4f} (±{cv_auc_scores.std():.4f})")
        
        return cv_results
    
    def _save_training_artifacts(self, models, best_model, feature_names, 
                               evaluation_results, cv_results, file_info):
        """Save all training artifacts"""
        Config.setup_directories()
        
        # Save individual models
        for name, model in models.items():
            model_path = os.path.join(Config.MODEL_DIR, f'{name.lower()}_model.joblib')
            joblib.dump(model, model_path)
        
        # Save ensemble model data
        ensemble_data = {
            'models': {name: model for name, model in models.items()},
            'best_model': best_model,
            'scaler': self.scaler,
            'feature_names': feature_names.tolist(),
            'feature_extractor': AdvancedFeatureExtractor(),
            'evaluation_results': evaluation_results,
            'cv_results': cv_results,
            'training_info': {
                'timestamp': datetime.now().isoformat(),
                'num_samples': len(file_info),
                'feature_count': len(feature_names),
                'best_model_score': evaluation_results[best_model]['auc_score']
            }
        }
        
        ensemble_path = os.path.join(Config.MODEL_DIR, 'keystroke_ensemble_model.joblib')
        joblib.dump(ensemble_data, ensemble_path)
        
        # Save model info
        model_info = {
            'best_model': best_model,
            'available_models': list(models.keys()),
            'feature_count': len(feature_names),
            'training_date': datetime.now().isoformat(),
            'performance_metrics': {
                name: {
                    'accuracy': results['accuracy'],
                    'auc_score': results['auc_score'],
                    'pr_auc': results['pr_auc']
                } for name, results in evaluation_results.items()
            }
        }
        
        info_path = os.path.join(Config.MODEL_DIR, 'model_info.json')
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"💾 Models and artifacts saved to {Config.MODEL_DIR}/")
    
    def _generate_training_reports(self, models, X_test, y_test, evaluation_results):
        """Generate comprehensive training reports and visualizations"""
        try:
            # Create feature importance plot
            self._plot_feature_importance(models, X_test.shape[1])
            
            # Create performance comparison
            self._plot_model_comparison(evaluation_results)
            
            # Create confusion matrices
            self._plot_confusion_matrices(models, X_test, y_test)
            
        except Exception as e:
            logger.warning(f"Could not generate reports: {e}")
    
    def _plot_feature_importance(self, models, num_features):
        """Plot feature importance for all models"""
        fig, axes = plt.subplots(1, len(models), figsize=(20, 6))
        if len(models) == 1:
            axes = [axes]
        
        for idx, (name, model) in enumerate(models.items()):
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1][:15]
                
                axes[idx].barh(range(15), importances[indices])
                axes[idx].set_yticks(range(15))
                # Note: We don't have feature names here, but in practice you would use them
                axes[idx].set_yticklabels([f'Feature {i}' for i in indices])
                axes[idx].set_title(f'{name} - Top 15 Features')
                axes[idx].set_xlabel('Feature Importance')
        
        plt.tight_layout()
        plt.savefig(os.path.join(Config.MODEL_DIR, 'feature_importance.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_comparison(self, evaluation_results):
        """Plot model performance comparison"""
        models = list(evaluation_results.keys())
        accuracy = [results['accuracy'] for results in evaluation_results.values()]
        auc_scores = [results['auc_score'] for results in evaluation_results.values()]
        
        x = np.arange(len(models))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, accuracy, width, label='Accuracy')
        bars2 = ax.bar(x + width/2, auc_scores, width, label='AUC Score')
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Scores')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(Config.MODEL_DIR, 'model_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrices(self, models, X_test, y_test):
        """Plot confusion matrices for all models"""
        fig, axes = plt.subplots(1, len(models), figsize=(15, 5))
        if len(models) == 1:
            axes = [axes]
        
        for idx, (name, model) in enumerate(models.items()):
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{name} - Confusion Matrix')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
            axes[idx].set_xticklabels(['Impostor', 'Legitimate'])
            axes[idx].set_yticklabels(['Impostor', 'Legitimate'])
        
        plt.tight_layout()
        plt.savefig(os.path.join(Config.MODEL_DIR, 'confusion_matrices.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main training function"""
    try:
        trainer = OptimizedModelTrainer()
        trainer.train_ensemble_model()
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"📁 Models saved in: {Config.MODEL_DIR}/")
        print(f"📊 Training reports generated")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())