# Configuration settings
import os
from datetime import datetime

class Config:
    # Paths
    DATA_DIR = 'data'
    MODEL_DIR = 'models'
    LOG_DIR = 'logs'
    
    # Model settings
    MODELS = {
        'RandomForest': {
            'n_estimators': 300,
            'max_depth': 20,
            'min_samples_split': 3,
            'min_samples_leaf': 2,
            'random_state': 42,
            'class_weight': 'balanced'
        },
        'LightGBM': {
            'n_estimators': 500,
            'max_depth': 12,
            'learning_rate': 0.05,
            'random_state': 42,
            'class_weight': 'balanced'
        },
        'XGBoost': {
            'n_estimators': 500,
            'max_depth': 12,
            'learning_rate': 0.05,
            'random_state': 42,
            'eval_metric': 'logloss'
        }
    }
    
    # Feature settings
    FEATURE_SET = [
        'dwell_mean', 'dwell_std', 'dwell_median', 'dwell_min', 'dwell_max',
        'dwell_q25', 'dwell_q75', 'dwell_skew', 'dwell_kurtosis',
        'flight_mean', 'flight_std', 'flight_median', 'flight_min', 'flight_max',
        'flight_q25', 'flight_q75', 'flight_skew', 'flight_kurtosis',
        'dwell_flight_ratio', 'total_time', 'num_keystrokes',
        'dwell_cv', 'flight_cv', 'typing_speed', 'pause_ratio',
        'avg_key_pressure', 'rhythm_consistency', 'error_rate'
    ]
    
    # Security thresholds
    CONFIDENCE_THRESHOLD = 75.0
    ANOMALY_THRESHOLD = 30.0
    MIN_KEYSTROKES = 8
    
    # Training
    TRAIN_START_IDX = 1
    TRAIN_END_IDX = 5106
    TEST_SIZE = 0.2
    CROSS_VALIDATION = 5
    
    # Logging
    LOG_LEVEL = 'INFO'
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        for directory in [cls.DATA_DIR, cls.MODEL_DIR, cls.LOG_DIR]:
            os.makedirs(directory, exist_ok=True)