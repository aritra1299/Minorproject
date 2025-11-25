import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import logging
from config import Config

class AdvancedFeatureExtractor:
    """
    Minimal implementation of an advanced feature extractor that computes
    common timing statistics from a keystroke dataframe. This provides the
    extract_comprehensive_features(df) method expected by DataLoader.
    """
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def extract_comprehensive_features(self, df):
        # Ensure required columns exist and cast to float
        dwell = df['dwell_time'].astype(float)
        flight = df['flight_time'].astype(float)

        total_time = float((dwell.sum() + flight.sum()))
        dwell_mean = float(dwell.mean())
        flight_mean = float(flight.mean())
        dwell_std = float(dwell.std(ddof=0)) if len(dwell) > 0 else 0.0
        flight_std = float(flight.std(ddof=0)) if len(flight) > 0 else 0.0
        dwell_cv = float(dwell_std / dwell_mean) if dwell_mean != 0 else 0.0
        flight_cv = float(flight_std / flight_mean) if flight_mean != 0 else 0.0
        dwell_flight_ratio = float(dwell_mean / flight_mean) if flight_mean != 0 else 0.0

        # Optional pause column if present, otherwise 0
        if 'pause' in df.columns:
            pause_ratio = float(df['pause'].astype(float).mean())
        else:
            pause_ratio = 0.0

        features = {
            'dwell_mean': dwell_mean,
            'flight_mean': flight_mean,
            'total_time': total_time,
            'dwell_std': dwell_std,
            'flight_std': flight_std,
            'dwell_cv': dwell_cv,
            'flight_cv': flight_cv,
            'dwell_flight_ratio': dwell_flight_ratio,
            'pause_ratio': pause_ratio,
        }
        return features

class DataLoader:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.feature_extractor = AdvancedFeatureExtractor()
        
    def load_training_data(self, data_dir=Config.DATA_DIR, start_idx=Config.TRAIN_START_IDX, 
                          end_idx=Config.TRAIN_END_IDX, progress_bar=True):
        """
        Load and process training data with comprehensive error handling
        """
        features_list = []
        labels = []
        file_info = []
        failed_files = []
        
        self.logger.info(f"Loading training data from {data_dir}...")
        
        file_range = range(start_idx, end_idx + 1)
        if progress_bar:
            file_range = tqdm(file_range, desc="Loading samples")
        
        for idx in file_range:
            file_path = os.path.join(data_dir, f'sample{idx}.csv')
            if not os.path.exists(file_path):
                failed_files.append(file_path)
                continue
                
            try:
                df = self._load_and_validate_csv(file_path)
                if df is None:
                    failed_files.append(file_path)
                    continue
                    
                # Extract features
                features = self.feature_extractor.extract_comprehensive_features(df)
                
                features_list.append(features)
                labels.append(1)  # Legitimate user
                file_info.append(f'sample{idx}.csv')
                
            except Exception as e:
                self.logger.warning(f"Failed to process {file_path}: {e}")
                failed_files.append(file_path)
                continue
        
        self.logger.info(f"Successfully loaded {len(features_list)} samples")
        self.logger.info(f"Failed to load {len(failed_files)} samples")
        
        if len(features_list) == 0:
            raise ValueError("No valid training data found!")
            
        return self._create_training_dataset(features_list, labels, file_info)
    
    def _load_and_validate_csv(self, file_path):
        """Load and validate CSV file"""
        try:
            df = pd.read_csv(file_path)
            
            # Standardize column names
            df.columns = df.columns.str.strip().str.lower()
            
            # Validate required columns
            required_columns = ['dwell_time', 'flight_time']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.warning(f"Missing columns in {file_path}: {missing_columns}")
                return None
            
            # Validate data quality
            if len(df) < Config.MIN_KEYSTROKES:
                self.logger.warning(f"Insufficient keystrokes in {file_path}: {len(df)}")
                return None
                
            # Check for extreme outliers
            if (df['dwell_time'] > 5000).any() or (df['flight_time'] > 5000).any():
                self.logger.warning(f"Extreme outliers detected in {file_path}")
                return None
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def _create_training_dataset(self, features_list, labels, file_info):
        """Create balanced training dataset with synthetic impostor data"""
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Generate synthetic impostor data
        impostor_features = self._generate_impostor_data(features_df, n_samples=len(features_list))
        impostor_labels = [0] * len(impostor_features)
        impostor_files = ['synthetic_impostor'] * len(impostor_features)
        
        # Combine datasets
        all_features = pd.concat([features_df, pd.DataFrame(impostor_features)], ignore_index=True)
        all_labels = labels + impostor_labels
        all_file_info = file_info + impostor_files
        
        self.logger.info(f"Final dataset: {len(all_features)} samples ({len(features_df)} legitimate, {len(impostor_features)} impostor)")
        
        return all_features, np.array(all_labels), all_file_info
    
    def _generate_impostor_data(self, legitimate_features, n_samples=1000):
        """Generate realistic impostor data using advanced perturbation"""
        impostor_features = []
        
        for _ in range(n_samples):
            # Select random legitimate sample with bias toward more common patterns
            base_sample = legitimate_features.sample(1).iloc[0].copy()
            
            # Apply sophisticated perturbations
            impostor_sample = self._perturb_features(base_sample)
            impostor_features.append(impostor_sample)
        
        return impostor_features
    
    def _perturb_features(self, base_sample):
        """Apply realistic perturbations to create impostor samples"""
        impostor_sample = base_sample.copy()
        
        # Different perturbation strategies for different feature types
        timing_features = ['dwell_mean', 'flight_mean', 'total_time']
        variability_features = ['dwell_std', 'flight_std', 'dwell_cv', 'flight_cv']
        ratio_features = ['dwell_flight_ratio', 'pause_ratio']
        
        # Perturb timing (20-50% variation)
        for feature in timing_features:
            if feature in impostor_sample:
                variation = np.random.normal(1.0, 0.15)  # 15% std
                impostor_sample[feature] *= max(0.5, min(2.0, variation))
        
        # Perturb variability (30-100% variation)
        for feature in variability_features:
            if feature in impostor_sample:
                variation = np.random.normal(1.0, 0.25)  # 25% std
                impostor_sample[feature] *= max(0.3, min(3.0, variation))
        
        # Perturb ratios (40-150% variation)
        for feature in ratio_features:
            if feature in impostor_sample:
                variation = np.random.normal(1.0, 0.2)  # 20% std
                impostor_sample[feature] *= max(0.4, min(2.5, variation))
        
        return impostor_sample