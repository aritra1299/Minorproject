import pandas as pd
import numpy as np
from scipy import stats
import logging

class AdvancedFeatureExtractor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def extract_comprehensive_features(self, df):
        """
        Extract comprehensive features from keystroke data with error handling
        """
        try:
            # Validate input
            if df.empty or 'dwell_time' not in df.columns or 'flight_time' not in df.columns:
                raise ValueError("Invalid dataframe or missing required columns")
            
            dwell = pd.Series(df['dwell_time'].dropna())
            flight = pd.Series(df['flight_time'].dropna())
            
            if len(dwell) < 3 or len(flight) < 2:
                raise ValueError("Insufficient keystroke data")
            
            features = {}
            
            # Basic dwell time features
            features.update(self._extract_basic_stats(dwell, 'dwell'))
            
            # Basic flight time features  
            features.update(self._extract_basic_stats(flight, 'flight'))
            
            # Advanced statistical features
            features.update(self._extract_advanced_stats(dwell, flight))
            
            # Rhythm and timing patterns
            features.update(self._extract_rhythm_features(dwell, flight))
            
            # Error pattern detection
            features.update(self._extract_error_patterns(df))
            
            # Consistency metrics
            features.update(self._extract_consistency_metrics(dwell, flight))
            
            return features
            
        except Exception as e:
            self.logger.error(f"Feature extraction error: {e}")
            return self._get_default_features()
    
    def _extract_basic_stats(self, series, prefix):
        """Extract basic statistical features"""
        return {
            f'{prefix}_mean': float(series.mean()),
            f'{prefix}_std': float(series.std()),
            f'{prefix}_median': float(series.median()),
            f'{prefix}_min': float(series.min()),
            f'{prefix}_max': float(series.max()),
            f'{prefix}_q25': float(series.quantile(0.25)),
            f'{prefix}_q75': float(series.quantile(0.75)),
            f'{prefix}_skew': float(series.skew()),
            f'{prefix}_kurtosis': float(series.kurtosis()),
        }
    
    def _extract_advanced_stats(self, dwell, flight):
        """Extract advanced statistical features"""
        # Remove outliers using IQR
        dwell_clean = self._remove_outliers(dwell)
        flight_clean = self._remove_outliers(flight)
        
        features = {
            'dwell_flight_ratio': dwell.mean() / flight.mean() if flight.mean() > 0 else 0,
            'dwell_cv': dwell.std() / dwell.mean() if dwell.mean() > 0 else 0,
            'flight_cv': flight.std() / flight.mean() if flight.mean() > 0 else 0,
            'total_time': dwell.sum() + flight.sum(),
            'num_keystrokes': len(dwell),
            'typing_speed': len(dwell) / ((dwell.sum() + flight.sum()) / 1000) if (dwell.sum() + flight.sum()) > 0 else 0,
        }
        
        # Clean data statistics
        if len(dwell_clean) > 0:
            features['dwell_clean_mean'] = dwell_clean.mean()
            features['dwell_clean_std'] = dwell_clean.std()
        
        if len(flight_clean) > 0:
            features['flight_clean_mean'] = flight_clean.mean()
            features['flight_clean_std'] = flight_clean.std()
            
        return features
    
    def _extract_rhythm_features(self, dwell, flight):
        """Extract rhythm and timing pattern features"""
        # Pause detection
        long_pauses = len(flight[flight > 500])
        very_long_pauses = len(flight[flight > 1000])
        
        # Rhythm consistency (CV of consecutive keystroke intervals)
        if len(dwell) > 1:
            intervals = dwell.values[:-1] + flight.values[1:]
            rhythm_cv = intervals.std() / intervals.mean() if intervals.mean() > 0 else 0
        else:
            rhythm_cv = 0
            
        return {
            'pause_ratio': long_pauses / len(flight) if len(flight) > 0 else 0,
            'very_long_pause_ratio': very_long_pauses / len(flight) if len(flight) > 0 else 0,
            'rhythm_consistency': rhythm_cv,
            'avg_key_pressure': dwell.mean() / 100.0,  # Normalized pressure estimate
        }
    
    def _extract_error_patterns(self, df):
        """Detect error patterns and corrections"""
        # Backspace detection
        backspaces = len(df[df['key'].str.lower() == 'backspace']) if 'key' in df.columns else 0
        
        # Unusual key patterns (very short dwells might indicate errors)
        very_short_dwells = len(df[df['dwell_time'] < 50]) if 'dwell_time' in df.columns else 0
        
        return {
            'error_rate': backspaces / len(df) if len(df) > 0 else 0,
            'very_short_dwell_ratio': very_short_dwells / len(df) if len(df) > 0 else 0,
        }
    
    def _extract_consistency_metrics(self, dwell, flight):
        """Extract consistency and pattern metrics"""
        # Trend analysis
        if len(dwell) > 2:
            dwell_trend = np.polyfit(range(len(dwell)), dwell.values, 1)[0]
            flight_trend = np.polyfit(range(len(flight)), flight.values, 1)[0]
        else:
            dwell_trend = flight_trend = 0
            
        return {
            'dwell_trend': dwell_trend,
            'flight_trend': flight_trend,
            'pattern_consistency': 1.0 / (1.0 + dwell.std() + flight.std()),  # Inverse of variability
        }
    
    def _remove_outliers(self, series):
        """Remove outliers using IQR method"""
        if len(series) < 4:
            return series
            
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return series[(series >= lower_bound) & (series <= upper_bound)]
    
    def _get_default_features(self):
        """Return default feature set when extraction fails"""
        default_features = {}
        for feature in [
            'dwell_mean', 'dwell_std', 'dwell_median', 'dwell_min', 'dwell_max',
            'dwell_q25', 'dwell_q75', 'dwell_skew', 'dwell_kurtosis',
            'flight_mean', 'flight_std', 'flight_median', 'flight_min', 'flight_max',
            'flight_q25', 'flight_q75', 'flight_skew', 'flight_kurtosis',
            'dwell_flight_ratio', 'total_time', 'num_keystrokes',
            'dwell_cv', 'flight_cv', 'typing_speed', 'pause_ratio',
            'avg_key_pressure', 'rhythm_consistency', 'error_rate'
        ]:
            default_features[feature] = 0.0
            
        return default_features