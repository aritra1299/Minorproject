import pandas as pd
import numpy as np
import os
import sys
import json
import logging
from datetime import datetime
import joblib

from config import Config
from utils.feature_extractor import AdvancedFeatureExtractor
from utils.security_analyzer import SecurityAnalyzer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedKeystrokeVerifier:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(Config.MODEL_DIR, 'keystroke_ensemble_model.joblib')
        
        self.model_path = model_path
        self.load_model()
        self.security_analyzer = SecurityAnalyzer()
        
    def load_model(self):
        """Load the trained ensemble model"""
        if not os.path.exists(self.model_path):
            logger.error(f"Model file not found: {self.model_path}")
            logger.info("\nPlease train the model first by running:")
            logger.info("python train_optimized.py")
            sys.exit(1)
        
        try:
            model_data = joblib.load(self.model_path)
            
            self.models = model_data['models']
            self.best_model = model_data['best_model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.feature_extractor = model_data['feature_extractor']
            self.training_info = model_data.get('training_info', {})
            
            logger.info("✅ Model loaded successfully")
            logger.info(f"   Best model: {self.best_model}")
            logger.info(f"   Training samples: {self.training_info.get('num_samples', 'Unknown')}")
            logger.info(f"   Feature count: {len(self.feature_names)}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            sys.exit(1)
    
    def verify_keystroke_file(self, csv_file_path):
        """Verify a keystroke CSV file with comprehensive analysis"""
        logger.info(f"🔍 Verifying: {os.path.basename(csv_file_path)}")
        
        try:
            # Load and validate data
            df = self._load_keystroke_data(csv_file_path)
            if df is None:
                return None
            
            # Extract features
            features = self.feature_extractor.extract_comprehensive_features(df)
            
            # Get predictions from all models
            predictions, probabilities = self._get_model_predictions(features)
            
            # Perform security analysis
            security_analysis = self.security_analyzer.analyze_authentication_attempt(
                features, predictions, probabilities, 'ensemble'
            )
            
            # Compile final result
            result = self._compile_verification_result(
                csv_file_path, df, features, predictions, probabilities, security_analysis
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Verification failed for {csv_file_path}: {e}")
            return None
    
    def _load_keystroke_data(self, file_path):
        """Load and validate keystroke data"""
        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip().str.lower()
            
            # Validate data
            if len(df) < Config.MIN_KEYSTROKES:
                logger.warning(f"Insufficient keystrokes: {len(df)} (minimum: {Config.MIN_KEYSTROKES})")
                return None
                
            if 'dwell_time' not in df.columns or 'flight_time' not in df.columns:
                logger.warning("Missing required columns: dwell_time or flight_time")
                return None
                
            logger.info(f"✅ Loaded {len(df)} keystrokes")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return None
    
    def _get_model_predictions(self, features):
        """Get predictions from all models"""
        # Create feature vector
        feature_vector = pd.DataFrame([features])
        
        # Ensure all expected features are present
        for feature in self.feature_names:
            if feature not in feature_vector.columns:
                feature_vector[feature] = 0.0
        
        feature_vector = feature_vector[self.feature_names]
        feature_vector = feature_vector.fillna(0)
        
        # Scale features
        X_scaled = self.scaler.transform(feature_vector)
        
        # Get predictions from all models
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X_scaled)[0]
                proba = model.predict_proba(X_scaled)[0]
                
                predictions[name] = pred
                probabilities[name] = proba
                
            except Exception as e:
                logger.warning(f"Prediction failed for {name}: {e}")
                predictions[name] = 0  # Default to impostor on error
                probabilities[name] = [1.0, 0.0]  # [impostor_prob, legitimate_prob]
        
        return predictions, probabilities
    
    def _compile_verification_result(self, file_path, df, features, predictions, 
                                   probabilities, security_analysis):
        """Compile comprehensive verification result"""
        # Determine final decision
        is_authentic = security_analysis['ensemble_prediction'] == 'Legitimate'
        confidence = security_analysis['confidence']
        
        # Calculate overall anomaly score
        overall_anomaly = (security_analysis['feature_anomaly_score'] + 
                          security_analysis['timing_anomaly_score']) / 2
        
        result = {
            'file_info': {
                'filename': os.path.basename(file_path),
                'keystroke_count': len(df),
                'verification_time': datetime.now().isoformat()
            },
            'authentication_result': {
                'is_authentic': is_authentic,
                'confidence': confidence,
                'ensemble_decision': security_analysis['ensemble_prediction'],
                'model_agreement': security_analysis['agreement_rate'],
                'final_decision': 'AUTHENTICATED' if is_authentic and confidence >= Config.CONFIDENCE_THRESHOLD else 'REJECTED'
            },
            'security_analysis': {
                'risk_level': security_analysis['risk_level'],
                'feature_anomaly_score': security_analysis['feature_anomaly_score'],
                'timing_anomaly_score': security_analysis['timing_anomaly_score'],
                'overall_anomaly_score': overall_anomaly,
                'behavior_consistency': security_analysis['consistency_score'],
                'recommendations': security_analysis['security_recommendations']
            },
            'model_predictions': {
                'individual_predictions': predictions,
                'individual_probabilities': {
                    name: [float(p[0]), float(p[1])] for name, p in probabilities.items()
                },
                'best_model': self.best_model,
                'best_model_prediction': predictions.get(self.best_model, 0),
                'best_model_confidence': max(probabilities.get(self.best_model, [0, 0])) * 100
            },
            'typing_statistics': {
                'dwell_time_mean': features.get('dwell_mean', 0),
                'dwell_time_std': features.get('dwell_std', 0),
                'flight_time_mean': features.get('flight_mean', 0),
                'flight_time_std': features.get('flight_std', 0),
                'total_time': features.get('total_time', 0),
                'typing_speed': features.get('typing_speed', 0),
                'dwell_flight_ratio': features.get('dwell_flight_ratio', 0),
                'pause_ratio': features.get('pause_ratio', 0),
                'error_rate': features.get('error_rate', 0)
            }
        }
        
        return result
    
    def print_detailed_results(self, result):
        """Print comprehensive verification results"""
        if not result:
            print("❌ No results to display")
            return
        
        print("\n" + "="*80)
        print("🔐 ADVANCED KEYSTROKE DYNAMICS VERIFICATION RESULTS")
        print("="*80)
        
        # File information
        print(f"📁 FILE: {result['file_info']['filename']}")
        print(f"⌨️  KEYSTROKES: {result['file_info']['keystroke_count']}")
        print(f"🕒 TIME: {result['file_info']['verification_time']}")
        
        print("\n" + "─" * 80)
        
        # Authentication Result
        auth = result['authentication_result']
        print("🎯 AUTHENTICATION RESULT:")
        print(f"   Status: {auth['final_decision']}")
        print(f"   Confidence: {auth['confidence']:.2f}%")
        print(f"   Ensemble Decision: {auth['ensemble_decision']}")
        print(f"   Model Agreement: {auth['model_agreement']:.1f}%")
        
        print("\n" + "─" * 80)
        
        # Security Analysis
        security = result['security_analysis']
        print("🛡️ SECURITY ANALYSIS:")
        print(f"   Risk Level: {security['risk_level']}")
        print(f"   Feature Anomaly: {security['feature_anomaly_score']:.1f}%")
        print(f"   Timing Anomaly: {security['timing_anomaly_score']:.1f}%")
        print(f"   Overall Anomaly: {security['overall_anomaly_score']:.1f}%")
        print(f"   Behavior Consistency: {security['behavior_consistency']:.1f}%")
        
        print("\n   📋 Recommendations:")
        for rec in security['recommendations']:
            print(f"      • {rec}")
        
        print("\n" + "─" * 80)
        
        # Model Predictions
        models = result['model_predictions']
        print("🤖 MODEL PREDICTIONS:")
        for name, prediction in models['individual_predictions'].items():
            status = "✅ Legitimate" if prediction == 1 else "❌ Impostor"
            confidence = max(models['individual_probabilities'][name]) * 100
            best_indicator = " 🏆" if name == models['best_model'] else ""
            print(f"   {name:15}: {status} ({confidence:.1f}%){best_indicator}")
        
        print("\n" + "─" * 80)
        
        # Typing Statistics
        stats = result['typing_statistics']
        print("📊 TYPING STATISTICS:")
        print(f"   Dwell Time: {stats['dwell_time_mean']:.1f} ± {stats['dwell_time_std']:.1f} ms")
        print(f"   Flight Time: {stats['flight_time_mean']:.1f} ± {stats['flight_time_std']:.1f} ms")
        print(f"   Total Time: {stats['total_time']:.1f} ms")
        print(f"   Typing Speed: {stats['typing_speed']:.1f} keys/sec")
        print(f"   Dwell/Flight Ratio: {stats['dwell_flight_ratio']:.2f}")
        print(f"   Pause Ratio: {stats['pause_ratio']:.2f}")
        print(f"   Error Rate: {stats['error_rate']:.2f}")
        
        print("\n" + "─" * 80)
        
        # Final Decision with Emoji
        if auth['final_decision'] == 'AUTHENTICATED':
            print("🎉 FINAL DECISION: ✅ AUTHENTICATED - Access Granted")
        else:
            print("🚫 FINAL DECISION: ❌ REJECTED - Access Denied")
            
            # Provide reasons for rejection
            rejection_reasons = []
            if auth['confidence'] < Config.CONFIDENCE_THRESHOLD:
                rejection_reasons.append(f"Low confidence ({auth['confidence']:.1f}% < {Config.CONFIDENCE_THRESHOLD}%)")
            if security['overall_anomaly_score'] > Config.ANOMALY_THRESHOLD:
                rejection_reasons.append(f"High anomaly score ({security['overall_anomaly_score']:.1f}% > {Config.ANOMALY_THRESHOLD}%)")
            if auth['model_agreement'] < 70:
                rejection_reasons.append(f"Low model agreement ({auth['model_agreement']:.1f}%)")
                
            if rejection_reasons:
                print("   📝 Reasons:")
                for reason in rejection_reasons:
                    print(f"      • {reason}")
        
        print("="*80)

def main():
    """Main verification function"""
    # Test with your sample file
    sample_file = 'data/sample5109.csv'
    
    if not os.path.exists(sample_file):
        print(f"❌ Sample file not found: {sample_file}")
        print("Please ensure your data files are in the 'data/' directory")
        return 1
    
    try:
        verifier = OptimizedKeystrokeVerifier()
        result = verifier.verify_keystroke_file(sample_file)
        
        if result:
            verifier.print_detailed_results(result)
            
            # Save detailed results
            output_file = f"verification_result_{os.path.basename(sample_file).split('.')[0]}.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            print(f"\n💾 Detailed results saved to: {output_file}")
        else:
            print("❌ Verification failed - no result returned")
            return 1
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())