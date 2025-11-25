import numpy as np
import pandas as pd
from scipy import stats
import logging

class SecurityAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def analyze_authentication_attempt(self, features, predictions, probabilities, model_type):
        """Comprehensive security analysis of authentication attempt"""
        analysis = {}
        
        # Basic prediction analysis
        analysis['ensemble_prediction'] = self._get_ensemble_prediction(predictions)
        analysis['confidence'] = self._calculate_confidence(probabilities, model_type)
        analysis['agreement_rate'] = self._calculate_model_agreement(predictions)
        
        # Feature-based anomaly detection
        analysis['feature_anomaly_score'] = self._detect_feature_anomalies(features)
        analysis['timing_anomaly_score'] = self._analyze_timing_patterns(features)
        analysis['consistency_score'] = self._analyze_behavior_consistency(features)
        
        # Risk assessment
        analysis['risk_level'] = self._assess_risk_level(analysis)
        analysis['security_recommendations'] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _get_ensemble_prediction(self, predictions):
        """Calculate ensemble prediction from multiple models"""
        legit_votes = sum(1 for pred in predictions.values() if pred == 1)
        total_votes = len(predictions)
        
        return 'Legitimate' if legit_votes / total_votes >= 0.5 else 'Impostor'
    
    def _calculate_confidence(self, probabilities, model_type):
        """Calculate confidence score based on model type and probabilities"""
        legit_probs = [proba[1] for proba in probabilities.values()]
        
        if model_type == 'ensemble':
            # Weighted average based on model performance
            weights = {'RandomForest': 0.3, 'LightGBM': 0.4, 'XGBoost': 0.3}
            weighted_avg = sum(prob * weights.get(name, 0.33) 
                             for name, prob in zip(probabilities.keys(), legit_probs))
            confidence = weighted_avg * 100
        else:
            confidence = np.mean(legit_probs) * 100
            
        return min(100, max(0, confidence))
    
    def _calculate_model_agreement(self, predictions):
        """Calculate how much models agree on the prediction"""
        if not predictions:
            return 0
            
        majority_vote = self._get_ensemble_prediction(predictions)
        agreed_models = sum(1 for pred in predictions.values() 
                          if (pred == 1 and majority_vote == 'Legitimate') or 
                             (pred == 0 and majority_vote == 'Impostor'))
        
        return agreed_models / len(predictions) * 100
    
    def _detect_feature_anomalies(self, features):
        """Detect anomalies based on feature values"""
        anomaly_score = 0
        anomaly_factors = []
        
        # Check dwell time anomalies
        if features.get('dwell_mean', 0) > 500:  # Too slow
            anomaly_score += 25
            anomaly_factors.append("Very slow typing")
        elif features.get('dwell_mean', 0) < 50:  # Too fast
            anomaly_score += 25
            anomaly_factors.append("Unnaturally fast typing")
            
        # Check flight time anomalies
        if features.get('flight_mean', 0) > 800:  # Too many pauses
            anomaly_score += 20
            anomaly_factors.append("Excessive pauses")
            
        # Check variability anomalies
        if features.get('dwell_std', 0) > 200:  # High inconsistency
            anomaly_score += 15
            anomaly_factors.append("High timing inconsistency")
            
        # Check ratio anomalies
        dwell_flight_ratio = features.get('dwell_flight_ratio', 0)
        if dwell_flight_ratio > 2.0 or dwell_flight_ratio < 0.1:
            anomaly_score += 20
            anomaly_factors.append("Abnormal dwell-flight ratio")
            
        return min(100, anomaly_score)
    
    def _analyze_timing_patterns(self, features):
        """Analyze timing patterns for anomalies"""
        timing_score = 0
        
        # Rhythm analysis
        rhythm_consistency = features.get('rhythm_consistency', 1)
        if rhythm_consistency > 0.5:  # High variability in rhythm
            timing_score += 30
            
        # Pause pattern analysis
        pause_ratio = features.get('pause_ratio', 0)
        if pause_ratio > 0.3:  # Too many pauses
            timing_score += 25
            
        # Speed consistency
        typing_speed = features.get('typing_speed', 0)
        if typing_speed > 15 or typing_speed < 2:  # Unrealistic typing speed
            timing_score += 25
            
        return min(100, timing_score)
    
    def _analyze_behavior_consistency(self, features):
        """Analyze behavioral consistency"""
        consistency_score = 100  # Start with perfect score
        
        # Reduce score based on inconsistencies
        dwell_std = features.get('dwell_std', 0)
        flight_std = features.get('flight_std', 0)
        
        if dwell_std > 100:
            consistency_score -= 30
        elif dwell_std > 50:
            consistency_score -= 15
            
        if flight_std > 150:
            consistency_score -= 30
        elif flight_std > 75:
            consistency_score -= 15
            
        # Check for error patterns
        error_rate = features.get('error_rate', 0)
        if error_rate > 0.1:  # High error rate
            consistency_score -= 20
            
        return max(0, consistency_score)
    
    def _assess_risk_level(self, analysis):
        """Assess overall risk level"""
        confidence = analysis['confidence']
        anomaly_score = analysis['feature_anomaly_score']
        timing_anomaly = analysis['timing_anomaly_score']
        consistency = analysis['consistency_score']
        agreement = analysis['agreement_rate']
        
        risk_score = 0
        
        # Confidence factors
        if confidence < 60:
            risk_score += 30
        elif confidence < 75:
            risk_score += 15
            
        # Anomaly factors
        risk_score += (anomaly_score + timing_anomaly) / 4
        
        # Consistency factors
        risk_score += (100 - consistency) / 3
        
        # Agreement factors
        if agreement < 70:
            risk_score += 20
            
        if risk_score >= 70:
            return "HIGH"
        elif risk_score >= 50:
            return "MEDIUM"
        elif risk_score >= 30:
            return "LOW"
        else:
            return "VERY_LOW"
    
    def _generate_recommendations(self, analysis):
        """Generate security recommendations"""
        recommendations = []
        
        if analysis['confidence'] < 75:
            recommendations.append("Consider multi-factor authentication")
            
        if analysis['feature_anomaly_score'] > 40:
            recommendations.append("Unusual typing pattern detected")
            
        if analysis['timing_anomaly_score'] > 50:
            recommendations.append("Abnormal timing patterns observed")
            
        if analysis['consistency_score'] < 70:
            recommendations.append("Inconsistent typing behavior")
            
        if analysis['agreement_rate'] < 80:
            recommendations.append("Model predictions show disagreement")
            
        if not recommendations:
            recommendations.append("Pattern appears normal")
            
        return recommendations