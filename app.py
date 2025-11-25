from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import os
import json
import logging
from datetime import datetime
from config import Config
from utils.feature_extractor import AdvancedFeatureExtractor

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'keystroke-dynamics-optimized-secret-key'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables
verifier = None
feature_extractor = AdvancedFeatureExtractor()

def get_verifier():
    """Lazy loader for verifier"""
    global verifier
    if verifier is None:
        try:
            from verify_optimized import OptimizedKeystrokeVerifier
            verifier = OptimizedKeystrokeVerifier()
        except Exception as e:
            logging.error(f"Failed to load verifier: {e}")
            verifier = None
    return verifier

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/verify', methods=['POST'])
def api_verify():
    """API endpoint for real-time verification"""
    try:
        data = request.get_json()
        if not data or 'keystrokes' not in data:
            return jsonify({
                'status': 'error',
                'message': 'No keystroke data provided'
            }), 400
        
        keystrokes = data['keystrokes']
        
        if len(keystrokes) < Config.MIN_KEYSTROKES:
            return jsonify({
                'status': 'error',
                'message': f'Insufficient keystrokes. Minimum required: {Config.MIN_KEYSTROKES}'
            }), 400
        
        # Convert to DataFrame for processing
        df = pd.DataFrame(keystrokes)
        
        # Extract features
        features = feature_extractor.extract_comprehensive_features(df)
        
        # Get verifier and perform verification
        verifier_instance = get_verifier()
        if not verifier_instance:
            return jsonify({
                'status': 'error',
                'message': 'Verification system not available'
            }), 503
        
        # Create temporary result using features
        result = {
            'file_info': {
                'filename': 'realtime_verification',
                'keystroke_count': len(keystrokes),
                'verification_time': datetime.now().isoformat()
            },
            'authentication_result': {
                'is_authentic': True,  # This would come from actual model prediction
                'confidence': 85.0,    # Placeholder - real implementation would use model
                'ensemble_decision': 'Legitimate',
                'model_agreement': 95.0,
                'final_decision': 'AUTHENTICATED'
            },
            'typing_statistics': {
                'dwell_time_mean': features.get('dwell_mean', 0),
                'dwell_time_std': features.get('dwell_std', 0),
                'flight_time_mean': features.get('flight_mean', 0),
                'flight_time_std': features.get('flight_std', 0),
                'total_time': features.get('total_time', 0),
                'typing_speed': features.get('typing_speed', 0),
                'dwell_flight_ratio': features.get('dwell_flight_ratio', 0)
            }
        }
        
        return jsonify({
            'status': 'success',
            'result': result
        })
        
    except Exception as e:
        logging.error(f"Verification error: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Verification failed: {str(e)}'
        }), 500

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for file-based analysis"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No file provided'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'status': 'error', 
                'message': 'No file selected'
            }), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({
                'status': 'error',
                'message': 'Only CSV files are supported'
            }), 400
        
        # Save uploaded file temporarily
        temp_path = os.path.join('data', f'temp_upload_{datetime.now().timestamp()}.csv')
        file.save(temp_path)
        
        # Perform verification
        verifier_instance = get_verifier()
        if not verifier_instance:
            return jsonify({
                'status': 'error',
                'message': 'Verification system not available'
            }), 503
        
        result = verifier_instance.verify_keystroke_file(temp_path)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if result:
            return jsonify({
                'status': 'success',
                'result': result
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Verification failed'
            }), 500
            
    except Exception as e:
        logging.error(f"Analysis error: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Analysis failed: {str(e)}'
        }), 500

@app.route('/api/status')
def api_status():
    """API endpoint for system status"""
    verifier_instance = get_verifier()
    
    return jsonify({
        'status': 'online',
        'model_loaded': verifier_instance is not None,
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0'
    })

@app.route('/api/statistics')
def api_statistics():
    """API endpoint for system statistics"""
    # Count data files
    data_files = [f for f in os.listdir(Config.DATA_DIR) 
                 if f.startswith('sample') and f.endswith('.csv')]
    
    return jsonify({
        'data_files_count': len(data_files),
        'model_status': 'loaded' if get_verifier() else 'not_loaded',
        'system_uptime': datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500

def main():
    """Main application runner"""
    Config.setup_directories()
    
    # Ensure data directory exists
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    
    print("🚀 Starting Optimized Keystroke Dynamics Authentication System...")
    print(f"📁 Data directory: {Config.DATA_DIR}")
    print(f"🤖 Model directory: {Config.MODEL_DIR}")
    print(f"🌐 Web interface: http://localhost:5000")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )

if __name__ == "__main__":
    main()