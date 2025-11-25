// Keystroke Dynamics Authentication System
class KeystrokeAnalyzer {
    constructor() {
        this.keystrokes = [];
        this.pressTimes = {};
        this.startTime = null;
        this.isRecording = false;
        this.rhythmBars = [];
        this.visualizationBars = [];
        this.activityLog = [];
        
        this.initializeEventListeners();
        this.initializeVisualization();
        this.checkSystemStatus();
        this.updateKeystrokeCounter();
    }

    initializeEventListeners() {
        const passwordInput = document.getElementById('passwordInput');
        const passwordToggle = document.getElementById('passwordToggle');
        
        // Password visibility toggle
        passwordToggle.addEventListener('click', () => {
            const type = passwordInput.getAttribute('type') === 'password' ? 'text' : 'password';
            passwordInput.setAttribute('type', type);
            passwordToggle.innerHTML = type === 'password' ? 
                '<i class="fas fa-eye"></i>' : '<i class="fas fa-eye-slash"></i>';
        });

        // Keystroke recording events
        passwordInput.addEventListener('focus', () => this.startRecording());
        passwordInput.addEventListener('blur', () => this.stopRecording());
        passwordInput.addEventListener('keydown', (e) => this.handleKeyDown(e));
        passwordInput.addEventListener('keyup', (e) => this.handleKeyUp(e));
        passwordInput.addEventListener('input', () => this.updateKeystrokeCounter());

        // Enter key submission
        passwordInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                this.startAuthentication();
            }
        });

        // Initialize rhythm visualization
        this.initializeRhythmBars();
    }

    initializeVisualization() {
        const container = document.getElementById('typingVisualization');
        // Create 20 bars for visualization
        for (let i = 0; i < 20; i++) {
            const bar = document.createElement('div');
            bar.className = 'typing-bar';
            bar.style.left = `${i * 5}%`;
            bar.style.height = '0%';
            container.appendChild(bar);
            this.visualizationBars.push(bar);
        }
    }

    initializeRhythmBars() {
        const container = document.getElementById('typingRhythm');
        container.innerHTML = '';
        this.rhythmBars = [];
        
        // Create 10 rhythm bars
        for (let i = 0; i < 10; i++) {
            const bar = document.createElement('div');
            bar.className = 'rhythm-bar';
            bar.style.height = '5px';
            container.appendChild(bar);
            this.rhythmBars.push(bar);
        }
    }

    startRecording() {
        if (!this.isRecording) {
            this.startTime = performance.now();
            this.isRecording = true;
            this.keystrokes = [];
            this.pressTimes = {};
            this.updateActivity('Started recording keystrokes');
        }
    }

    stopRecording() {
        if (this.isRecording) {
            this.isRecording = false;
            this.updateActivity('Stopped recording keystrokes');
        }
    }

    handleKeyDown(event) {
        if (!this.isRecording) return;

        const key = event.key;
        const currentTime = performance.now() - this.startTime;

        // Ignore modifier keys and special keys
        if (key === 'Shift' || key === 'Control' || key === 'Alt' || key === 'Meta' || key === 'CapsLock') {
            return;
        }

        // Record press time
        if (!this.pressTimes[key]) {
            this.pressTimes[key] = currentTime;
        }

        // Update rhythm visualization
        this.updateRhythmBars();
    }

    handleKeyUp(event) {
        if (!this.isRecording) return;

        const key = event.key;
        const currentTime = performance.now() - this.startTime;

        if (this.pressTimes[key] !== undefined) {
            const dwell = currentTime - this.pressTimes[key];
            
            const keystrokeData = {
                key: key,
                press_time: parseFloat(this.pressTimes[key].toFixed(3)),
                release_time: parseFloat(currentTime.toFixed(3)),
                dwell_time: parseFloat(dwell.toFixed(3)),
                flight_time: 0.0
            };

            // Compute flight time for previous key
            if (this.keystrokes.length > 0) {
                const prev = this.keystrokes[this.keystrokes.length - 1];
                this.keystrokes[this.keystrokes.length - 1].flight_time = parseFloat(
                    (keystrokeData.press_time - prev.release_time).toFixed(3)
                );
            }

            this.keystrokes.push(keystrokeData);
            delete this.pressTimes[key];

            // Update live metrics
            this.updateLiveMetrics();
            this.updateVisualizationBars();
        }
    }

    updateRhythmBars() {
        // Animate rhythm bars with random heights for visual feedback
        this.rhythmBars.forEach((bar, index) => {
            const delay = index * 50;
            setTimeout(() => {
                const height = 5 + Math.random() * 50;
                bar.style.height = `${height}px`;
                
                // Reset after animation
                setTimeout(() => {
                    bar.style.height = '5px';
                }, 200);
            }, delay);
        });
    }

    updateVisualizationBars() {
        if (this.keystrokes.length === 0) return;

        // Get the last few dwell times
        const recentDwells = this.keystrokes.slice(-20).map(k => k.dwell_time);
        
        this.visualizationBars.forEach((bar, index) => {
            if (index < recentDwells.length) {
                const dwell = recentDwells[recentDwells.length - 1 - index];
                // Normalize height (assuming max dwell of 500ms)
                const height = Math.min((dwell / 500) * 100, 100);
                bar.style.height = `${height}%`;
                // Color based on dwell time
                bar.style.background = dwell > 300 ? '#dc3545' : dwell > 150 ? '#ffc107' : '#28a745';
            }
        });
    }

    updateLiveMetrics() {
        if (this.keystrokes.length === 0) return;

        const dwellTimes = this.keystrokes.map(k => k.dwell_time);
        const flightTimes = this.keystrokes.map(k => k.flight_time).filter(t => t > 0);

        const avgDwell = dwellTimes.reduce((a, b) => a + b, 0) / dwellTimes.length;
        const avgFlight = flightTimes.length > 0 ? 
            flightTimes.reduce((a, b) => a + b, 0) / flightTimes.length : 0;

        document.getElementById('liveDwellTime').textContent = `${avgDwell.toFixed(0)}ms`;
        document.getElementById('liveFlightTime').textContent = `${avgFlight.toFixed(0)}ms`;
    }

    updateKeystrokeCounter() {
        const count = this.keystrokes.length;
        document.getElementById('keystrokeCount').textContent = count;
        document.getElementById('keystrokeCountResult').textContent = count;
    }

    updateActivity(message) {
        const timestamp = new Date().toLocaleTimeString();
        this.activityLog.unshift(`${timestamp}: ${message}`);
        
        // Keep only last 5 activities
        if (this.activityLog.length > 5) {
            this.activityLog.pop();
        }

        this.updateActivityDisplay();
    }

    updateActivityDisplay() {
        const container = document.getElementById('activityLog');
        
        if (this.activityLog.length === 0) {
            container.innerHTML = '<p style="text-align: center; color: var(--gray); padding: 20px;">No recent activity</p>';
            return;
        }

        container.innerHTML = this.activityLog.map(activity => 
            `<div style="padding: 8px 0; border-bottom: 1px solid var(--border); font-size: 0.9em;">
                ${activity}
            </div>`
        ).join('');
    }

    async startAuthentication() {
        if (this.keystrokes.length === 0) {
            this.showNotification('Please type your password first!', 'warning');
            return;
        }

        if (this.keystrokes.length < 8) {
            this.showNotification('Please type at least 8 characters for accurate analysis', 'warning');
            return;
        }

        this.showLoading(true);
        this.updateActivity('Starting authentication analysis...');

        try {
            const response = await fetch("/api/verify", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    keystrokes: this.keystrokes
                })
            });

            const data = await response.json();

            if (data.status === "success") {
                this.displayResults(data.result);
                this.updateActivity('Authentication analysis completed');
            } else {
                throw new Error(data.message || 'Authentication failed');
            }

        } catch (error) {
            console.error("Authentication error:", error);
            this.showNotification(`Authentication failed: ${error.message}`, 'error');
            this.updateActivity('Authentication failed');
        } finally {
            this.showLoading(false);
        }
    }

    displayResults(result) {
        const resultsDiv = document.getElementById('results');
        const authResult = document.getElementById('authResult');
        const resultTitle = document.getElementById('resultTitle');
        const resultText = document.getElementById('resultText');

        resultsDiv.style.display = 'block';

        // Scroll to results
        resultsDiv.scrollIntoView({ behavior: 'smooth' });

        // Update main result display
        if (result.authentication_result.final_decision === 'AUTHENTICATED') {
            authResult.className = 'result-card result-authentic';
            resultTitle.innerHTML = '<i class="fas fa-check-circle"></i> Authentication Successful';
            resultText.innerHTML = `
                <strong>Identity Verified</strong><br>
                Your typing pattern matches the enrolled user with high confidence.
            `;
        } else {
            authResult.className = 'result-card result-impostor';
            resultTitle.innerHTML = '<i class="fas fa-times-circle"></i> Authentication Failed';
            resultText.innerHTML = `
                <strong>Identity Not Verified</strong><br>
                Your typing pattern does not match the enrolled user.
            `;
        }

        // Update metrics
        this.updateMetrics(result);
        
        // Update model predictions
        this.updateModelPredictions(result);
        
        // Update typing statistics
        this.updateTypingStatistics(result);
        
        // Update security analysis
        this.updateSecurityAnalysis(result);

        // Show notification
        const message = result.authentication_result.final_decision === 'AUTHENTICATED' ?
            'Authentication successful! Access granted.' :
            'Authentication failed! Access denied.';
        
        this.showNotification(message, 
            result.authentication_result.final_decision === 'AUTHENTICATED' ? 'success' : 'error');
    }

    updateMetrics(result) {
        const auth = result.authentication_result;
        const security = result.security_analysis;

        // Confidence
        document.getElementById('confidenceValue').textContent = `${auth.confidence.toFixed(1)}%`;
        document.getElementById('confidenceFill').style.width = `${auth.confidence}%`;

        // Security level with color coding
        const securityElement = document.getElementById('securityLevel');
        securityElement.textContent = security.risk_level;
        securityElement.className = `stat-value security-${security.risk_level.toLowerCase().replace('_', '-')}`;

        // Anomaly score
        document.getElementById('anomalyScore').textContent = `${security.overall_anomaly_score.toFixed(1)}%`;

        // Keystroke count
        document.getElementById('keystrokeCountResult').textContent = result.file_info.keystroke_count;
    }

    updateModelPredictions(result) {
        const container = document.getElementById('modelPredictions');
        const predictions = result.model_predictions;

        container.innerHTML = Object.entries(predictions.individual_predictions).map(([name, prediction]) => {
            const isLegitimate = prediction === 1;
            const confidence = predictions.individual_probabilities[name][1] * 100;
            const isBestModel = name === predictions.best_model;

            return `
                <div class="model-card">
                    <div class="model-name">${name}${isBestModel ? ' 🏆' : ''}</div>
                    <div class="model-status ${isLegitimate ? 'status-legitimate' : 'status-impostor'}">
                        ${isLegitimate ? 'Legitimate' : 'Impostor'}
                    </div>
                    <div class="model-confidence">${confidence.toFixed(1)}%</div>
                </div>
            `;
        }).join('');
    }

    updateTypingStatistics(result) {
        const container = document.getElementById('typingStats');
        const stats = result.typing_statistics;

        const features = [
            { name: 'Dwell Time Mean', value: `${stats.dwell_time_mean.toFixed(1)} ms` },
            { name: 'Dwell Time Std', value: `${stats.dwell_time_std.toFixed(1)} ms` },
            { name: 'Flight Time Mean', value: `${stats.flight_time_mean.toFixed(1)} ms` },
            { name: 'Flight Time Std', value: `${stats.flight_time_std.toFixed(1)} ms` },
            { name: 'Total Time', value: `${stats.total_time.toFixed(0)} ms` },
            { name: 'Typing Speed', value: `${stats.typing_speed.toFixed(1)} keys/sec` },
            { name: 'Dwell/Flight Ratio', value: stats.dwell_flight_ratio.toFixed(2) },
            { name: 'Pause Ratio', value: stats.pause_ratio.toFixed(2) }
        ];

        container.innerHTML = features.map(feature => `
            <div class="feature-item">
                <div class="feature-name">${feature.name}</div>
                <div class="feature-value">${feature.value}</div>
            </div>
        `).join('');
    }

    updateSecurityAnalysis(result) {
        const container = document.getElementById('securityAnalysis');
        const security = result.security_analysis;

        container.innerHTML = `
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">${security.feature_anomaly_score.toFixed(1)}%</div>
                    <div class="stat-label">Feature Anomaly</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${security.timing_anomaly_score.toFixed(1)}%</div>
                    <div class="stat-label">Timing Anomaly</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${security.behavior_consistency.toFixed(1)}%</div>
                    <div class="stat-label">Behavior Consistency</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${security.risk_level}</div>
                    <div class="stat-label">Risk Level</div>
                </div>
            </div>

            <div class="recommendations">
                <h4 style="margin: 15px 0 10px 0;">Security Recommendations:</h4>
                <ul class="recommendation-list">
                    ${security.recommendations.map(rec => `
                        <li class="recommendation-item">
                            <i class="fas fa-exclamation-circle recommendation-icon"></i>
                            ${rec}
                        </li>
                    `).join('')}
                </ul>
            </div>
        `;
    }

    showLoading(show) {
        const loadingDiv = document.getElementById('loading');
        const authButton = document.getElementById('authButton');
        const progressBar = document.getElementById('progressBar');

        if (show) {
            loadingDiv.style.display = 'block';
            authButton.disabled = true;
            authButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';

            // Animate progress bar
            let progress = 0;
            const interval = setInterval(() => {
                progress += Math.random() * 10;
                if (progress >= 100) {
                    progress = 100;
                    clearInterval(interval);
                }
                progressBar.style.width = `${progress}%`;
            }, 200);
        } else {
            loadingDiv.style.display = 'none';
            authButton.disabled = false;
            authButton.innerHTML = '<i class="fas fa-lock"></i> Authenticate';
            progressBar.style.width = '0%';
        }
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            background: ${type === 'success' ? '#28a745' : type === 'error' ? '#dc3545' : type === 'warning' ? '#ffc107' : '#17a2b8'};
            color: white;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 1000;
            max-width: 300px;
            animation: slideInRight 0.3s ease;
        `;

        notification.innerHTML = `
            <div style="display: flex; align-items: center; gap: 10px;">
                <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : type === 'warning' ? 'exclamation-triangle' : 'info-circle'}"></i>
                <span>${message}</span>
            </div>
        `;

        document.body.appendChild(notification);

        // Remove after 5 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.3s ease';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 5000);
    }

    clearInput() {
        document.getElementById('passwordInput').value = '';
        this.keystrokes = [];
        this.pressTimes = {};
        this.startTime = null;
        this.isRecording = false;
        
        document.getElementById('results').style.display = 'none';
        document.getElementById('keystrokeCount').textContent = '0';
        document.getElementById('keystrokeCountResult').textContent = '0';
        document.getElementById('liveDwellTime').textContent = '0ms';
        document.getElementById('liveFlightTime').textContent = '0ms';
        
        // Reset visualization bars
        this.visualizationBars.forEach(bar => {
            bar.style.height = '0%';
        });
        
        this.updateActivity('Input cleared');
        this.showNotification('Input cleared successfully', 'info');
    }

    async handleFileUpload(files) {
        if (files.length === 0) return;

        const file = files[0];
        if (!file.name.endsWith('.csv')) {
            this.showNotification('Please upload a CSV file', 'error');
            return;
        }

        this.showLoading(true);
        this.updateActivity(`Uploading file: ${file.name}`);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.status === 'success') {
                this.displayResults(data.result);
                this.updateActivity('File analysis completed');
            } else {
                throw new Error(data.message || 'File analysis failed');
            }
        } catch (error) {
            console.error('File upload error:', error);
            this.showNotification(`File analysis failed: ${error.message}`, 'error');
            this.updateActivity('File analysis failed');
        } finally {
            this.showLoading(false);
        }
    }

    async checkSystemStatus() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();

            const statusDot = document.getElementById('statusDot');
            const statusText = document.getElementById('statusText');
            const modelStatus = document.getElementById('modelStatus');

            if (data.status === 'online') {
                statusDot.className = 'status-dot status-online';
                statusText.textContent = 'System Online';
                
                if (data.model_loaded) {
                    modelStatus.textContent = 'Model Ready';
                    modelStatus.style.color = '#28a745';
                } else {
                    modelStatus.textContent = 'Model Not Loaded';
                    modelStatus.style.color = '#dc3545';
                }
            } else {
                statusDot.className = 'status-dot status-offline';
                statusText.textContent = 'System Offline';
                modelStatus.textContent = 'Unavailable';
                modelStatus.style.color = '#dc3545';
            }
        } catch (error) {
            console.error('Status check failed:', error);
            document.getElementById('statusDot').className = 'status-dot status-offline';
            document.getElementById('statusText').textContent = 'Connection Failed';
        }
    }
}

// Initialize the application when the page loads
let analyzer;

document.addEventListener('DOMContentLoaded', function() {
    analyzer = new KeystrokeAnalyzer();
    
    // Add CSS animations
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideInRight {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        
        @keyframes slideOutRight {
            from { transform: translateX(0); opacity: 1; }
            to { transform: translateX(100%); opacity: 0; }
        }
    `;
    document.head.appendChild(style);
    
    // Check system status every 30 seconds
    setInterval(() => analyzer.checkSystemStatus(), 30000);
});

// Global functions for HTML onclick handlers
function startAuthentication() {
    if (analyzer) {
        analyzer.startAuthentication();
    }
}

function clearInput() {
    if (analyzer) {
        analyzer.clearInput();
    }
}

function handleFileUpload(files) {
    if (analyzer) {
        analyzer.handleFileUpload(files);
    }
}