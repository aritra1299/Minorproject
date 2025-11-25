import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
from config import Config
from utils.feature_extractor import AdvancedFeatureExtractor

class DataAnalyzer:
    def __init__(self):
        self.feature_extractor = AdvancedFeatureExtractor()
        self.logger = logging.getLogger(__name__)
    
    def analyze_dataset(self, data_dir=Config.DATA_DIR, sample_size=1000):
        """Comprehensive analysis of the keystroke dataset"""
        print("🔍 Analyzing keystroke dataset...")
        
        # Load sample data
        features_list = []
        file_count = 0
        
        for file_name in tqdm(os.listdir(data_dir)[:sample_size], desc="Processing files"):
            if file_name.startswith('sample') and file_name.endswith('.csv'):
                file_path = os.path.join(data_dir, file_name)
                try:
                    df = pd.read_csv(file_path)
                    features = self.feature_extractor.extract_comprehensive_features(df)
                    features_list.append(features)
                    file_count += 1
                except Exception as e:
                    continue
        
        if not features_list:
            print("❌ No valid data found for analysis")
            return
        
        features_df = pd.DataFrame(features_list)
        
        print(f"\n📊 Dataset Analysis Summary:")
        print(f"   Samples analyzed: {file_count}")
        print(f"   Features extracted: {len(features_df.columns)}")
        
        # Generate comprehensive reports
        self._generate_summary_statistics(features_df)
        self._plot_feature_distributions(features_df)
        self._plot_correlation_heatmap(features_df)
        self._analyze_data_quality(features_df)
        
        print(f"\n✅ Analysis complete! Check generated plots in current directory.")
    
    def _generate_summary_statistics(self, features_df):
        """Generate summary statistics for key features"""
        key_features = [
            'dwell_mean', 'dwell_std', 'flight_mean', 'flight_std',
            'typing_speed', 'dwell_flight_ratio', 'num_keystrokes'
        ]
        
        available_features = [f for f in key_features if f in features_df.columns]
        
        if not available_features:
            return
            
        summary = features_df[available_features].describe()
        print("\n📈 Key Feature Statistics:")
        print(summary.round(3))
        
        # Save to file
        summary.to_csv('dataset_summary_statistics.csv')
        print("   💾 Saved to: dataset_summary_statistics.csv")
    
    def _plot_feature_distributions(self, features_df):
        """Plot distributions of key features"""
        key_features = ['dwell_mean', 'flight_mean', 'typing_speed', 'num_keystrokes']
        available_features = [f for f in key_features if f in features_df.columns]
        
        if len(available_features) < 2:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for i, feature in enumerate(available_features[:4]):
            axes[i].hist(features_df[feature].dropna(), bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'Distribution of {feature}')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequency')
            
            # Add vertical line for mean
            mean_val = features_df[feature].mean()
            axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
            axes[i].legend()
        
        plt.tight_layout()
        plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_heatmap(self, features_df):
        """Plot correlation heatmap of features"""
        # Select numeric columns only
        numeric_df = features_df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return
            
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        plt.figure(figsize=(16, 14))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('feature_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Find highly correlated features
        high_corr = self._find_high_correlations(corr_matrix)
        if high_corr:
            print("\n🔗 Highly Correlated Features (|r| > 0.8):")
            for (f1, f2), corr in high_corr:
                print(f"   {f1} ↔ {f2}: {corr:.3f}")
    
    def _find_high_correlations(self, corr_matrix, threshold=0.8):
        """Find pairs of highly correlated features"""
        high_corr = []
        columns = corr_matrix.columns
        
        for i in range(len(columns)):
            for j in range(i+1, len(columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    high_corr.append(((columns[i], columns[j]), corr_matrix.iloc[i, j]))
        
        return high_corr
    
    def _analyze_data_quality(self, features_df):
        """Analyze data quality and identify issues"""
        print("\n🔍 Data Quality Analysis:")
        
        # Missing values
        missing_percent = (features_df.isnull().sum() / len(features_df)) * 100
        high_missing = missing_percent[missing_percent > 5]
        
        if not high_missing.empty:
            print("   ⚠️ Features with >5% missing values:")
            for feature, percent in high_missing.items():
                print(f"      {feature}: {percent:.1f}%")
        else:
            print("   ✅ No significant missing values")
        
        # Constant features
        constant_features = []
        for feature in features_df.columns:
            if features_df[feature].nunique() <= 1:
                constant_features.append(feature)
        
        if constant_features:
            print("   ⚠️ Constant features detected:")
            for feature in constant_features:
                print(f"      {feature}")
        else:
            print("   ✅ No constant features")
        
        # Extreme outliers
        outlier_features = []
        for feature in ['dwell_mean', 'flight_mean', 'typing_speed']:
            if feature in features_df.columns:
                Q1 = features_df[feature].quantile(0.25)
                Q3 = features_df[feature].quantile(0.75)
                IQR = Q3 - Q1
                outliers = features_df[(features_df[feature] < Q1 - 3*IQR) | 
                                      (features_df[feature] > Q3 + 3*IQR)]
                if len(outliers) > 0:
                    outlier_features.append((feature, len(outliers)))
        
        if outlier_features:
            print("   ⚠️ Features with extreme outliers:")
            for feature, count in outlier_features:
                print(f"      {feature}: {count} outliers")
        else:
            print("   ✅ No extreme outliers detected")

def main():
    """Main analysis function"""
    analyzer = DataAnalyzer()
    analyzer.analyze_dataset(sample_size=1000)  # Analyze first 1000 files

if __name__ == "__main__":
    main()