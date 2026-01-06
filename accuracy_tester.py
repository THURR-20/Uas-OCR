import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

class AccuracyTester:
    def __init__(self):
        self.results = []
    
    def test_dataset(self, dataset_path):
        """Test OCR dengan dataset yang memiliki ground truth"""
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        from main import OCRApplication
        ocr = OCRApplication()
        
        for item in dataset:
            image_path = item['image_path']
            ground_truth = item['text']
            
            # Extract text
            extracted = ocr.extract_text(image_path)
            
            # Calculate accuracy
            accuracy = ocr.calculate_accuracy(extracted, ground_truth)
            
            self.results.append({
                'image': image_path,
                'ground_truth': ground_truth,
                'extracted': extracted,
                **accuracy
            })
        
        return self.analyze_results()
    
    def analyze_results(self):
        """Analisis statistik hasil testing"""
        if not self.results:
            return None
        
        df = pd.DataFrame(self.results)
        
        print("=== OCR Accuracy Test Results ===")
        print(f"Total samples: {len(df)}")
        print(f"\nAverage Accuracy Metrics:")
        print(f"Character Accuracy: {df['character_accuracy'].mean():.2f}%")
        print(f"Word Accuracy: {df['word_accuracy'].mean():.2f}%")
        print(f"Similarity Score: {df['similarity_score'].mean():.2f}%")
        print(f"Overall Accuracy: {df['average_accuracy'].mean():.2f}%")
        
        # Visualize
        self.plot_results(df)
        
        return df
    
    def plot_results(self, df):
        """Plot hasil akurasi"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Histogram akurasi
        axes[0, 0].hist(df['average_accuracy'], bins=20, alpha=0.7, color='blue')
        axes[0, 0].set_title('Distribution of OCR Accuracy')
        axes[0, 0].set_xlabel('Accuracy (%)')
        axes[0, 0].set_ylabel('Frequency')
        
        # Box plot
        accuracy_metrics = ['character_accuracy', 'word_accuracy', 'similarity_score', 'average_accuracy']
        data_to_plot = [df[metric] for metric in accuracy_metrics]
        axes[0, 1].boxplot(data_to_plot, labels=[m.replace('_', '\n') for m in accuracy_metrics])
        axes[0, 1].set_title('Accuracy Metrics Comparison')
        axes[0, 1].set_ylabel('Accuracy (%)')
        
        # Scatter plot
        axes[1, 0].scatter(df['character_accuracy'], df['word_accuracy'], alpha=0.5)
        axes[1, 0].set_xlabel('Character Accuracy (%)')
        axes[1, 0].set_ylabel('Word Accuracy (%)')
        axes[1, 0].set_title('Character vs Word Accuracy')
        
        # Cumulative accuracy
        sorted_acc = np.sort(df['average_accuracy'])
        cum_acc = np.arange(1, len(sorted_acc) + 1) / len(sorted_acc)
        axes[1, 1].plot(sorted_acc, cum_acc * 100, marker='.', linestyle='none')
        axes[1, 1].set_xlabel('Accuracy Threshold (%)')
        axes[1, 1].set_ylabel('Cumulative Percentage (%)')
        axes[1, 1].set_title('Cumulative Accuracy Distribution')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()

# Contoh dataset JSON
sample_dataset = [
    {
        "image_path": "images/test1.png",
        "text": "Ini adalah contoh teks untuk testing OCR"
    },
    {
        "image_path": "images/test2.png",
        "text": "Accuracy testing is important for OCR"
    }
]

if __name__ == "__main__":
    tester = AccuracyTester()
    
    # Simpan sample dataset
    with open('test_dataset.json', 'w') as f:
        json.dump(sample_dataset, f, indent=2)
    
    print("Sample dataset created. Please add your test images and update the JSON file.")