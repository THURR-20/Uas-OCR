import os
import sys
import cv2
import pytesseract
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Konfigurasi path Tesseract (sesuaikan dengan sistem Anda)
# Windows
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Linux/Mac (biasanya sudah di PATH)

class OCRApplication:
    def __init__(self):
        self.supported_languages = {
            'eng': 'English',
            'ind': 'Indonesian',
            'eng+ind': 'English+Indonesian'
        }
        
    def preprocess_image(self, image_path):
        """Preprocessing gambar untuk meningkatkan akurasi"""
        try:
            # Baca gambar
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Gambar tidak dapat dibaca")
            
            # Konversi ke grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Noise reduction
            denoised = cv2.medianBlur(gray, 3)
            
            # Thresholding
            _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Deskewing (rotasi otomatis)
            coords = np.column_stack(np.where(thresh > 0))
            angle = cv2.minAreaRect(coords)[-1]
            
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
                
            (h, w) = thresh.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(thresh, M, (w, h), 
                                    flags=cv2.INTER_CUBIC, 
                                    borderMode=cv2.BORDER_REPLICATE)
            
            return rotated
            
        except Exception as e:
            print(f"Error preprocessing: {e}")
            return None
    
    def extract_text(self, image_path, lang='eng', preprocess=True):
        """Ekstrak teks dari gambar"""
        try:
            if preprocess:
                processed_img = self.preprocess_image(image_path)
                if processed_img is None:
                    return ""
                text = pytesseract.image_to_string(processed_img, lang=lang)
            else:
                text = pytesseract.image_to_string(Image.open(image_path), lang=lang)
            
            return text.strip()
            
        except Exception as e:
            print(f"Error extracting text: {e}")
            return ""
    
    def extract_with_details(self, image_path, lang='eng'):
        """Ekstrak teks dengan detail confidence"""
        try:
            img = Image.open(image_path)
            data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)
            
            results = []
            for i in range(len(data['text'])):
                if data['text'][i].strip():
                    results.append({
                        'text': data['text'][i],
                        'confidence': int(data['conf'][i]),
                        'position': {
                            'x': data['left'][i],
                            'y': data['top'][i],
                            'width': data['width'][i],
                            'height': data['height'][i]
                        }
                    })
            
            return results
            
        except Exception as e:
            print(f"Error: {e}")
            return []
    
    def calculate_accuracy(self, extracted_text, ground_truth):
        """Hitung akurasi dengan berbagai metrik"""
        if not extracted_text or not ground_truth:
            return 0.0
        
        extracted = extracted_text.lower().strip()
        ground = ground_truth.lower().strip()
        
        # Character Accuracy
        correct_chars = sum(1 for a, b in zip(extracted, ground) if a == b)
        char_accuracy = (correct_chars / max(len(extracted), len(ground))) * 100
        
        # Word Accuracy
        extracted_words = set(extracted.split())
        ground_words = set(ground.split())
        common_words = extracted_words.intersection(ground_words)
        
        if ground_words:
            word_accuracy = (len(common_words) / len(ground_words)) * 100
        else:
            word_accuracy = 0
        
        # Levenshtein Distance (similarity)
        distance = self.levenshtein_distance(extracted, ground)
        max_len = max(len(extracted), len(ground))
        similarity = ((max_len - distance) / max_len) * 100 if max_len > 0 else 0
        
        return {
            'character_accuracy': round(char_accuracy, 2),
            'word_accuracy': round(word_accuracy, 2),
            'similarity_score': round(similarity, 2),
            'average_accuracy': round((char_accuracy + word_accuracy + similarity) / 3, 2)
        }
    
    def levenshtein_distance(self, s1, s2):
        """Menghitung Levenshtein distance antara dua string"""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def batch_process(self, image_folder, output_csv='results.csv', lang='eng'):
        """Proses batch multiple images"""
        results = []
        
        for filename in os.listdir(image_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_path = os.path.join(image_folder, filename)
                print(f"Processing: {filename}")
                
                text = self.extract_text(image_path, lang=lang)
                
                results.append({
                    'filename': filename,
                    'extracted_text': text,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'language': lang
                })
        
        # Save to CSV
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"Results saved to {output_csv}")
        
        return df
    
    def visualize_results(self, image_path, lang='eng'):
        """Visualisasi hasil OCR dengan bounding boxes"""
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Get OCR data with boxes
        data = pytesseract.image_to_data(gray, lang=lang, output_type=pytesseract.Output.DICT)
        
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            if int(data['conf'][i]) > 60:  # Only high confidence boxes
                (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                img = cv2.putText(img, data['text'][i], (x, y - 10), 
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        plt.figure(figsize=(15, 10))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('OCR Results with Bounding Boxes')
        plt.axis('off')
        plt.show()
        
        return img

def main():
    app = OCRApplication()
    
    while True:
        print("\n=== OCR Application ===")
        print("1. Extract text from single image")
        print("2. Batch process folder")
        print("3. Test accuracy")
        print("4. Visualize OCR results")
        print("5. Exit")
        
        choice = input("\nChoose option (1-5): ")
        
        if choice == '1':
            image_path = input("Enter image path: ")
            lang = input("Language (eng/ind/eng+ind): ") or 'eng'
            
            if os.path.exists(image_path):
                text = app.extract_text(image_path, lang=lang)
                print("\n=== Extracted Text ===")
                print(text)
                
                # Save to file
                save = input("\nSave to file? (y/n): ").lower()
                if save == 'y':
                    filename = f"extracted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(text)
                    print(f"Saved to {filename}")
            else:
                print("File not found!")
                
        elif choice == '2':
            folder_path = input("Enter folder path: ")
            if os.path.exists(folder_path):
                app.batch_process(folder_path)
            else:
                print("Folder not found!")
                
        elif choice == '3':
            image_path = input("Enter image path: ")
            ground_truth = input("Enter ground truth text: ")
            
            extracted = app.extract_text(image_path)
            accuracy = app.calculate_accuracy(extracted, ground_truth)
            
            print(f"\nExtracted Text: {extracted[:100]}...")
            print("\n=== Accuracy Metrics ===")
            for metric, value in accuracy.items():
                print(f"{metric.replace('_', ' ').title()}: {value}%")
                
        elif choice == '4':
            image_path = input("Enter image path: ")
            lang = input("Language (eng/ind/eng+ind): ") or 'eng'
            
            if os.path.exists(image_path):
                app.visualize_results(image_path, lang=lang)
            else:
                print("File not found!")
                
        elif choice == '5':
            print("Exiting...")
            break
            
        else:
            print("Invalid option!")

if __name__ == "__main__":
    main()