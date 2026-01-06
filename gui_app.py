import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import tkinter.ttk as ttk
from PIL import Image, ImageTk
import cv2
import pytesseract
from datetime import datetime
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class OCRGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("OCR Application - High Accuracy")
        self.root.geometry("1200x700")
        
        self.image_path = None
        self.current_image = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Frame untuk kontrol
        control_frame = tk.Frame(self.root, relief=tk.RAISED, bd=2)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Label judul
        tk.Label(control_frame, text="OCR Controls", 
                font=("Arial", 14, "bold")).pack(pady=10)
        
        # Tombol pilih gambar
        tk.Button(control_frame, text="Select Image", 
                 command=self.select_image,
                 width=20).pack(pady=5)
        
        # Pilihan bahasa
        tk.Label(control_frame, text="Language:").pack(pady=5)
        self.lang_var = tk.StringVar(value='eng')
        lang_options = ['eng', 'ind', 'eng+ind', 'spa', 'fra']
        tk.OptionMenu(control_frame, self.lang_var, *lang_options).pack()
        
        # Checkbox preprocessing
        self.preprocess_var = tk.BooleanVar(value=True)
        tk.Checkbutton(control_frame, text="Enable Preprocessing", 
                      variable=self.preprocess_var).pack(pady=5)
        
        # Tombol proses
        tk.Button(control_frame, text="Extract Text", 
                 command=self.extract_text,
                 bg="green", fg="white",
                 width=20).pack(pady=10)
        
        # Tombol batch process
        tk.Button(control_frame, text="Batch Process", 
                 command=self.batch_process).pack(pady=5)
        
        # Tombol clear
        tk.Button(control_frame, text="Clear", 
                 command=self.clear_all).pack(pady=5)
        
        # Frame utama untuk gambar dan teks
        main_frame = tk.Frame(self.root)
        main_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Frame untuk gambar
        image_frame = tk.Frame(main_frame, relief=tk.SUNKEN, bd=2)
        image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        tk.Label(image_frame, text="Image Preview", 
                font=("Arial", 12)).pack()
        
        self.image_label = tk.Label(image_frame, bg='white')
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Frame untuk teks hasil
        text_frame = tk.Frame(main_frame, relief=tk.SUNKEN, bd=2)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(text_frame, text="Extracted Text", 
                font=("Arial", 12)).pack()
        
        self.text_area = scrolledtext.ScrolledText(text_frame, 
                                                  wrap=tk.WORD,
                                                  height=10)
        self.text_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Frame untuk statistik
        stats_frame = tk.Frame(main_frame, relief=tk.SUNKEN, bd=2)
        stats_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.stats_label = tk.Label(stats_frame, text="Statistics: Not processed", 
                                   font=("Arial", 10))
        self.stats_label.pack(pady=5)
        
    def select_image(self):
        filetypes = [
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            self.image_path = filename
            self.display_image(filename)
            
    def display_image(self, image_path):
        try:
            # Baca dan resize gambar
            img = Image.open(image_path)
            img.thumbnail((600, 400))  # Resize untuk preview
            
            # Convert ke PhotoImage
            self.current_image = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.current_image)
            
        except Exception as e:
            messagebox.showerror("Error", f"Cannot load image: {e}")
            
    def extract_text(self):
        if not self.image_path:
            messagebox.showwarning("Warning", "Please select an image first!")
            return
        
        try:
            # Baca gambar
            img = cv2.imread(self.image_path)
            if self.preprocess_var.get():
                # Preprocessing
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                img = thresh
            
            # Konversi ke PIL Image
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            # Ekstrak teks
            lang = self.lang_var.get()
            text = pytesseract.image_to_string(pil_img, lang=lang)
            
            # Tampilkan teks
            self.text_area.delete(1.0, tk.END)
            self.text_area.insert(1.0, text)
            
            # Hitung statistik
            char_count = len(text)
            word_count = len(text.split())
            line_count = len(text.split('\n'))
            
            stats_text = (f"Characters: {char_count} | "
                         f"Words: {word_count} | "
                         f"Lines: {line_count} | "
                         f"Language: {lang}")
            
            self.stats_label.config(text=stats_text)
            
            # Tawarkan untuk save
            if text.strip():
                self.ask_save_text(text)
                
        except Exception as e:
            messagebox.showerror("Error", f"OCR processing failed: {e}")
            
    def ask_save_text(self, text):
        save = messagebox.askyesno("Save", "Save extracted text to file?")
        if save:
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(text)
                messagebox.showinfo("Saved", f"Text saved to {filename}")
                
    def batch_process(self):
        folder = filedialog.askdirectory(title="Select folder with images")
        if folder:
            results = []
            for filename in os.listdir(folder):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(folder, filename)
                    
                    try:
                        img = Image.open(image_path)
                        text = pytesseract.image_to_string(img, lang=self.lang_var.get())
                        
                        results.append({
                            'File': filename,
                            'Text': text[:100] + '...' if len(text) > 100 else text
                        })
                        
                    except Exception as e:
                        results.append({
                            'File': filename,
                            'Text': f'ERROR: {str(e)}'
                        })
            
            # Tampilkan hasil di window baru
            self.show_batch_results(results)
            
    def show_batch_results(self, results):
        result_window = tk.Toplevel(self.root)
        result_window.title("Batch Processing Results")
        result_window.geometry("800x600")
        
        # Treeview untuk hasil
        tree = ttk.Treeview(result_window, columns=('File', 'Text'), show='headings')
        tree.heading('File', text='Filename')
        tree.heading('Text', text='Extracted Text (first 100 chars)')
        
        for result in results:
            tree.insert('', 'end', values=(result['File'], result['Text']))
        
        tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(tree, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        
    def clear_all(self):
        self.image_path = None
        self.image_label.config(image='')
        self.text_area.delete(1.0, tk.END)
        self.stats_label.config(text="Statistics: Not processed")

def main():
    root = tk.Tk()
    app = OCRGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()