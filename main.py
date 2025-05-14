import tkinter as tk
from src.app import EmotionDetectionApp

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = EmotionDetectionApp(root)
        root.mainloop()  # Add parentheses
    except Exception as e:
        print(f"Error: {e}")