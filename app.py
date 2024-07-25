# main.py
import tkinter as tk
from tkinter import simpledialog, messagebox
from manga_recommendation_content_filter import get_recommendations_from_prompt

def get_manga_recommendations():
    prompt = simpledialog.askstring("Input", "Enter a brief description or a title of a manga you like:")
    if prompt:
        recommendations = get_recommendations_from_prompt(prompt)
        recommendation_text = "\n".join([f"{i+1}. {title}" for i, title in enumerate(recommendations)])
        messagebox.showinfo("Manga Recommendations", recommendation_text)

# Create the main application window
root = tk.Tk()
root.title("Manga Recommendation System")
root.geometry("400x200")

# Create and place a button to get recommendations
recommend_button = tk.Button(root, text="Get Manga Recommendations", command=get_manga_recommendations)
recommend_button.pack(pady=50)

# Run the application
root.mainloop()
