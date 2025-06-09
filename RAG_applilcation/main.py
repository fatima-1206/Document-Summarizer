import tkinter as tk
from tkinter import filedialog, messagebox
from summarizer import get_summary
from docHandling import get_text_from_file
from storage_and_retrieval import chunk_text, store_chunks, db
from queryProcessor import get_query_response

def generate_summary():
    file_path = selected_file.get()
    if not file_path:
        messagebox.showwarning("No file selected", "Please select a file first!")
        return
    
    summary = f"Summary generated for file: \n {get_summary()}"
    
    result_text.delete("1.0", tk.END) 
    result_text.insert(tk.END, summary)

def submit_query():
    query = query_text.get()
    if not query.strip():
        messagebox.showwarning("Empty query", "Please enter a query before submitting!")
        return
    
    response = f"Query Response: \n {get_query_response(query)}"
    
    # Display query response in the results box
    result_text.delete("1.0", tk.END)
    result_text.insert(tk.END, response)

def select_file():
    filetypes = [("Documents", "*.pdf *.md *.txt"), ("All files", "*.*")]
    filepath = filedialog.askopenfilename(title="Select a file", filetypes=filetypes)
    if filepath:
        selected_file.set(filepath)
        file_label.config(text=f"Selected file: {filepath}")
        text= get_text_from_file(filepath)
        chunks, metadatas = chunk_text(text, filepath)
        store_chunks(chunks, metadatas)
        messagebox.showinfo("File Selected", f"File '{filepath}' has been selected and processed.")



root = tk.Tk()
root.title("RAG Application")
root.geometry("600x600") 

query_text = tk.StringVar()
selected_file = tk.StringVar()

# Query input
tk.Label(root, text="Enter your query:").pack(pady=5)
tk.Entry(root, textvariable=query_text, width=50).pack(pady=5)
tk.Button(root, text="Submit Query", command=submit_query).pack(pady=5)

# File selection
tk.Button(root, text="Select File", command=select_file).pack(pady=5)
file_label = tk.Label(root, text="No file selected")
file_label.pack(pady=5)

# Summary generation button
tk.Button(root, text="Generate Summary", command=generate_summary).pack(pady=10)

# Results display area - multiline text box
tk.Label(root, text="Results:").pack(pady=5)
result_text = tk.Text(root, height=10, width=70)
result_text.pack(pady=5)

root.mainloop()


def on_close():
    if messagebox.askokcancel("Quit", "Do you really want to quit?"):
        db.delete_collection()  # Deletes the Chroma DB collection
        root.destroy() 

root.protocol("WM_DELETE_WINDOW", on_close) 