import os
import pdfplumber
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import markdown
from bs4 import BeautifulSoup
import re
import unicodedata

file_path = ""

# get file extension
def get_file_extension(file_path):
    return os.path.splitext(file_path)[-1].lower()

# use the appropriate text parser based on file extension (pdf, markdown, txt)
def parse_file(file_path):
    file_extension = get_file_extension(file_path)
    
    if file_extension == '.pdf':
        with pdfplumber.open(file_path) as pdf:
            text = ''
            for page in pdf.pages:
                text += page.extract_text() + '\n'
            return text
    elif file_extension == '.md' or file_extension == '.markdown':
        with open(file_path, "r", encoding="utf-8") as file:
            html = markdown.markdown(file.read())
        return BeautifulSoup(html, "html.parser").get_text()
    elif file_extension == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

def clean_text(text):
    # Normalize line breaks and spaces
    text = re.sub(r'\r\n|\r', '\n', text)           # Convert \r\n or \r to \n
    text = re.sub(r'\n{2,}', '\n\n', text)          # Collapse many newlines into 2
    text = re.sub(r'[ \t]+', ' ', text)             # Remove extra spaces/tabs

    # Normalize unicode 
    text = unicodedata.normalize("NFKD", text)
    def add_period_to_bullet(match):
        line = match.group(0).strip()
        if not line.endswith('.'):
            return line + '.'
        return line

    # Add periods to lines that start with bullet markers (before removing markers)
    text = re.sub(r'(?m)^\s*[-*+]\s+(.*)', lambda m: "- " + add_period_to_bullet(m), text)
    # Remove common bullet points
    text = re.sub(
        r'[\u2022\u2023\u25E6\u2043\u2219\u25AA\u25AB\u25CB\u25CF\u25A0\u25B8\u29BE\u29BF]',
          '', text)

    # Remove markdown or ASCII-style tables
    text = re.sub(r'\|.*?\|', '', text)      # Remove markdown tables
    text = re.sub(r'[-=]{3,}', '', text)     # Remove underlines in tables
    text = re.sub(r'^\s*[\-\*+]\s+', '', text, flags=re.MULTILINE)  # Bulleted list lines

    # Remove figure/table/image captions
    text = re.sub(r'(Figure|Table|Image|Chart|Diagram)\s*\d+[\.:]?', '', text, flags=re.IGNORECASE)

    # Remove bracketed footnotes like [1], [12], (Fig. 3), etc.
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\(.*?fig.*?\)', '', text, flags=re.IGNORECASE)

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Fix line breaks and hyphens split across lines
    text = re.sub(r'-\n', '', text)  # Remove hyphenated line-breaks
    text = re.sub(r'\n+', '\n', text)  # Collapse newlines
    text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces

    # Strip remaining non-ASCII or odd symbols
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # before every \n add a period if it doesn't end with one special character
    text = re.sub(r'(?<![.!?:])\n', '. \n', text)     

    return text.strip()

def get_text_from_file(file_path_):
    file_path = file_path_
    raw_text = parse_file(file_path)
    # Clean the raw text
    cleaned_text = clean_text(raw_text)
    print("Doc parsed.....")
    return cleaned_text

