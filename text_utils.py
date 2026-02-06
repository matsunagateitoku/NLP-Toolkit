# text_utils.py
import spacy
import logging
from spacy import displacy
from html import escape




import requests
from bs4 import BeautifulSoup

def fetch_website_text(url):
    """Fetch and extract text content from a URL."""
    logging.debug(f"Fetching content from URL: {url}")
    
    try:
        # Add headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Fetch the webpage
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes
        
        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer"]):
            script.decompose()
        
        # Get text
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        logging.debug(f"Successfully fetched {len(text)} characters from URL")
        return text
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching URL: {e}")
        return None
    except Exception as e:
        logging.error(f"Error processing webpage: {e}")
        return None


# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Try to load the spaCy model for NER and POS
try:
    logging.debug("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")
    logging.debug("spaCy model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading spaCy model: {e}")
    nlp = None  # If spaCy model fails to load, set nlp to None.

def extract_dependencies(text):
    """Extract dependency parse information from text using spaCy."""
    logging.debug("Starting dependency parsing...")
    
    try:
        if nlp is None:
            logging.error("spaCy model is not loaded.")
            raise ValueError("spaCy model is not loaded. Please ensure the model is correctly installed.")
        
        logging.debug(f"Processing text for dependency parsing: {text!r}")
        doc = nlp(text)
        
        # Extract dependency information: (token, dependency_label, head_token)
        dependencies = [(token.text, token.dep_, token.head.text) for token in doc]
        
        # Generate the displacy HTML for dependency visualization
        html = displacy.render(doc, style="dep", page=True)
        
        logging.debug(f"Dependencies extracted: {dependencies}")
        logging.debug("Displacy dependency visualization generated.")
        
        return dependencies, html
        
    except Exception as e:
        logging.error(f"Error in extract_dependencies: {e}")
        return None, None

from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import base64
from io import BytesIO

def generate_wordcloud(text, max_words=100, background_color='white'):
    """Generate a word cloud from text and return as base64 encoded image."""
    logging.debug("Starting word cloud generation...")
    
    try:
        if not text or len(text.strip()) == 0:
            logging.error("Empty text provided for word cloud")
            return None
        
        logging.debug(f"Generating word cloud for text of length: {len(text)}")
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400,
            background_color=background_color,
            max_words=max_words,
            colormap='viridis'  # Nice color scheme
        ).generate(text)
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout(pad=0)
        
        # Convert to base64 for HTML embedding
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        
        logging.debug("Word cloud generated successfully")
        return image_base64
        
    except Exception as e:
        logging.error(f"Error generating word cloud: {e}")
        return None




def extract_named_entities(text):
    """Extract named entities from the text using spaCy and generate displacy visualization."""
    logging.debug("Starting NER extraction...")
    
    try:
        # Check if spaCy model is loaded
        if nlp is None:
            logging.error("spaCy model is not loaded.")
            raise ValueError("spaCy model is not loaded. Please ensure the model is correctly installed.")
        
        logging.debug(f"Processing text for NER: {text!r}")
        doc = nlp(text)  # Process the text
        
        # Extract named entities (entity text and their labels)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Generate the displacy HTML for the named entities
        html = displacy.render(doc, style="ent", page=True)

        logging.debug(f"Named entities extracted: {entities}")
        logging.debug("Displacy visualization generated for NER.")

        return entities, html

    except Exception as e:
        # Handle errors gracefully
        logging.error(f"Error in extract_named_entities: {e}")
        return None, None

def extract_pos_tags(text, visualize=False):
    """
    Perform POS tagging on the given text.

    Returns:
      - pos_tags: list of tuples (token_text, coarse_pos, fine_grained_tag)
      - html: HTML string with color-coded POS tags under each word
      - grouped_tags: dictionary grouping words by POS category
    """
    logging.debug("Starting POS tagging...")
    try:
        if nlp is None:
            logging.error("spaCy model is not loaded.")
            raise ValueError("spaCy model is not loaded. Please ensure the model is correctly installed.")
        
        logging.debug(f"Processing text for POS tagging: {text!r}")
        doc = nlp(text)

        # Create a list of token-level POS information
        pos_tags = [(token.text, token.pos_, token.tag_) for token in doc]
        logging.debug(f"POS tags extracted: {pos_tags}")

        # Color mapping for different POS categories
        pos_colors = {
            'NOUN': '#3498db',      # Blue
            'PROPN': '#2980b9',     # Dark Blue
            'VERB': '#e74c3c',      # Red
            'ADJ': '#2ecc71',       # Green
            'ADV': '#f39c12',       # Orange
            'PRON': '#9b59b6',      # Purple
            'DET': '#1abc9c',       # Teal
            'ADP': '#34495e',       # Dark Gray
            'CONJ': '#e67e22',      # Dark Orange
            'CCONJ': '#e67e22',     # Dark Orange
            'SCONJ': '#d35400',     # Darker Orange
            'AUX': '#c0392b',       # Dark Red
            'NUM': '#16a085',       # Dark Teal
            'PART': '#7f8c8d',      # Gray
            'INTJ': '#8e44ad',      # Dark Purple
            'PUNCT': '#95a5a6',     # Light Gray
            'SYM': '#7f8c8d',       # Gray
            'X': '#bdc3c7'          # Very Light Gray
        }

        html = None
        grouped_tags = {}
        
        if visualize:
            # Create inline word display with tags underneath
            token_spans = []
            for token in doc:
                t = escape(token.text)
                pos = escape(token.pos_)
                color = pos_colors.get(token.pos_, '#95a5a6')
                
                token_spans.append(
                    f'<span style="display:inline-block;margin:8px 4px;text-align:center;vertical-align:top;">'
                    f'<div style="font-size:16px;font-weight:500;margin-bottom:4px;">{t}</div>'
                    f'<div style="font-size:11px;font-weight:600;color:{color};'
                    f'background-color:{color}22;padding:2px 6px;border-radius:3px;'
                    f'border:1px solid {color};">{pos}</div>'
                    f'</span>'
                )
            
            html = '<div style="font-family:Arial,Helvetica,sans-serif;line-height:2.5;padding:10px;">' + ''.join(token_spans) + '</div>'
        
        # Group words by POS type
        for token in doc:
            pos_type = token.pos_
            if pos_type not in grouped_tags:
                grouped_tags[pos_type] = []
            grouped_tags[pos_type].append(token.text)

        logging.debug("POS tagging completed.")
        return pos_tags, html, grouped_tags

    except Exception as e:
        logging.error(f"Error in extract_pos_tags: {e}")
        return None, None, None

    except Exception as e:
        logging.error(f"Error in extract_pos_tags: {e}")
        return None, None

    except Exception as e:
        logging.error(f"Error in extract_pos_tags: {e}")
        return None, None
