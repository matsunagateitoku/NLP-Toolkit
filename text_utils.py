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

# def extract_pos_tags(text, visualize=False):
#     """
#     Perform POS tagging on the given text.

#     Returns:
#       - pos_tags: list of tuples (token_text, coarse_pos, fine_grained_tag)
#       - html: If visualize==True, an HTML string representing tokens + POS; otherwise None

#     visualize:
#       - If False (default), function returns only the pos_tags list and html is None.
#       - If True, function will attempt to generate an HTML visualization. It prefers to
#         use spaCy's displaCy dependency visualizer (style='dep') for a richer visual,
#         falling back to a simple inline HTML token/POS representation if that fails.
#     """
#     logging.debug("Starting POS tagging...")
#     try:
#         if nlp is None:
#             logging.error("spaCy model is not loaded.")
#             raise ValueError("spaCy model is not loaded. Please ensure the model is correctly installed.")
        
#         logging.debug(f"Processing text for POS tagging: {text!r}")
#         doc = nlp(text)

#         # Create a list of token-level POS information
#         pos_tags = [(token.text, token.pos_, token.tag_) for token in doc]
#         logging.debug(f"POS tags extracted: {pos_tags}")

#         html = None
#         if visualize:
#             try:
#                 # Try rendering a dependency visualization (useful to inspect token relations + POS)
#                 logging.debug("Generating displacy dependency visualization for POS.")
#                 # displacy 'dep' includes tokens and their POS/dependency; returns a full HTML page when page=True
#                 html = displacy.render(doc, style="dep", page=True)
#             except Exception as e:
#                 # Fallback: produce a simple inline HTML listing tokens with POS tags
#                 logging.warning(f"displacy dependency render failed, falling back to simple HTML. Error: {e}")
#                 token_spans = []
#                 for token in doc:
#                     # Escape token text to avoid HTML injection
#                     t = escape(token.text)
#                     token_spans.append(
#                         f'<span style="display:inline-block;margin:6px;padding:4px;border-radius:4px;'
#                         f'background:#f2f2f2;border:1px solid #ddd;">'
#                         f'<strong>{t}</strong><br/><small>{escape(token.pos_)} ({escape(token.tag_)})</small>'
#                         f'</span>'
#                     )
#                 html = '<div style="font-family:Arial,Helvetica,sans-serif;">' + ''.join(token_spans) + '</div>'

#         logging.debug("POS tagging completed.")
#         return pos_tags, html


    except Exception as e:
        logging.error(f"Error in extract_pos_tags: {e}")
        return None, None
