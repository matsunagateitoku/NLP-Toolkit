# text_utils.py
import spacy
import logging
from spacy import displacy
import requests
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Try to load the spaCy model for NER
try:
    logging.debug("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")
    logging.debug("spaCy model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading spaCy model: {e}")
    nlp = None  # If spaCy model fails to load, set nlp to None.

def fetch_website_text(url):
    """Fetch and extract text from a website."""
    try:
        logging.debug(f"Fetching URL: {url}")
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors

        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract text from the website
        text = soup.get_text(separator=' ', strip=True)
        logging.debug(f"Text extracted from website: {text[:200]}...")  # Log the first 200 characters

        return text

    except requests.RequestException as e:
        logging.error(f"Error fetching URL: {e}")
        return None

def extract_named_entities(text):
    """Extract named entities from the text using spaCy and generate displacy visualization."""
    logging.debug("Starting NER extraction...")
    
    try:
        # Check if spaCy model is loaded
        if nlp is None:
            logging.error("spaCy model is not loaded.")
            raise ValueError("spaCy model is not loaded. Please ensure the model is correctly installed.")
        
        logging.debug(f"Processing text: {text}")
        doc = nlp(text)  # Process the text
        
        # Extract named entities (entity text and their labels)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Generate the displacy HTML for the named entities
        html = displacy.render(doc, style="ent", page=True)

        logging.debug(f"Named entities extracted: {entities}")
        logging.debug(f"Displacy visualization generated.")

        return entities, html

    except Exception as e:
        # Handle errors gracefully
        logging.error(f"Error in extract_named_entities: {e}")
        return None, None
