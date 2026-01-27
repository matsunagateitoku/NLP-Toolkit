# text_utils.py
import spacy
import logging
from spacy import displacy
from html import escape


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
"""
def extract_pos_tags(text, visualize=False):
    """
    Perform POS tagging on the given text.

    Returns:
      - pos_tags: list of tuples (token_text, coarse_pos, fine_grained_tag)
      - html: If visualize==True, an HTML string representing tokens + POS; otherwise None

    visualize:
      - If False (default), function returns only the pos_tags list and html is None.
      - If True, function will attempt to generate an HTML visualization. It prefers to
        use spaCy's displaCy dependency visualizer (style='dep') for a richer visual,
        falling back to a simple inline HTML token/POS representation if that fails.
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

        html = None
        if visualize:
            try:
                # Try rendering a dependency visualization (useful to inspect token relations + POS)
                logging.debug("Generating displacy dependency visualization for POS.")
                # displacy 'dep' includes tokens and their POS/dependency; returns a full HTML page when page=True
                html = displacy.render(doc, style="dep", page=True)
            except Exception as e:
                # Fallback: produce a simple inline HTML listing tokens with POS tags
                logging.warning(f"displacy dependency render failed, falling back to simple HTML. Error: {e}")
                token_spans = []
                for token in doc:
                    # Escape token text to avoid HTML injection
                    t = escape(token.text)
                    token_spans.append(
                        f'<span style="display:inline-block;margin:6px;padding:4px;border-radius:4px;'
                        f'background:#f2f2f2;border:1px solid #ddd;">'
                        f'<strong>{t}</strong><br/><small>{escape(token.pos_)} ({escape(token.tag_)})</small>'
                        f'</span>'
                    )
                html = '<div style="font-family:Arial,Helvetica,sans-serif;">' + ''.join(token_spans) + '</div>'

        logging.debug("POS tagging completed.")
        return pos_tags, html
"""


    except Exception as e:
        logging.error(f"Error in extract_pos_tags: {e}")
        return None, None
