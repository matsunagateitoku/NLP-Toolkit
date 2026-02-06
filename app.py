# app.py
from flask import Flask, render_template, request
from text_utils import extract_named_entities, generate_wordcloud, fetch_website_text, extract_pos_tags
import logging

# Set up logging for the Flask app
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html") # note: changing to home.html breaks the app

@app.route('/ner' , methods=["GET", "POST"])
def ner():
    named_entities = None
    displacy_html = None
    
    if request.method == "POST":
        input_text = request.form.get("user_input")
        
        if input_text:
            logging.debug(f"Received input: {input_text}")
            named_entities, displacy_html = extract_named_entities(input_text)
            if named_entities is None:
                named_entities = [("Error", "Unable to process the text.")]
    
    return render_template("ner.html", named_entities=named_entities, displacy_html=displacy_html)


@app.route('/pos', methods=["GET", "POST"])
def pos():
    pos_tags = None
    pos_html = None
    error_message = None
    
    if request.method == "POST":
        input_text = request.form.get("user_input")
        
        if input_text:
            logging.debug(f"Received input for POS tagging: {input_text}")
            try:
                # extract_pos_tags returns (pos_tags_list, html)
                pos_tags, pos_html = extract_pos_tags(input_text, visualize=True)
                if pos_tags is None:
                    error_message = "Unable to process the text for POS tagging."
            except Exception as e:
                logging.exception("POS tagging failed")
                error_message = "Unable to process the text for POS tagging."
    
    return render_template("pos.html", pos_tags=pos_tags, pos_html=pos_html, error_message=error_message)


# /web route: accepts a URL from a form, fetches the webpage text
# (requires fetch_website_text function), performs named entity recognition (NER),
# and renders the results in web.html. Shows an error message if processing fails.

@app.route('/web', methods=["GET", 'POST'])
def web():
    url = request.form.get('url_input')
    text = fetch_website_text(url) if url else None
    
    if text:
        named_entities, displacy_html = extract_named_entities(text)
        return render_template('web.html', named_entities=named_entities, displacy_html=displacy_html)
    else:
        error_message = "Error fetching or processing the URL. Please check the URL and try again."
        return render_template('web.html', error_message=error_message)


@app.route('/wordcloud', methods=["GET", "POST"])
def wordcloud():
    wordcloud_image = None
    word_count = None
    error_message = None
    
    if request.method == "POST":
        input_text = request.form.get("user_input")
        url_input = request.form.get("url_input")
        max_words = request.form.get("max_words", 100)
        
        try:
            max_words = int(max_words)
        except:
            max_words = 100
        
        # Determine source: URL or text input
        text_to_process = None
        
        if url_input:
            # Fetch text from URL
            logging.debug(f"Processing URL: {url_input}")
            text_to_process = fetch_website_text(url_input)
            if text_to_process is None:
                error_message = "Error fetching content from URL. Please check the URL and try again."
        elif input_text:
            text_to_process = input_text
        
        if text_to_process:
            logging.debug(f"Generating word cloud for text of length: {len(text_to_process)}")
            
            # Generate word cloud
            wordcloud_image = generate_wordcloud(text_to_process, max_words=max_words)
            
            # Count words for stats
            word_count = len(text_to_process.split())
            
            if wordcloud_image is None:
                error_message = "Error generating word cloud. Please try again."
    
    return render_template('wc.html', 
                         wordcloud_image=wordcloud_image, 
                         word_count=word_count,
                         error_message=error_message)





@app.route('/semantic', methods=["GET"])
def semantic():
    return render_template('semantic.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/help')
def help():
    return render_template('help.html')

if __name__ == "__main__":
    logging.debug("Starting Flask app...")
    app.run(debug=True)
