# app.py
from flask import Flask, render_template, request
from text_utils import extract_named_entities, extract_pos_tags
import logging

# Set up logging for the Flask app
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

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
    return render_template('pos.html')

    if request.method == "POST":
        input_text = request.form.get("user_input")
        if input_text:
            logging.debug(f"Received input for POS tagging: {input_text}")
            try:
                # extract_pos_tags returns (pos_tags_list, html)
                pos_tags, pos_html = extract_pos_tags(input_text, visualize=True)
                if pos_tags is None:
                    pos_tags = [("Error", "Unable to process the text for POS tagging.")]
            except Exception:
                logging.exception("POS tagging failed")
                pos_tags = [("Error", "Unable to process the text for POS tagging.")]
    return render_template("pos.html", pos_tags=pos_tags, pos_html=pos_html)


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

@app.route('/wordcloud')
def wordcloud():
    return render_template('wordcloud.html')

@app.route('/semantic-parse', methods=["GET"])
def semantic_parse():
    return render_template('semantic_parse.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/help')
def help():
    return render_template('help.html')

if __name__ == "__main__":
    logging.debug("Starting Flask app...")
    app.run(debug=True)
