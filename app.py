from flask import Flask, render_template, request
try:
    from text_utils import (
        extract_named_entities,
        extract_pos_tags,
        fetch_website_text
    )
except ImportError as e:
    print("IMPORT ERROR:", e)
    fetch_website_text = None


import logging

# Logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = Flask(__name__)

# --------------------
# Home
# --------------------
@app.route("/")
def home():
    return render_template("home.html")


# --------------------
# NER
# --------------------
@app.route("/ner", methods=["GET", "POST"])
def ner():
    named_entities = None
    displacy_html = None

    if request.method == "POST":
        input_text = request.form.get("user_input")
        if input_text:
            logging.debug(f"NER input: {input_text}")
            named_entities, displacy_html = extract_named_entities(input_text)

    return render_template(
        "ner.html",
        named_entities=named_entities,
        displacy_html=displacy_html
    )


# --------------------
# POS Tagging
# --------------------
@app.route("/pos", methods=["GET", "POST"])
def pos():
    pos_tags = None
    pos_html = None

    if request.method == "POST":
        input_text = request.form.get("user_input")
        if input_text:
            logging.debug(f"POS input: {input_text}")
            try:
                pos_tags, pos_html = extract_pos_tags(
                    input_text, visualize=True
                )
            except Exception:
                logging.exception("POS tagging failed")
                pos_tags = [("Error", "Unable to process text.")]

    return render_template(
        "pos.html",
        pos_tags=pos_tags,
        pos_html=pos_html
    )


# --------------------
# Web / Semantic Parse
# --------------------
@app.route("/web", methods=["GET", "POST"])
def semantic_parse():
    named_entities = None
    displacy_html = None
    error_message = None

    if request.method == "POST":
        url = request.form.get("url_input")
        if url:
            try:
                text = fetch_website_text(url)
                named_entities, displacy_html = extract_named_entities(text)
            except Exception:
                logging.exception("Web processing failed")
                error_message = "Error fetching or processing the URL."

    return render_template(
        "semantic_parse.html",
        named_entities=named_entities,
        displacy_html=displacy_html,
        error_message=error_message
    )


# --------------------
# Wordcloud
# --------------------
@app.route("/wordcloud")
def wordcloud():
    return render_template("wordcloud.html")


# --------------------
# Static Pages
# --------------------
@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/help")
def help_page():
    return render_template("help.html")


# --------------------
# Run app
# --------------------
if __name__ == "__main__":
    logging.debug("Starting Flask app...")
    app.run(debug=True)
