import spacy

# Load the English model
nlp = spacy.load("en_core_web_sm")

# Process a sample text
text = "Barack Obama was born in Hawaii."
doc = nlp(text)

# Print out the named entities
for ent in doc.ents:
    print(f"{ent.text} -> {ent.label_}")
