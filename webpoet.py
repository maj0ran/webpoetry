from flask import Flask
from webpoetry.poem_ai import generate_text, get_poem_generator, stylize

poet = Flask(__name__)

@poet.route("/")
def generate_poem():
    poem_gen = get_poem_generator()
    poem = generate_text(
                    poem_gen,
                    start_string="E")

    html = stylize(poem)

    return html
