from flask import render_template
from app import app
from descriptor.util import getTestImages


@app.route('/')
@app.route('/index')
def index():
    test_images=getTestImages()
    return render_template('index.html',
                           title='Home',
                           test_images=test_images)