from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename

import process as proc
import globalvar as gv

from markupsafe import Markup
from datetime import datetime


app = Flask(__name__)

@app.route('/')
def main():
    gv.OUTPUT = 'Not Recognized'
    page_html = 'main.html'

    return render_template(page_html)

@app.route('/result')
def result():
    page_html = 'main.html'

    return render_template(page_html, result=1, output=gv.OUTPUT)


@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    gv.OUTPUT = 'Not Recognized'
    if request.method == 'POST':

        f = request.files['audio_data']
        gv.temp_file = secure_filename(f.filename)
        f.save(gv.temp_file)

        gv.OUTPUT = proc.match_audio(gv.temp_file)

        if gv.OUTPUT == '':
            gv.OUTPUT = 'Not Recognized'

    return 'tes'



if __name__ == '__main__':
    app.run(port=9999)