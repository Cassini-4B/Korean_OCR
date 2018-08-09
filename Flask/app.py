# app.py

import os
from flask import Flask, request, redirect, render_template
from flask import send_from_directory
from flask_uploads import UploadSet, configure_uploads, IMAGES
from letter_predict import letter_identification


# create a flask object
app = Flask(__name__)



# creates an association between the / page and the entry_page function (defaults to GET)
@app.route('/')
def entry_page():
    return render_template('index.html')





# creates an association between the /predict_recipe page and the render_message function
# (includes POST requests which allow users to enter in data via form)
@app.route('/korean_ocr/', methods=['GET', 'POST'])
def render_message():

    input_image = request.form['photo']
    # show user final message
    final_message = letter_identification(input_image)
    
    return render_template('ocr.html', message=final_message)



if __name__ == '__main__':
	#app.run(debug=True)
	app.run()


