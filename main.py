from flask import Flask
from flask import request
from flask import render_template

import sample

text_modeling = 'chr'
model_dir = 'pretrained/model_0.ckpt'
vocab_dir = 'pretrained/vocab.pkl'
model_info = {'hidden_size':128}
prefix = u""
gw = sample.GhostWriter(text_modeling, model_dir, vocab_dir, model_info)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/', methods=['POST'])
def index_post():
    text = request.form['text']
    result = gw.sample_topk(unicode(text))
    str_result = []
    for s, p in result:
        str_result.append((s, "{:.6f}".format(p)))
    return render_template("index.html", prefix = text, result = str_result)

if __name__ == '__main__':
    app.run()
