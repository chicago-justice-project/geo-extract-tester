import flask
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt
import io
import base64

app = flask.Flask(__name__)

@app.route('/')
def main():
    return flask.render_template('main.html', error='')


@app.route('/download')
def download():
    with open('data/validation.txt') as f:
        txt = f.read()
    docs = txt.split('\n\n')
    docs = ['\n'.join([' '.join(word_row.split(' ')[:-1])
                       for word_row in doc.split('\n')])
            for doc in docs]
    txt = '\n\n'.join(docs)
    output = flask.make_response(txt)
    output.headers["Content-Disposition"] = "attachment; fname=validation.txt"
    output.headers["Content-type"] = "plain/text"
    return output

@app.route('/score', methods=['POST'])
def score():
    with open('data/validation.txt') as f:
        txt = f.read()
    docs = txt.split('\n\n')
    docs = [float(word_row.split(' ')[-1])
            for doc in docs for word_row in doc.strip().split('\n') if word_row]
    expected = np.array(docs)

    f = flask.request.files['file']
    if f:
        txt = f.read().decode('utf-8')
        docs = txt.split('\n\n')
        docs = [float(word_row.split(' ')[-1])
                for doc in docs for word_row in doc.strip().split('\n') if word_row]
        received = np.array(docs)

        perf = sklearn.metrics.roc_curve(expected, received)

        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(perf[0], perf[1])
        plt.gca().set_aspect('equal')
        plt.ylim([0, 1])
        plt.xlim([0, 1])
        plt.title('AUC = {}'.format(sklearn.metrics.roc_auc_score(expected, received)))

        img = io.BytesIO()

        plt.savefig(img, format='png')
        plt.close('all')
        img.seek(0)

        img = base64.b64encode(img.getvalue())

        return flask.render_template('performance.html', plot=str(img)[2:-1])

    return flask.render_template('main.html', error='Must upload file!')
