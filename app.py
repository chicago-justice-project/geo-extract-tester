import io
import base64

import flask
import numpy as np
import sklearn.metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

app = flask.Flask(__name__)

@app.route('/', defaults={'error': ''})
@app.route('/<error>')
def main(error):
    return flask.render_template('main.html', error=error)


@app.route('/download')
def download():
    with open('data/validation.txt', encoding='utf-8') as f:
        txt = f.read()
    docs = txt.split('\n\n')
    docs = ['\n'.join([' '.join(word_row.split(' ')[:-1])
                       for word_row in doc.split('\n')])
            for doc in docs]
    txt = '\n\n'.join(docs)
    output = flask.make_response(txt)
    output.headers["Content-Disposition"] = "attachment; filename=validation.txt"
    output.headers["Content-type"] = "plain/text"
    return output


def get_leader():
    with open('data/leader.lcsv') as f:
        name = f.readline().strip()
        auc = float(f.readline().strip())
        data = np.array([d for d in csv.reader(f.read().split('\n'))][:-1])
    return name, auc, data


def set_leader(name, auc, roc):
    with open('data/leader.lcsv', 'w') as f:
        f.write('{}\n{}\n{}\n'.format(
            name, auc, '\n'.join([','.join([str(x) for x in roc_row])
                                  for roc_row in roc])
        ))


def score_from_upload(uploaded_text):
    with open('data/validation.txt', encoding='utf-8') as f:
        txt = f.read()
    docs = txt.split('\n\n')
    docs = [float(word_row.split(' ')[-1])
            for doc in docs for word_row in doc.strip().split('\n') if word_row]
    expected = np.array(docs)

    docs = uploaded_text.split('\n\n')
    try:
        docs = [float(word_row.strip().split(' ')[-1])
                for doc in docs for word_row in doc.strip().split('\n') if word_row.strip()]
    except ValueError as e:
        return flask.redirect(
            ('/Problem with uploaded file. It likely has non-numbers'
             ' in it: ') + repr(e)
        )
    received = np.array(docs)

    roc = sklearn.metrics.roc_curve(expected, received)
    auc = sklearn.metrics.roc_auc_score(expected, received)

    return roc, auc


@app.route('/leader/')
def download_leader():
    return flask.send_from_directory('data/',
                                     'leader.lcsv',
                                     as_attachment=True,
                                     attachment_filename='leader.lcsv')


@app.route('/api/score', methods=['POST'])
def api_score():
    f = flask.request.files['file']
    try:
        _, auc = score_from_upload(f.read().decode('utf-8'))
    except Exception as e:
        return flask.jsonify({'error': repr(e)})
    return flask.jsonify({'auc': auc})


@app.route('/score', methods=['POST'])
def score():
    f = flask.request.files['file']
    if f:
        try:
            perf, your_auc = score_from_upload(f.read().decode('utf-8'))
        except ValueError as e:
            return flask.redirect(
                ('/Problem with uploaded file. ') + repr(e)
            )

        leader_name, leader_auc, leader_roc = get_leader()

        plt.plot([0, 1], [0, 1], 'k--', label='random')
        plt.plot(leader_roc[:,0], leader_roc[:,1], '--',
                 label=leader_name + ' (current leader)')
        plt.plot(perf[0], perf[1], label='your model')
        plt.gca().set_aspect('equal')
        plt.ylim([0, 1])
        plt.xlim([0, 1])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.legend()
        plt.title('Your AUC = {:.4f}, Leader AUC = {:.4f}'.format(
            your_auc, leader_auc
        ))

        img = io.BytesIO()

        plt.savefig(img, format='png')
        plt.close('all')
        img.seek(0)

        img = base64.b64encode(img.getvalue())

        if your_auc > leader_auc:
            if not flask.request.form['name']:
                error = 'You have the new high score, but I need a name to store it!'
            else:
                set_leader(flask.request.form['name'],
                           your_auc,
                           np.array([perf[0], perf[1]]).T)
                error = 'New high score!'
        else:
            error = ''

        return flask.render_template('performance.html',
                                     error=error,
                                     plot=str(img)[2:-1])

    return flask.render_template('main.html', error='Must upload file!')
