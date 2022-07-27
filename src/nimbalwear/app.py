from pathlib import Path
from threading import Thread

from flask import Flask, render_template, request, redirect, url_for

from .__version__ import __version__
from .pipeline import Pipeline

PIPELINE_ROOT = Path("W:/NiMBaLWEAR")

app = Flask(__name__)
version = __version__

@app.route("/")
def home():

    studies = [x.name for x in PIPELINE_ROOT.iterdir() if x.is_dir()]

    return render_template('index.html', version=version, studies=studies)

@app.route("/study/<study>")
def study(study):

    studies = [x.name for x in PIPELINE_ROOT.iterdir() if x.is_dir()]

    study_dir = PIPELINE_ROOT / study

    pl = Pipeline(study_dir)
    collections = ["_".join(coll) for coll in pl.get_collections()]
    stages = ['convert', 'nonwear', 'crop', 'save_sensors', 'gait', 'sleep', 'activity']

    return render_template('study.html', version=version, studies=studies, study=study, collections=collections, stages=stages)

@app.route("/study/<study>/process_confirm", methods=['POST'])
def process_confirm(study):

    collections = request.form.getlist('collections')
    #collections = ','.join(collections)
    #collections = [(coll.split("_")[0], coll.split("_")[1]) for coll in collections]
    single_stage = request.form.get('single_stage')
    #single_stage = None if single_stage == 'all' else single_stage
    quiet = request.form.get('quiet')
    log = request.form.get('log')

    return render_template('process_confirm.html', version=version, study=study, collections=collections,
                           single_stage=single_stage, quiet=quiet, log=log)


@app.route("/study/<study>/process", methods=['POST'])
def process(study):

    study_dir = PIPELINE_ROOT / study

    collections = request.form.getlist('collections[]')
    collections = [(coll.split("_")[0], coll.split("_")[1]) for coll in collections]
    single_stage = request.form.get('single_stage')
    single_stage = None if single_stage == 'all' else single_stage
    quiet = request.form.get('quiet')
    log = request.form.get('log')

    # dialog box showing settings and to confirm
    pl = Pipeline(study_dir)

    plargs = {'collections': collections, 'single_stage': single_stage, 'quiet': quiet, 'log': log}

    plthread = Thread(target=pl.run, kwargs=plargs)
    plthread.start()

    # link to processing.log (

    return redirect(url_for('process_initiated', study=study, collections=collections, single_stage=single_stage,
                            quiet=quiet, log=log))

@app.route("/study/<study>/process_initiated")
def process_initiated(study):
    return render_template('process_initiated.html', version=version, study=study)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True, threaded=True)
