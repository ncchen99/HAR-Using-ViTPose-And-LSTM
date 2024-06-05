import os
import time

from flask import Flask
from flask import render_template, Response, request, send_from_directory, flash, url_for
from flask import current_app as app
from werkzeug.utils import secure_filename

start = time.time()

from src.lstm_vitpose import ActionClassificationLSTM
from src.video_analyzer import analyse_video, stream_video

app = Flask(__name__)
UPLOAD_FOLDER = './'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"



checkpoint_paths = [
    "models/0.ckpt",
    "models/1.ckpt",
    "models/2.ckpt",
    "models/3.ckpt",
    "models/4.ckpt",
    "models/5.ckpt",
    "models/6.ckpt"
]

class_names = [
    "蹲太低", "身體太過前傾", "擺手太低", "向後跳", "起跳不完全", "你向後甩頭了", "團身不夠緊"
]

# Load pretrained LSTM model from checkpoint file
lstm_classifiers = [ActionClassificationLSTM.load_from_checkpoint(checkpoint_path) for checkpoint_path in checkpoint_paths]
lstm_classifiers = [lstm_classifier.eval() for lstm_classifier in lstm_classifiers]

model_load_done = time.time()

print("Vitpose and LSTM model loaded in ", model_load_done - start)

class DataObject():
    pass


def checkFileType(f: str):
    return f.split('.')[-1].lower() in ['mp4', 'avi', 'mov', 'mkv']


def cleanString(v: str):
    out_str = v
    delm = ['_', '-', '.']
    for d in delm:
        out_str = out_str.split(d)
        out_str = " ".join(out_str)
    return out_str


@app.route('/', methods=['GET'])
def index():
    obj = DataObject
    obj.video = "sample_video.mp4"
    return render_template('/index.html', obj=obj)


@app.route('/upload', methods=['POST'])
def upload():
    obj = DataObject
    obj.is_video_display = False
    obj.video = ""
    print("files", request.files)
    if request.method == 'POST' and 'video' in request.files:
        video_file = request.files['video']
        if checkFileType(video_file.filename):
            filename = secure_filename(video_file.filename)
            print("filename", filename)
            # save file to /static/uploads
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print("filepath", filepath)
            video_file.save(filepath)
            obj.video = filename
            obj.is_video_display = True
            return render_template('/index.html', obj=obj)
        else:
            if video_file.filename:
                msg = f"{video_file.filename} is not a video file"
            else:
                msg = "Please select a video file"
            flash(msg)
        return render_template('/index.html', obj=obj)
    return render_template('/index.html', obj=obj)


@app.route('/sample', methods=['POST'])
def sample():
    obj = DataObject
    obj.is_video_display = True
    obj.video = "sample_video.mp4"
    return render_template('/index.html', obj=obj)


@app.route('/files/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/analyzed_files/<filename>')
def get_analyzed_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], "res_{}".format(filename), as_attachment=True)


@app.route('/result_video/<filename>')
def get_result_video(filename):
    stream = stream_video("{}res_{}".format(
        app.config['UPLOAD_FOLDER'], filename))
    return Response(stream, mimetype='multipart/x-mixed-replace; boundary=frame')


# route definition for video upload for analysis
@app.route('/analyze/<filename>')
def analyze(filename):
    # invokes method analyse_video
    return Response(analyse_video(lstm_classifiers, filename, class_names), mimetype='text/event-stream')


if __name__ == '__main__':
    app.run()
#debug=True, use_reloader=True
#host='0.0.0.0', port=80 
# running on 0.0.0.0 need to be run as root user