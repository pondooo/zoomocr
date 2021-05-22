from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from datetime import datetime
import cv2
import numpy as np

import myocr


IMAGE_DIR = './images'
app = Flask(__name__)
my_ocr = myocr.MyOCR()


@app.route('/')
def index():
    return render_template('index.html', result=my_ocr.get_result())


@app.route('/upload', methods=['POST'])
def get_paticipants():
    if request.files['image']:
        # 画像として読み込み
        stream = request.files['image'].stream
        img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
        img_cv2 = cv2.imdecode(img_array, 1)

        # 保存
        dt_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_")
        img_path = os.path.join(IMAGE_DIR, dt_now + ".png")
        cv2.imwrite(img_path, img_cv2)

        # OCR実行
        result = my_ocr.predict_pyocr(img_path)
        #result = my_ocr.predict_gcv(img_path)
        #print('result:\n', result)

        # 画像削除
        os.remove(img_path)

        # OCR結果を保存
        my_ocr.save_result()

        return redirect('/')



if __name__ == '__main__':
    app.run(debug=True)
