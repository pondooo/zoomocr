import os
import io
from PIL import Image
import pyocr
import pyocr.builders
from google.cloud import vision

class MyOCR():
    """
    pyocrをまとめる．
    
    Attributes
    ----------
    path_tesseract: str
        tesseractへのパス．
    tool: pyocr.tools?
        OCRエンジン
    builder: pyocr.builders.TextBuilder()
        文字認識用のビルダー
    result: str
        文字認識結果
    """

    def __init__(self, path_tesseract='C:\Program Files\Tesseract-OCR'):
        """
        Parameters
        ----------
        path_tesseract: str
            tesseractへのパス．
        """
        
        # pyocrの設定
        # インストール済みのTesseractのパスを通す
        self.path_tesseract = path_tesseract
        if path_tesseract not in os.environ['PATH'].split(os.pathsep):
            os.environ['PATH'] += os.pathsep + path_tesseract
        
        # OCRエンジンの取得
        tools = pyocr.get_available_tools()
        self.tool = tools[0]

        # builderの設定？
        self.builder = pyocr.builders.TextBuilder()

        # GCVの設定
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'GCVのサービスアカウントキー(json)へのパス' # キーを記載したら下のコメントを外す
        #self.client = vision.ImageAnnotatorClient()

        # OCR結果
        self.result = ''

    
    def predict_pyocr(self, img_path, lang='jpn'):
        """
        pyocrを利用したOCRを実行する．

        Parameters
        ----------
        img_path: str
            OCRで文字認識をしたい画像のパス．
        lang: string
            認識する文字の言語．
        
        Returns
        -------
        result: str
            OCRの結果．
        """
        
        # PIL形式で呼び出し
        img = Image.open(img_path)

        # 推論
        self.result = self.tool.image_to_string(img, lang=lang, builder=self.builder)
        return self.result
    

    def predict_gcv(self, img_path):
        """
        google cloud vision を利用したOCRを実行する．
        
        Paramerters
        -----------
        img_path: str
            OCRで文字認識をしたい画像のパス．
        """

        # byte64形式で呼び出し
        with io.open(img_path, 'rb') as image_file:
            content = image_file.read()
        
        image = vision.Image(content=content)
        response = self.client.document_text_detection(image=image)
        self.result = response.text_annotations[0].description
        return self.result


    def get_result(self):
        """
        保持しているOCRの実行結果を返す．

        Returns
        -------
        result: str
            OCRの結果．
        """
        return self.result


    def save_result(self, save_path='./result.txt'):
        """
        OCRの結果をテキストファイル出力．

        Parameters
        ----------
        save_path: str
            ファイルの保存パス．
        """

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(self.result)
