from flask import Flask, render_template, request, redirect, url_for
import glob
import os
import shutil


app = Flask(__name__)

@app.route("/")
def index():

    # 画像ファイルパスを取得
    img_path = glob.glob("static/non_labeled/*")
    print(img_path)
    # 分類するフォルダ名を取得
    category_path = [os.path.basename(path) for path in glob.glob("static/labeled/*")]
    print(category_path)
    return render_template('index.html', img=img_path[0], category_list=category_path)

@app.route("/labelling", methods=['POST'])
def labelling():

    # 分類するフォルダパスを取得
    category_path = os.path.join("static", os.path.join("labeled", request.form['category']))
    # ファイルの移動
    shutil.move(request.form['img_path'], category_path)
    return redirect("/")

if __name__ == "__main__":
    app.debug = True
    app.run()
