from flask import Flask,render_template,request,session,url_for
import os
from werkzeug.utils import secure_filename

path=os.getcwd()
#UPLOAD_FOLDER=os.path.join(path,'static/uploads')
UPLOAD_FOLDER=os.path.join('static','uploads')
ALLOWED_EXTENSIONS={'txt','pdf','png','jpg','jpeg'}
app=Flask(__name__,template_folder='Templates',static_folder='static')
app.config["UPLOAD_FOLDER"]=UPLOAD_FOLDER

app.secret_key='kkkkey'

@app.route('/')
def index():
    return render_template('index2.html')
@app.route('/',methods=['POST','GET'])
def uploadFile():
    if request.method=='POST':
        uploaded_img=request.files['uploaded-file']
        img_filename=secure_filename(uploaded_img.filename)
        uploaded_img.save(os.path.join(app.config['UPLOAD_FOLDER'],img_filename))
        session['uploaded_img_file_path']=os.path.join(app.config['UPLOAD_FOLDER'],img_filename)

        return render_template('index3.html')
@app.route('/show_image')
def displayImage():
    img_file_path=session.get('uploaded_img_file_path',None)
    #img_file_path=os.path.join(app.config['UPLOAD_FOLDER'],'uploaded_img_file_path')
    return render_template('show_image.html',user_image=img_file_path)
if __name__=='__main__':
    app.run(debug=True)