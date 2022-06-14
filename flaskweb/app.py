from flask import Flask, render_template,request
from PIL import Image
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
  return render_template('index.html')

@app.route('/mnist',methods=['GET','POST'])
def mnist():
  if request.method == 'GET':
    return render_template('mnist_form.html')
  else:
    f = request.files['mnistfile']
    # print(f.filename)
    path = 'data/'+f.filename
    f.save(path)
    img = Image.open(path).convert('L')
    img = np.resize(img,(1,784))
    img = 255 - img
    # print(img.shape)
    # print(img)
    f = open('model.pickle','rb')
    model = pickle.load(f)
    f.close()
    pred = model.predict(img)
    return render_template('mnist_result.html',data=pred)

if __name__ == '__main__':
  app.run(debug=True)