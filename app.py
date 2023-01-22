from flask import Flask,redirect,url_for,request,render_template
import numpy as np
import pickle

model=pickle.load(open('model2.pkl','rb'))
app=Flask(__name__)

@app.route('/')
def welcome():
    return render_template('diabetes.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    features=[float(x) for x in request.form.values()]
    features=np.array(features).reshape(-1,1)
    features.resize(1,6)
    ans=model.predict_proba(features)
    ans='{0:.{1}f}'.format(ans[0][1],2)
    res=""
    if(ans > str(0.5)):
        res="DIABETIC"
        return render_template('output.html',result=res)
    else:
        res="NOT DIABETIC"
        return render_template('output.html', result=res)

if __name__ == '__main__':
    app.run(debug=True)