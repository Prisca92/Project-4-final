from flask import Flask, render_template,request
import joblib
from markupsafe import Markup
from sklearn import model_selection

app = Flask(__name__)

#set secret keys for flask application
#app.secret_key =""

#Homepage 






@app.route("/")
@app.route("/home")
def home():
 return render_template('index.html')


  #Result page  
@app.route("/output",methods =["POST","GET"])
    

#Task when recieving a POST request
def output():
    if request.method =='POST':
        #sex
        s = request.form['Sex']
        if s == "Male":
         s = 1
        elif s == "Female":
            s=0
        else :
            s =2

        #age
        a = request.form['age']
        a = int(a)
        a = ((a-0.08)/(82-0.08))
        #Excersice Angina
        ea = request.form['ExcerciseAngina']
        ea = ea.lower()
        if ea == "yes":
            ea = 1
        else:
            ea = 0
        #heart-disease
        hd= request.form['HeartDisease']
        hd = hd.lower()
        if hd == "yes":
            hd = 1
        else:
            hd = 0
        # 
        # m = request.form['blank']
        # m = m.lower()
        # if m == "yes":
        #     m = 1
        # else:
        #     m = 0
        #chestpaintype
        cp = request.form['ChestPainType']
        cp = cp.lower()
        if cp == "ASY":
            w = 0
        elif cp == "ATA":
            cp = 1
        elif cp == "NAP":
            cp = 2
        elif cp== "STA":
            cp = 3
        else:
            cp = 4
        # restingecg
        recg= request.form['RestingECG']
        if recg == "unknown":
            recg= 0
        elif recg== "LVH":
            recg = 1
        elif recg == "NORMAL":
            recg = 2
        elif recg  == "ST":
             recg= 3
        else:
             recg= 0
        #Cholesterol-levels
        cl = request.form['Cholesterol']
        cl = int(cl)
        cl =  ((int(cl) - 55)/(271 - 55))
        # RestingBP
        rbp = request.form['RestingBP']
        rbp = int(rbp)
        rbp =  ((int(rbp) - 55)/(271 - 55))
        #fastingbp
        fbs = request.form['FastingBS']
        fbs = int(fbs)
        fbs=  ((int(fbs) - 55)/(271 - 55))
        #maxhr
        mh = request.form['MaxHR']
        mh= int(mh)
        mh =  ((int(mh) - 55)/(271 - 55))
        #oldpeak
        op = request.form['OldPeak']
        op = int(op)
        op = ((op-10.3)/(97.6-10.3))
        #STslope
        st = request.form['ST_Slope']
        if st == "unknown":
            st = 0
        elif st == "UP":
            st = 1
        elif st == "FLAT":
            st = 2
        elif st == "DOWN":
            st = 3
        else:
            st = 0
        # try to make prediction, otherwise notify user the entries are invalid
        try:
            # make prediction
            prediction = heart_pred(s,a,ea,hd,cp,recg,cl,rbp,fbs,mh,op,st)
            # render index_2 for result page
            return render_template('index.html',prediction_html=prediction)
        except ValueError:
            return "Invalid Values"
#prediction-model
def heart_pred(s,a,ea,hd,cp,recg,cl,rbp,fbs,mh,op,st):
    # opening the file- model_jlib
    m_jlib = joblib.load('model_jlib')
    # check prediction
    m_jlib.predict([[5000]])    

    #predictions
    user_input=(s,a,ea,hd,cp,recg,cl,rbp,fbs,mh,op,st)
    decoded_user_input=fancy_deconstruct(user_input)
    print(decoded_user_input)
    result = model_selection.predict([decoded_user_input])
    print(result)
    #output
    if result == 1:
        return Markup("Patient is at risk of heart failure <br> Please provide recommendations for managing and reducing the risk of heart failure. Consider lifestyle modifications, preventive measures, and any other relevant advice to improve the patient's overall health and minimize the risk of heart failure.")
    elif result == 0:
        return  Markup("No risk of heart failure detected. <br> Please continue monitoring patient's health and well-being. Kindly offer recommendations to maintain their overall health, taking into account lifestyle factors, preventive measures, and any other relevant advice that can contribute to their well-being.")
    else:
        return "A problem has occured in processing your data. Try again."
# Decontruct and encode inputs, transform 10 columns to 20
# eg. female = 0, male = 1...
def fancy_deconstruct(user_input): 
    s,a,ea,hd,cp,recg,cl,rbp,fbs,mh,op,st=user_input
    s_female, s_male=(0, 0)
    if s==0: 
        s_female=1
    else: 
        s_male=1
    ea_yes, ea_no=(0, 0)
    if ea==1: 
        ea_yes=1
    else:
        ea_no=0

    cp_ASY,cp_ATA,cp_NAP, cp_TA=(0, 0, 0, 0)
    if cp==0: 
        cp_ASY=1
    elif cp==1: 
        cp_ATA=1
    elif cp==2:
        cp_NAP=1
    else: 
        cp_TA=1   

    recg_lvh, recg_normal,recg_st = (0, 0, 0)
    if recg ==0:
        recg_lvh=1
    elif recg ==1: 
        recg_normal=1
    else:
        recg_st=1

    st_down, st_flat, st_up=(0, 0, 0, 0)
    if st==0: 
        st_unknown=1
    elif st==1: 
        st_down=1
    elif st==2:
        st_flat=1
    else: 
        st_up=1

    #decoded input
    decoded_input=[a,ea_yes,ea_no,hd,cp_ASY,cp_ATA,cp_NAP,cp_TA,recg_lvh,recg_normal,recg_st,rbp,cl,op,st_down, st_flat, st_up,fbs,mh, s_female, s_male]
    return decoded_input

if __name__=='__main__': 
	app.debug = True
app.run()

    

#if __name__ == '__main__':
    #app.run(debug=True)