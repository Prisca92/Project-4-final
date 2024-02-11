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
 return render_template('index.thml')


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
        # #marriage
        # m = request.form['blank']
        # m = m.lower()
        # if m == "yes":
        #     m = 1
        # else:
        #     m = 0
        #worktype
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
        # #residency-type
        # r = request.form['blank']
        # r = r.lower()
        # if r == "":
        #     r = 1
        # elif r == "":
        #     r = 2
        # else: 
        #     r=0
        #Cholesterol-levels
        cl = request.form['Cholesterol']
        cl = int(cl)
        cl =  ((int(cl) - 55)/(271 - 55))
        # RestingBP
        rbp = request.form['Resting_BP']
        rbp = int(rbp)
        rbp =  ((int(rbp) - 55)/(271 - 55))
        #Oldpeak
        cl = request.form['Oldpeak']
        cl = int(cl)
        cl =  ((int(cl) - 55)/(271 - 55))
        #bmi
        op = request.form['OldPeak']
        op = int(op)
        op = ((op-10.3)/(97.6-10.3))
        #smoking
        s = request.form['ST_Slope']
        if s == "unknown":
            s = 0
        elif s == "UP":
            s = 1
        elif s == "FLAT":
            s = 2
        elif s == "DOWN":
            s = 3
        else:
            s = 0
        # try to make prediction, otherwise notify user the entries are invalid
        try:
            # make prediction
            prediction = heart_pred(s,a,ea,hd,cp,rbp,cl,op,s)
            # render index_2 for result page
            return render_template('index_2.html',prediction_html=prediction)
        except ValueError:
            return "Invalid Values"
#prediction-model
def heart_pred(s,a,ea,hd,cp,rbp,cl,op):
    # opening the file- model_jlib
    m_jlib = joblib.load('model_jlib')
    # check prediction
    m_jlib.predict([[5000]])    

    #predictions
    user_input=(s,a,ea,hd,cp,rbp,cl,op,s)
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
    s,a,ea,hd,cp,rbp,cl,op,s=user_input
    s_female, s_male=(0, 0)
    if s==0: 
        s_female=1
    else: 
        s_male=1

    # m_yes, m_no=(0, 0)
    # if m==1: 
    #     m_yes=1
    # else:
    #     m_no=0

    # r_urban, r_rural = (0, 0)
    # if r==1:
    #     r_urban=1
    # elif r==2: 
    #     r_rural=1

    # s_unknown, s_never, s_former, s_yes=(0, 0, 0, 0)
    # if s==0: 
    #     s_unknown=1
    # elif s==1: 
    #     s_never=1
    # elif s==2:
    #     s_former=1
    # else: 
    #     s_yes=1

    # decoded input
    decoded_input=[a,ea,hd,cp,rbp,cl,op,s, s_female, s_male]
    return decoded_input

if __name__=='__main__': 
	app.run()

    

#if __name__ == '__main__':
    #app.run(debug=True)