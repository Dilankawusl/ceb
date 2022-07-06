
from re import T
import streamlit as st
import pickle
import tensorflow as tf
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import save_model
from datetime import date
import datetime



file = r'finaldataf.csv'
df = pd.read_csv(file, parse_dates=[0])

df.index = df['Date']


def main():
  from PIL import Image
  st.title("CEB TRANSFORMER MAINTENANCE")
  image1 = Image.open('trans.jpg')
  st.image(image1,use_column_width='always')
  
  html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">INPUT DETAILS</h2>
    </div>
    """
  st.info('''Hello!!!!! This is a platform which developed  to choose the best day to service 
  the transformers of interconnected grid substaions without overloading power system.
  Please input the required details and Get the Results :) Have a nice day !
     ''')

  
  image = Image.open('ceblogo.png')
  with st.form('Form1'):
   st.sidebar.image(image)
   st.sidebar.title("INPUT DATA")
   st.sidebar.markdown('Maintance Transformer')
   activities=['Thulhiriya','Veyangoda','Kurunagala','Kegalle']
   option=st.sidebar.selectbox('Select the Grid Substation to maintance',activities)

   if st.sidebar.markdown(''):
      if option=='Thulhiriya':
        grid=1
        option1 = st.sidebar.selectbox('Select the Transformer',('Transformer 1', 'Transformer 2', 'Transformer 3'))
        if(option1=='Transformer 1'):
                    Transformer = 1
        elif(option1=='Transformer 2'):
                    Transformer = 2
        else:
                   Transformer = 3

      elif option=='Veyangoda':
        grid=2
        option1 = st.sidebar.selectbox('Select the Transformer',('Transformer 1', 'Transformer 2', 'Transformer 3'))
        if(option1=='Transformer 1'):
                    Transformer = 1
        elif(option1=='Transformer 2'):
                    Transformer = 2
        else:
                   Transformer = 3

      elif option=='Kurunagala':
        grid=3
        option1 = st.sidebar.selectbox('Select the Transformer',('Transformer 1', 'Transformer 2', 'Transformer 3'))
        if(option1=='Transformer 1'):
                    Transformer = 1
        elif(option1=='Transformer 2'):
                    Transformer = 2
        else:
                   Transformer = 3
      else:
        grid=4
        option1 = st.sidebar.selectbox('Select the Transformer',('Transformer 1', 'Transformer 2'))
        if(option1=='Transformer 1'):
                    Transformer = 1
        else:
                   Transformer = 2
   st.sidebar.markdown('Maintance Start Date and Time')
   stdate=st.sidebar.date_input("Input the start date",datetime.date(2022,6,25))
   sttime = st.sidebar.time_input('Input the start time', datetime.time(8,00))
   option5 = st.sidebar.selectbox(
     'Status of the the Maintance Start day',
     ('Holiday', 'Working day'))

   if(option5=='Holiday'):
     dayst = 1
   if(option5=='Working day'):
     dayst = 0



   st.sidebar.markdown('Maintance End Date and Time')
   endate=st.sidebar.date_input("Input the end date",datetime.date(2022,6,25))
   entime = st.sidebar.time_input('Input the end time', datetime.time(10,00))

   option6 = st.sidebar.selectbox(
     'Status of the Maintance End day',
     ('Holiday', 'Working day'))

   if(option6=='Holiday'):
     dayst1 = 1
   if(option6=='Working day'):
     dayst1 = 0


   start=datetime.datetime.combine(stdate, sttime)
   end=datetime.datetime.combine(endate, entime)
  

   submitted = st.form_submit_button('INPUT THE REQUIRED DATA AND CLICK ON HERE')


  
  if submitted:
   predtable=dataf(start,end,dayst)
   st.subheader('EXPECTED LOAD')
   st.dataframe(predtable)

   st.subheader('LOAD VARIATION')
   col5, col6= st.columns(2)
   with col5:
      st.subheader("THULHIRIYA")
      z11=(predtable[['Thulhiriya']])
      st.line_chart(z11)
   with col6:
      st.subheader("VEYANGODA")
      z22=(predtable[['Veyangoda']])
      st.line_chart(z22)
   col7, col8= st.columns(2)
   with col7:
      st.subheader("KURUNAGALA")
      z33=(predtable[['Kurunagala']])
      st.line_chart(z33)
   with col8:
      st.subheader("KEGALLE")
      z44=(predtable[['Kegalle']])
      st.line_chart(z44)
  

   st.line_chart(predtable[['Thulhiriya','Kurunagala','Kegalle','Veyangoda']])
   st.subheader('LOAD DISTRIBUTION PLAN')
   #st.line_chart(predtable[['ThulhiriyaEX','KurunagalaEX','KegalleEX','VeyangodaEX']])

   finn=selec(predtable,grid)
   st.dataframe(finn)
   if(grid==1):
      st.area_chart(finn[['Load Requirement','Thulhiriya Exp','Kurunegala Exp','Kegalle Exp','Veyangoda Exp']])
   if(grid==2):
      st.area_chart(finn[['Load Requirement','Veyangoda Exp','Thulhiriya Exp']])
   if(grid==3):
      st.area_chart(finn[['Load Requirement', 'Kurunegala Exp','Thulhiriya Exp','Kegalle Exp']])
   if(grid==4):
      st.area_chart(finn[['Load Requirement','Kegalle Exp','Thulhiriya Exp', 'Kurunegala Exp']])
      
  st.sidebar.success('''Developed by Dilanka Wijesena
  (dilankapwijesena@gmail.com)''')


def df_to_X_y(df1, window_size):
   df_as_np = df1.to_numpy()
   X = []
   y = []
   for i in range(len(df_as_np)-window_size):
     row = [[a] for a in df_as_np[i:i+window_size]]
     X.append(row)
     label = df_as_np[i+window_size]
     y.append(label)
   return np.array(X), np.array(y)

def Final(hour,day_of_month,day_of_week,dayst):

 model1= tf.keras.models.load_model('thul.hdf5')
 model2= tf.keras.models.load_model('veya.hdf5')
 model3= tf.keras.models.load_model('kuru.hdf5')
 model4= tf.keras.models.load_model('kega.hdf5')

 rslt_df=df.loc[(df['hour'] == hour ) & ((df['day_of_month'] ==day_of_month ) | (df['day_of_week'] == day_of_week))]

 temp1=rslt_df['T1']
 temp2=rslt_df['T2']
 temp3=rslt_df['T3']
 temp4=rslt_df['T4']

 WINDOW_SIZE = 12

 X1, y1 = df_to_X_y(temp1, WINDOW_SIZE)
 X2, y2 = df_to_X_y(temp2, WINDOW_SIZE)
 X3, y3 = df_to_X_y(temp3, WINDOW_SIZE)
 X4, y4 = df_to_X_y(temp4, WINDOW_SIZE)


 pred1=(model1.predict(X1).flatten())
 pred2=(model2.predict(X2).flatten())
 pred3=(model3.predict(X3).flatten())
 pred4=(model4.predict(X4).flatten())


 test_results1 = pd.DataFrame(data={'Test Predictions1':pred1})
 test_results2 = pd.DataFrame(data={'Test Predictions2':pred2})
 test_results3 = pd.DataFrame(data={'Test Predictions3':pred3})
 test_results4 = pd.DataFrame(data={'Test Predictions4':pred4})

#serching load for internal
 input_thul=(test_results1['Test Predictions1'].iloc[-1])*4
 input_veya=(test_results2['Test Predictions2'].iloc[-1])*4
 input_kuru=(test_results3['Test Predictions3'].iloc[-1])*4
 input_kagalla=(test_results4['Test Predictions4'].iloc[-1])*4

 return input_thul,input_veya,input_kuru,input_kagalla

def maintance(input_thul,input_veya,input_kuru,input_kagalla):
#Number of transformers

 tfthul=3
 tfkuru=3
 tfveya=3
 tfkagalle=2
 thul =[] 
 kuru =[] 
 veya =[] 
 kagalla =[] 


#DEFINE Transfoermer_loads

 for c in range(0, tfthul): 
     thul.append(input_thul)
 thul=np.array(thul)


 for d in range(0, tfkuru): 
     kuru.append(input_kuru)
 kuru=np.array(kuru)

 for e in range(0, tfveya): 
     veya.append(input_veya)
 veya=np.array(veya)

 for f in range(0, tfkagalle): 
     kagalla.append(input_kagalla)

 kagalla=np.array(kagalla)

 return thul,veya,kuru,kagalla


def dataf(start,end,dayst):
 
 newdata =pd.DataFrame(pd.date_range(start=start, end=end, freq='2h'))
 newdata.columns = ['Date']
 newdata.set_index("Date",  inplace=True)
 newdata['hour'] = newdata.index.hour
 newdata['day_of_month'] = newdata.index.day
 newdata['day_of_week'] = newdata.index.dayofweek
 newdata['month'] = newdata.index.month
 newdata['year'] = newdata.index.year

 val1 = newdata.hour.values
 val2= newdata.day_of_month.values
 val3= newdata.day_of_week.values

 newdata['Thulhiriya']=0
 newdata['Veyangoda']=0
 newdata['Kurunagala']=0
 newdata['Kegalle']=0
 for i in range(len(newdata)):
   hour=val1[i]
   day_of_month=val2[i]
   day_of_week= val3[i]

   newdata['Thulhiriya'][i]=Final(hour,day_of_month,day_of_week,dayst)[0]
   newdata['Veyangoda'][i]=Final(hour,day_of_month,day_of_week,dayst)[1]
   newdata['Kurunagala'][i]=Final(hour,day_of_month,day_of_week,dayst)[2]
   newdata['Kegalle'][i]=Final(hour,day_of_month,day_of_week,dayst)[3]

 #newdata['ThulhiriyaEX']= 550 - newdata['Thulhiriya']
 #newdata['VeyangodaEX']= 550 - newdata['Veyangoda']
 #newdata['KurunagalaEX']= 550 - newdata['Kurunagala']
 #newdata['KegalleEX']= 550 - newdata['Kegalle']


 return newdata

def selec(newdata,maintaince_no):
 arr=[]
 x=newdata.day_of_month.unique()
 for i in range(len(x)):
    uni=newdata.loc[ ((newdata['day_of_month'] ==x[i]) )]
    col = []
    for j in range(1):
       col.append(int(x[i]))  
       y1=uni['Thulhiriya'].max()
       col.append(int(y1))
       y2=int(uni['Veyangoda'].max())
       col.append(y2)
       y3=int(uni['Kurunagala'].max())
       col.append(y3)
       y4=int(uni['Kegalle'].max())
       col.append(y4)
    arr.append(col)


 
 if(maintaince_no==1):
   for z in range(len(x)):
     thul=maintance(arr[z][1],arr[z][2],arr[z][3],arr[z][4])[0]
     kuru=maintance(arr[z][1],arr[z][2],arr[z][3],arr[z][4])[1]
     veya=maintance(arr[z][1],arr[z][2],arr[z][3],arr[z][4])[2]
     kagalla=maintance(arr[z][1],arr[z][2],arr[z][3],arr[z][4])[3]
     arr[z].append(thulhiriyatm1(maintaince_no,thul,kuru,veya,kagalla)[0])
     arr[z].append(thulhiriyatm1(maintaince_no,thul,kuru,veya,kagalla)[1])
     arr[z].append(thulhiriyatm1(maintaince_no,thul,kuru,veya,kagalla)[2])
     arr[z].append(thulhiriyatm1(maintaince_no,thul,kuru,veya,kagalla)[3])
     arr[z].append(thulhiriyatm1(maintaince_no,thul,kuru,veya,kagalla)[4])
     arr[z].append(thulhiriyatm1(maintaince_no,thul,kuru,veya,kagalla)[5])
   dff = pd.DataFrame(arr, columns =['Day of the month','Thulhiriya', 'Veyangoda', 'Kurunegala','Kegalle','Load Requirement','Thulhiriya Exp','Veyangoda Exp', 'Kurunegala Exp','Kegalle Exp','Status'],dtype = float) 
 if(maintaince_no==2):
   inp=veyangodatm1(maintaince_no,thul,kuru,veya,kagalla)
   for z in range(len(x)):
     thul=maintance(arr[z][1],arr[z][2],arr[z][3],arr[z][4])[0]
     kuru=maintance(arr[z][1],arr[z][2],arr[z][3],arr[z][4])[1]
     veya=maintance(arr[z][1],arr[z][2],arr[z][3],arr[z][4])[2]
     kagalla=maintance(arr[z][1],arr[z][2],arr[z][3],arr[z][4])[3]
     arr[z].append(veyangodatm1(maintaince_no,thul,kuru,veya,kagalla)[0])
     arr[z].append(veyangodatm1(maintaince_no,thul,kuru,veya,kagalla)[1])
     arr[z].append(veyangodatm1(maintaince_no,thul,kuru,veya,kagalla)[2])
     arr[z].append(veyangodatm1(maintaince_no,thul,kuru,veya,kagalla)[3])
  
   dff = pd.DataFrame(arr, columns =['Day of the month','Thulhiriya', 'Veyangoda', 'Kurunegala','Kegalle','Load Requirement','Veyangoda Exp','Thulhiriya Exp','Status'],dtype = float) 
 if(maintaince_no==3):
   for z in range(len(x)):
     thul=maintance(arr[z][1],arr[z][2],arr[z][3],arr[z][4])[0]
     kuru=maintance(arr[z][1],arr[z][2],arr[z][3],arr[z][4])[1]
     veya=maintance(arr[z][1],arr[z][2],arr[z][3],arr[z][4])[2]
     kagalla=maintance(arr[z][1],arr[z][2],arr[z][3],arr[z][4])[3]
     arr[z].append(kurunagalatm1(maintaince_no,thul,kuru,veya,kagalla)[0])
     arr[z].append(kurunagalatm1(maintaince_no,thul,kuru,veya,kagalla)[1])
     arr[z].append(kurunagalatm1(maintaince_no,thul,kuru,veya,kagalla)[2])
     arr[z].append(kurunagalatm1(maintaince_no,thul,kuru,veya,kagalla)[3])
     arr[z].append(kurunagalatm1(maintaince_no,thul,kuru,veya,kagalla)[4])

   dff = pd.DataFrame(arr, columns =['Day of the month','Thulhiriya', 'Veyangoda', 'Kurunegala','Kegalle','Load Requirement', 'Kurunegala Exp','Thulhiriya Exp','Kegalle Exp','Status'],dtype = float) 
   
 if(maintaince_no==4):
   for z in range(len(x)):
     thul=maintance(arr[z][1],arr[z][2],arr[z][3],arr[z][4])[0]
     kuru=maintance(arr[z][1],arr[z][2],arr[z][3],arr[z][4])[1]
     veya=maintance(arr[z][1],arr[z][2],arr[z][3],arr[z][4])[2]
     kagalla=maintance(arr[z][1],arr[z][2],arr[z][3],arr[z][4])[3]
     arr[z].append(kagalletm1(maintaince_no,thul,kuru,veya,kagalla)[0])
     arr[z].append(kagalletm1(maintaince_no,thul,kuru,veya,kagalla)[1])
     arr[z].append(kagalletm1(maintaince_no,thul,kuru,veya,kagalla)[2])
     arr[z].append(kagalletm1(maintaince_no,thul,kuru,veya,kagalla)[3])
     arr[z].append(kagalletm1(maintaince_no,thul,kuru,veya,kagalla)[4])
    
   dff = pd.DataFrame(arr, columns =['Day of the month','Thulhiriya', 'Veyangoda', 'Kurunegala','Kegalle','Load Requirement','Kegalle Exp','Thulhiriya Exp', 'Kurunegala Exp','Status'],dtype = float) 

 return dff


#load observation Thulhitiya
def thulhiriyatm1(maintaince_no,thul,veya,kuru,kagalla):
  mainno = maintaince_no - 1 #maintance transformer number
  maxload=550 #transformer maximum load
  reqload=thul[mainno]
  reqirement=thul[mainno]
  dff=0
  tfthul=3
  tfkuru=3
  tfveya=3
  tfkagalle=2
  bulb=0
  internaload=0
  internaload1=0
  k2t_max=300
  internaload2=0
  v2t_max=200
  internaload3=0
  kagalle2t_max=100

  print('Load to distribute =',reqload)
  for d in range(0, (tfthul)): #remove the mainace feeder
     if(d == mainno):
        print('Maintance transformer =',mainno+1)
        

     else:
       if(reqload>=0):
          if(maxload > thul[d]):
               dff=maxload-thul[d]
               internaload = internaload + dff


               print('Thulhiriya Transfomer',d+1 ,'can give ',dff,'load')
          else:
               print('Thulhiriya Transfomer',d+1 ,'can not give alternative load,next please')

  print('Thulhiriya grid substation can provide =',internaload ,'load')



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#kuru
  for d in range(0, tfkuru): 
            if(maxload > kuru[d]):
              if(internaload1<k2t_max):
                dff1=maxload-kuru[d]
                internaload1 = internaload1 + dff1
              print('Kurunagala Transfomer',d+1 ,'can give ',dff1,'load')


                  
            else:
                print('Kurunagala Transfomer',d+1 ,'can not give alternative load,next please')

  
  if( internaload1>k2t_max):
    internaload1= k2t_max
    print('Kurunagala  grid substation can provide =',internaload1 ,'load')
    reqload=reqload-k2t_max
  else:
    print('Thulhiriya  grid substation can provide =',internaload1 ,'load')
    reqload=reqload-internaload2



#-------------------------------------------------------------------------------------------------------------------------------------------------
#VEYANGODA
 
  for d in range(0, tfveya): 
              if(maxload > veya[d]):
                if(internaload2 <= v2t_max):
                  dff2 = maxload - veya[d]
                  internaload2 = internaload2 + dff2
                print('veyangoda Transfomer',d+1 ,'can give ',dff2,'load')


              else:
                  print('veyangoda Transfomer',d+1,'can not give alternative load,next please')

    
  if( internaload2>v2t_max):
    internaload2= v2t_max
    print('veyangoda  grid substation can provide =',internaload2 ,'load')
    reqload=reqload-v2t_max
  else:
    print('veyangoda  grid substation can provide =',internaload2 ,'load')
    reqload=reqload-internaload2
  

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
#kagalle

  for d in range(0, tfkagalle): 
              if(maxload > kagalla[d]):
                 if(internaload3 <= kagalle2t_max):
                  dff3 = maxload - kagalla[d]
                  internaload3 = internaload3 + dff3
                  reqload=reqload-dff3
              print('kagalle Transfomer',d+1,'can give ',dff3,'load')

  if( internaload3>kagalle2t_max):
    internaload3= kagalle2t_max
    print('kagalle  grid substation can provide =',internaload3 ,'load')
    reqload=reqload-kagalle2t_max
  else:
    print('kagalle  grid substation can provide =',internaload3 ,'load')
    reqload=reqload-internaload3



  if(reqload > 0):
    print('This Date is not Good for Transformer maintance')
    bulb='Not Good'

  else:
     print('This Date is Good for Transformer maintance')
     bulb='Good'


  return int(reqirement),internaload,internaload1,internaload2,internaload3,bulb

#load observation
def veyangodatm1(maintaince_no,thul,veya,kuru,kagalla):
  mainno = maintaince_no - 1 #maintance transformer number
  maxload=550 #transformer maximum load
  reqload=veya[mainno]
  reqirement=veya[mainno]
  dff=0
  tfthul=3
  tfkuru=3
  tfveya=3
  tfkagalle=2

  internaload=0
   
  internaload2=0
  v2t_max=200


  print('Load to distribute =',reqload)
  for d in range(0, (tfveya)): #remove the mainace feeder
     if(d == mainno):
        print('Maintance transformer =',mainno+1)
        

     else:
       if(reqload>=0):
          if(maxload > veya[d]):
               dff=maxload-veya[d]
               internaload = internaload + dff
               reqload=reqload-dff

               print('Veyangoda Transfomer',d+1 ,'can give ',dff,'load')
          else:
               print('Veyangoda Transfomer',d+1 ,'can not give alternative load,next please')

  print('Veyangoda grid substation can provide =',internaload ,'load')

#external

#-------------------------------------------------------------------------------------------------------------------------------------------------
#Thulhiriya

  for d in range(0, tfthul): 
              if(maxload > thul[d]):
                if(internaload2 <= v2t_max):
                  dff2 = maxload - thul[d]
                  internaload2 = internaload2 + dff2
                  
                  print('Thulhiriya Transfomer',d+1 ,'can give ',dff2,'load')
              else:
                  print('Thulhiriya Transfomer',d+1,'can not give alternative load,next please')

  if(internaload2>v2t_max):
    internaload2= v2t_max
    print('Thulhiriya  grid substation can provide =',internaload2 ,'load')
    reqload=reqload-v2t_max
  else:
    print('Thulhiriya  grid substation can provide =',internaload2 ,'load')
    reqload=reqload-internaload2


  

        

#----------------------------------------------------------------------------------------------------------------------------------------------------------

  if(reqload > 0):
      print('This Date is not Good for Transformer maintance')
      bulb='Not Good'

  else:
      print('This Date is Good for Transformer maintance')
      bulb='Good'

  return  reqirement,internaload,internaload2,bulb

#load observation
def kurunagalatm1(maintaince_no,thul,veya,kuru,kagalla):
  mainno = maintaince_no - 1 #maintance transformer number
  maxload=550 #transformer maximum load
  reqload=kuru[mainno]
  reqirement=kuru[mainno]
  dff=0
  tfthul=3
  tfkuru=3
  tfveya=3
  tfkagalle=2

  internaload=0

  internaload1=0
  k2t_max=300

  internaload2=0
  ka2ku_max=200



  print('Load to distribute =',reqload)
  for d in range(0, (tfkuru)): #remove the mainace feeder
     if(d == mainno):
        print('Maintance transformer =',mainno+1)
        

     else:
       if(reqload>=0):
          if(maxload > kuru[d]):
               dff=maxload-kuru[d]
               internaload = internaload + dff
               reqload=reqload-dff

               print('Kurunagala Transfomer',d+1 ,'can give ',dff,'load')
          else:
               print('Kurunagala Transfomer',d+1 ,'can not give alternative load,next please')

  print('Kurunagala grid substation can provide =',internaload ,'load')


#external

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Thulhiriya 

  for d in range(0, tfthul): 
            if(maxload > thul[d]):
              if(internaload1<k2t_max):
                dff1=maxload-thul[d]
                internaload1 = internaload1 + dff1
              print('Thulhiriya Transfomer',d+1 ,'can give ',dff1,'load')               
            else:
                print('Thulhiriya Transfomer',d+1 ,'can not give alternative load,next please')

      
  if(internaload1>k2t_max):
    internaload1= k2t_max
    print('Thulhiriya  grid substation can provide =',internaload1 ,'load')
    reqload=reqload-k2t_max
  else:
    print('Thulhiriya  grid substation can provide =',internaload1 ,'load')
    reqload=reqload-internaload1

#-------------------------------------------------------------------------------------------------------------------------------------------------
#Kagalle

  for d in range(0, tfkagalle): 
              if(maxload > kagalla[d]):
                if(internaload2 <= ka2ku_max):
                  dff2 = maxload - kagalla[d]
                  internaload2 = internaload2 + dff2
                  print('Kegalle Transfomer',d+1 ,'can give ',dff2,'load')


              else:
                  print('Kegalle Transfomer',d+1,'can not give alternative load,next please')

        

        
  if(internaload2>ka2ku_max):
    internaload2= ka2ku_max
    print('Kegalle  grid substation can provide =',internaload2 ,'load')
    reqload=reqload-ka2ku_max
  else:
    print('Kegalle  grid substation can provide =',internaload2 ,'load')
    reqload=reqload-internaload2
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

  if(reqload > 0):
      print('This Date is not Good for Transformer maintance')
      bulb='Not Good'
  else:
      print('This Date is Good for Transformer maintance')
      bulb='Good'


  return  reqirement,internaload,internaload1,internaload2,bulb

#load observation
   
def kagalletm1(maintaince_no,thul,veya,kuru,kagalla):
  mainno = maintaince_no - 1 #maintance transformer number
  maxload=550 #transformer maximum load
  reqload=kagalla[mainno]
  reqirement=kagalla[mainno]
  dff=0
  tfthul=3
  tfkuru=3
  tfveya=3
  tfkagalle=2

  internaload=0
  internaload1=0
  k2t_max=200

  internaload2=0
  ka2ku_max=200



  print('Load to distribute =',reqload)
  for d in range(0, (tfkagalle)): #remove the mainace feeder
     if(d == mainno):
        print('Maintance transformer =',mainno+1)
        

     else:
       if(reqload>=0):
          if(maxload > kagalla[d]):
               dff=maxload-kagalla[d]
               internaload = internaload + dff
               reqload=reqload-dff

               print('Kagalle Transfomer',d+1 ,'can give ',dff,'load')
          else:
               print('Kagalle Transfomer',d+1 ,'can not give alternative load,next please')

  print('Kagalle grid substation can provide =',internaload ,'load')

  
#external

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Thulhiriya 
  for d in range(0, tfthul): 
            if(maxload > thul[d]):
              if(internaload1<k2t_max):
                dff1=maxload-thul[d]
                internaload1 = internaload1 + dff1

                print('Thulhiriya Transfomer',d+1 ,'can give ',dff1,'load')     
            else:
                print('Thulhiriya Transfomer',d+1 ,'can not give alternative load,next please')


   
  if(internaload1>k2t_max):
    internaload1= k2t_max
    print('Thulhiriya  grid substation can provide =',internaload1 ,'load')
    reqload=reqload-k2t_max
  else:
    print('Thulhiriya  grid substation can provide =',internaload1 ,'load')
    reqload=reqload-internaload1

#-------------------------------------------------------------------------------------------------------------------------------------------------
#Kurunagala
  for d in range(0, tfkuru): 
              if(maxload > kuru[d]):
                if(internaload2 <= ka2ku_max):
                  dff2 = maxload - kuru[d]
                  internaload2 = internaload2 + dff2
                  print('Kurunagala Transfomer',d+1 ,'can give ',dff2,'load')
              else:
                  print('Kurunagala Transfomer',d+1,'can not give alternative load,next please')

        
      
  if(internaload2>ka2ku_max):
    internaload2= ka2ku_max
    print('Kurunagala  grid substation can provide =',internaload2 ,'load')
    reqload=reqload-ka2ku_max
  else:
    print('Kurunagala  grid substation can provide =',internaload2 ,'load')
    reqload=reqload-internaload2
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

  if(reqload > 0):
      print('This Date is not Good for Transformer maintance')
      bulb='Not Good'

  else:
      print('This Date is Good for Transformer maintance')
      bulb='Good'


  return  reqirement,internaload,internaload1,internaload2,bulb


  

if __name__=='__main__':
    main()
