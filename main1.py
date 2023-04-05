import pandas as pd
import streamlit as st
import numpy as np
import requests
import pickle
from streamlit_lottie import st_lottie
import plotly.express as px
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report


st.set_page_config(page_title="Cars Price Prediction",  layout="wide")

df = pd.read_csv("E:\Epsilon AI\My_Projects\Final Project\Final\Cars.csv")

st.title('Cars Price Prediction :car: ')

st.write("Price of used cars are not stable according to the current situation of our country, so the prices in data should at least be updated or increased, i made change in the prices by increase all of them 20% to be nearest to the real one")
st.write("Our dataset has 3 Brands: ")
col11,col22,col33= st.columns(3)

col11.image('Chevorlet.jpg')
col22.image('Hyundai.png')
col33.image('Fiat.jpg')


def numberformat(x):
    return( "{:,}".format(x))

def load_lottieurl(url):
    r = requests.get(url)
    return r.json()




df.drop('Unnamed: 0' , axis= 1, inplace= True)
df.drop(df[(df['Price'] < 30)].index , axis= 0, inplace = True)
df.drop(df[(df['Price'] < 60) & (df['Year'] > 2005) ].index ,axis= 0, inplace = True)
df.Price = df.Price.apply(lambda x : x*1.2)
df['Price'] = df['Price'].round(decimals=2)
col1,col2 = st.columns(2)
col1.markdown('')
col1.markdown('')
col1.markdown('')
col1.markdown("## Here's sample of our dataset: ")
col2.write(df.head())

lottie_coding1 = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_a3emlnqk.json")
st_lottie(lottie_coding1, height=300, key="coding1")


#df = pd.get_dummies(columns= ['Brand','Body', 'Fuel' ,'Transmission', 'Engine' ] , data =df)

# loading the saved model



# creating a function for prediction

st.markdown('## Filter your car on your budget & Brand & years :mag:') 


budget = df['Price'].unique()
budget = df.sort_values(by='Price',ascending=True).Price.unique()

Price = st.select_slider('Select budget in K', options= budget , value = ((budget.min(), budget.max())))
st.write('You selected budget between', numberformat(Price[0]) , ' and ', numberformat(Price[1] ))

st.markdown('### Filter by years :calendar:')
years = df['Year'].unique()
years= df.sort_values(by='Year',ascending=True).Year.unique()


Year = st.select_slider('Select year', options = years , value = (years.min(), years.max()))
st.write('You selected years between', numberformat(Year[0] ), ' and ', numberformat(Year[1]) )

st.markdown('### Filter by Brand')
Brand = st.multiselect('Select Brand', options= df['Brand'].unique())


df = df[(df['Price'] >= Price[0]) & (df['Price'] <= Price[1])]
df = df[(df['Year'] >= Year[0]) & (df['Year'] <= Year[1])]
df = df[df['Brand'].isin(Brand)]

st.write('You selected', numberformat(len(df)) , ' Cars')

st.write(df[['Brand','Model','Body','Color','Year','Fuel','Kilometers','Engine','Transmission','Price','Gov']])

#input_data= {
   # "Brand": [Brand], "Model": [Model], "Body": [Body], "Color": [Color], "Year": [Year],
#}
st.write("# Making Prediction: ")

Brand_options= ['Hyundai', 'Chevrolet', 'Fiat']

Hyundai_models= ['Accent', 'Elantra', 'Verna', 'I10', 'Avante', 'Excel', 'Matrix', 'Tucson']
Chevrolet_models= ['Aveo','Optra','Lanos','Cruze']
Fiat_models= ['Tipo', 'Punto', 'Shahin', 'Uno','128', '131']

Color_options= ['Black', 'Silver', 'Gray', 'Blue- Navy Blue', 'Green', 'Red',
       'Gold', 'Other Color', 'Burgundy', 'White', 'Yellow', 'Brown',
       'Orange', 'Beige']

Fuel_options= ['Benzine', 'Natural Gas']

Kilometers_options= ['140000 to 159999', '180000 to 199999', '10000 to 19999',
       'More than 200000', '90000 to 99999', '100000 to 119999',
       '160000 to 179999', '120000 to 139999', '0 to 9999',
       '20000 to 29999', '30000 to 39999', '80000 to 89999',
       '60000 to 69999', '70000 to 79999', '40000 to 49999',
       '50000 to 59999']

Engine_options= ['1600 CC', '1000 - 1300 CC', '1400 - 1500 CC']
Engine_option= ['1600 CC',  '1400 - 1500 CC']
Engine_option_small= [ '1000 - 1300 CC', '1400 - 1500 CC']

Transmission_options= ['Automatic', 'Manual']

Gov_options= ['Giza', 'Qena', 'Cairo', 'Minya', 'Alexandria', 'Dakahlia', 'Suez',
       'Sharqia', 'Kafr al-Sheikh', 'Beheira', 'Ismailia', 'Sohag',
       'Monufia', 'Qalyubia', 'Beni Suef', 'Asyut', 'Fayoum', 'Gharbia',
       'Matruh', 'Damietta', 'Red Sea', 'Port Said', 'Luxor',
       'South Sinai', 'New Valley', 'Aswan']

lottie_coding = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_fsyj0fzz.json")
st_lottie(lottie_coding, height=300, key="coding")


Brand_Selected= st.selectbox("Select Brand", Brand_options)

car_body=['Sedan', 'Hatchback', 'SUV']


if Brand_Selected == 'Hyundai':
    car_Modell= st.selectbox("Select Model from Hyundai Models", Hyundai_models)
    Car_Bodyy= st.selectbox("Select Body",car_body)
    

    if ((car_Modell == 'Accent' or car_Modell == 'Elantra' or car_Modell == 'Verna' or car_Modell =='Avante') and (Car_Bodyy== "Sedan")):


        st.write(car_Modell,' is a good choice')
       
    elif ((car_Modell == 'Excel' or car_Modell == 'I10') and (Car_Bodyy== "Sedan" or Car_Bodyy== "Hatchback")):
        st.write(car_Modell,' is a good choice')
            
        
    elif ((car_Modell == 'Matrix') and (Car_Bodyy== "Hatchback")):
        st.write(car_Modell,' is a good choice')
       

    elif ((car_Modell == 'Tucson') and (Car_Bodyy== 'SUV')):
        st.write(car_Modell,' is a good choice')
         
        

    
    else:
        st.write("There is no ",car_Modell," ",Car_Bodyy)
        
    


        #st.selectbox("Select Body",car_body)
        #if car_body== "Sedan":
            #st.write('Hyunadi ',Hyundai_models,' is a good choice')
        #else:
           # st.write("There is no ",Hyundai_models,' Hatchback or SUV')

    #elif Hyundai_models == 'Excel' or 'I10':
        #st.selectbox("Select Body",car_body)
        #if car_body== "Sedan":
           # st.write('Hyunadi ',Hyundai_models,' is a good choice')
        #elif car_body== "Hatchback":
           # st.write('Hyunadi ',Hyundai_models,' is a good choice')
       # else:
           # st.write("There is no ",Hyundai_models,' SUV')
    #else:
       # st.selectbox("Select Body",car_body)
       # if car_body== 'SUV':
           # st.write('Hyunadi ',Hyundai_models,' is a good choice')
       # else:
           # st.write("There is no ",Hyundai_models,' Hatchback or Sedan')


elif Brand_Selected == 'Chevrolet':   
    car_Modell= st.selectbox("Select Model from Chevrolet Models", Chevrolet_models)
    Car_Bodyy= st.selectbox("Select Body",car_body)

    if ((car_Modell == 'Aveo' or car_Modell == 'Lanos' or car_Modell == 'Optra' or car_Modell =='Cruze') and (Car_Bodyy== "Sedan")):


        st.write(car_Modell,' is a good choice')
       
    
    else:
        st.write("There is no ",car_Modell," ",Car_Bodyy)
else:
    car_Modell= st.selectbox("Select Model from Fiat Models", Fiat_models)
    Car_Bodyy= st.selectbox("Select Body",car_body)

    if ((car_Modell == '128' or car_Modell == '131' or car_Modell == 'Shahin') and (Car_Bodyy== "Sedan")):


        st.write(car_Modell,' is a good choice')
    
    elif ((car_Modell == 'Punto' or car_Modell == 'Uno') and (Car_Bodyy== "Hatchback")):
        st.write(car_Modell,' is a good choice')

    elif ((car_Modell == 'Tipo') and (Car_Bodyy== "Sedan" or Car_Bodyy== "Hatchback")):
        st.write(car_Modell,' is a good choice')

       
    
    else:
        st.write("There is no ",car_Modell," ",Car_Bodyy)

#Car_Bodyy= st.selectbox("Select Body",car_body)

Car_Colorr= st.selectbox("Select Color", Color_options)

year_range= range(1972,2022)
Car_Yearr= st.select_slider("Select Year",options= year_range, value=2000)

Car_Fuell= st.selectbox("Select Fuel",Fuel_options)

Car_Kilometers= st.selectbox("Select range of Kilometers",Kilometers_options)

if (car_Modell == 'Accent' or car_Modell =='Excel' ):

    Car_Engine= st.selectbox("Select Engine",Engine_options)

elif (car_Modell == 'Elantra' or car_Modell== 'Avante' or car_Modell== 'Matrix' or car_Modell== 'Verna' or car_Modell== 'Lanos' or car_Modell== 'Optra'):

    Car_Engine= st.selectbox("Select Engine",Engine_option)

elif (car_Modell == 'I10'):

    Car_Engine= st.selectbox("Select Engine",Engine_option_small)

elif (car_Modell== 'Aveo' or car_Modell== 'Shahin'):

    Car_Engine= '1400 - 1500 CC'


elif (car_Modell== '128' or car_Modell== 'Punto' or car_Modell== 'Uno'):

    Car_Engine= '1000 - 1300 CC'

else:
    Car_Engine= '1600 CC'



Car_Transmission= st.selectbox("Select Type of Transmission",Transmission_options)

Car_Gov= st.selectbox("Select Gov",Gov_options)

trained_model=pickle.load(open('trained_model_rf.sav','rb'))

trained_model_scaler=pickle.load(open('scaler.sav','rb'))

train_model_encoder= pickle.load(open('encoder.sav','rb'))


input_data = {}
input_data = { 'Brand' :Brand_Selected,
              "Model" :car_Modell,
              "Body" : Car_Bodyy , 
              "Color" : Car_Colorr, 
              "Year" :[Car_Yearr],
              "Fuel" :Car_Fuell, 
              "Kilometers" :Car_Kilometers,
              "Engine":Car_Engine,
              "Transmission" :Car_Transmission,
              "Gov":Car_Gov}
x_x = pd.DataFrame.from_dict(input_data)
x_x = train_model_encoder.transform(x_x)
numerical_columns= ["Year"]
x_x[numerical_columns] = trained_model_scaler.transform(x_x[numerical_columns])
predection =trained_model.predict(x_x)
#

if st.button("Predict") :
    st.write("Price is " , round(predection[0]),'K')





