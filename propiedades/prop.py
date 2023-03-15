
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st 
import requests
import json

st.set_option('deprecation.showPyplotGlobalUse', False)
#pd.set_option("display.precision", 2)
#pd.option_context('display.float_format', '{:0.2f}'.format)

def app():

    cols = ['price', 'expenses', 'm2', 'rooms', 'bathrooms', 'years', 'address', 'barrio', 'subbarrio', 'note', 'link']
    df = pd.read_csv('./propiedades/outArgScrap.csv', names=cols, header=None)
    df = df.reset_index(drop=True)


    df = df[pd.to_numeric(df['price'], errors='coerce').notnull()]
    df = df.astype({"price": int})

    df = df[pd.to_numeric(df['rooms'], errors='coerce').notnull()]

    df['m2'] = df['m2'].str.replace(',','.')
    df = df[pd.to_numeric(df['m2'], errors='coerce').notnull()]

    df = df.astype({"m2": float, "rooms": int})

    df['barrio'] = df['barrio'].str.strip()    #elimino blancos

    df['m2value'] = df['price']/df['m2']
    df.round(2)

    #SOLO PARA DEBUG:
    df.drop(['note', 'link'], axis=1, inplace=True)


    def title(s):
        st.markdown(f"<h1 style='text-align: center; color: darkblue;'>{s}</h1>", unsafe_allow_html=True)
    def sep():
        st.markdown('<hr style="height:2px;border-width:0;color:gray;background-color:darkblue">', unsafe_allow_html=True)

    #st.markdown("<style> .css-fg4pbf {background: #d7ddde;}</style>",unsafe_allow_html=True )
    title("Venta de Propiedades Buenos Aires:")
    sep()


    st.subheader('Data Scraping: ')
    st.write(df.head(100))
    st.write(f'Shape: {df.shape}')

    # Cantidad Null:
    #st.write(df.isnull().sum())


    #missing data
    st.subheader('Missing Data: ')

    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=False)
    plt.title("Missing values");
    st.pyplot()

    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    st.write(missing_data.head(20))

    # baños: al menos uno
    df['bathrooms'].fillna(1, inplace=True)
    df = df.astype({"bathrooms": int})

    # expensas
    df['expenses'].fillna(0, inplace=True)
    df = df.astype({"expenses": int})

    df['years'] = np.floor(pd.to_numeric(df['years'], errors='coerce')).astype('Int64')

    # Remove outliers years:
    df.loc[df['years'] > 100, 'years'] = 100
    # Fill NAN years with mean:
    df['years'].fillna(np.floor(pd.to_numeric(df['years'].mean(), errors='coerce')).astype(np.int64), inplace=True)

    #st.write(df)
    st.write(' ')
    st.write(' ')

    with st.form(key='my_form1'):
        with st.expander("📋 Seleccion por Barrio:", expanded=False):
            optionCl = st.multiselect("Barrios:", options=df['subbarrio'].unique() )
            if len(optionCl) > 0:
                df = df.loc[df['subbarrio'].isin(optionCl) ]
        st.form_submit_button(label='Aplicar')


    df['subbarrio']= np.where(df['subbarrio'].isnull(), df['barrio'], df['subbarrio'])



    st.write('---')
    st.subheader('Cantidad de ambientes: ')
    agree5 = st.checkbox('Activar', value=False, key=5)
    if agree5:
        sns.countplot(df[(df['rooms'] <= 15 ) ]['rooms'])
        st.pyplot()


    st.write('---')
    st.subheader('Precios por Barrios: (en M u$s)')
    agree4 = st.checkbox('Activar', value=False, key=4)
    if agree4:
        plt.figure(figsize=(12,15))
        sns.scatterplot(y='subbarrio', x='price', data=df[(df['price'] < 5000000 )  ])
        st.pyplot()

    st.write('---')
    st.subheader('Precios:')
    agree3 = st.checkbox('Activar', value=False, key=3)
    if agree3:
        plt.figure(figsize=(12,6))
        sns.distplot(df[(df['price'] > 10000) & (df['price'] < 1000000)]['price'], rug=True)
        st.pyplot()



    st.write('---')
    st.subheader('Dispersion valor del m2 por Barrios: ')
    agree1 = st.checkbox('Activar', value=False, key=1)
    if agree1:
        # Find the order
        grouped = df.loc[:,['subbarrio', 'm2value']] \
            .groupby(['subbarrio']) \
            .median() \
            .sort_values(by='m2value', ascending=False)
        plt.figure(figsize=(12,18))
        sns.boxplot(y='subbarrio', x='m2value', data=df[ (df['m2value'] > 500 ) & (df['m2value'] < 10000 ) ],  order=grouped.index)
        st.pyplot()
        st.write(grouped)

    st.write('---')
    st.subheader('Precio m2: ')
    agree2 = st.checkbox('Activar', value=False, key=2)
    if agree2:
        option = st.selectbox(
            'M2 segun barrio',
            df['subbarrio'].unique())
        if len(option) > 0:
            st.write('Valor del m2 promedio:', grouped[grouped.index == option]['m2value'])





    # subset = df[ (df['m2value'] > 500 ) & (df['m2value'] < 10000 ) & (df['rooms'] <= 12 ) ]
    # st.write('---')
    # st.subheader('Precio m2: ')
    # import altair as alt
    # ch = alt.Chart(subset).mark_circle(size=60).encode(
    #     alt.X('rooms', scale=alt.Scale(zero=False)),
    #     alt.Y('m2value', scale=alt.Scale(zero=False, padding=1)),
    #     alt.OpacityValue(0.6),
    #     color='m2value',
    #     size='m2value',
    #     tooltip=['barrio', 'm2value']
    # ).properties(
    #  width=600,
    #  height=600,
    #  title='m2 value VS rooms'
    # ).interactive()
    # st.write(ch)


    # MACHINE LEARNING:

    #df = pd.get_dummies(df)
    #print(df.shape)
    #train = df[:ntrain]
    #test = df[ntrain:]



    lstBarrios = df['barrio'].unique()

    bar  = pd.get_dummies(df['barrio'], prefix='ba', drop_first = True)
    #sbar  = pd.get_dummies(df['subbarrio'], prefix='sb', drop_first = True)
    df = pd.concat([df, bar], axis=1)

    df.drop(['address', 'barrio', 'subbarrio'], axis=1, inplace=True)



    #st.subheader('Missing Data after: '
    #st.write(df.shape
    #plt.figure(figsize=(10, 6)
    #sns.heatmap(df.isnull(), yticklabels=False, cbar=False
    #plt.title("Missing values")
    #st.pyplot(



    def generateModel():
        st.write('Propiedad 1 a analizar:')
        st.write(df.iloc[100])

        st.write('Propiedad 2 a analizar:')
        st.write(df.iloc[500])


        #Elimino columna redundante
        df.drop('price', axis=1, inplace=True)

        #Single Value
        micasa = df.iloc[[100]]
        micasa2 = df.iloc[[500]]



        X = df
        y = df.pop('m2value')
        print(X.shape)
        print(y.shape)

        #X_train.drop('m2value', axis=1, inplace=True)

        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import GridSearchCV
        from sklearn.model_selection import train_test_split #to create validation data set

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) #X_valid and y_valid are the validation sets

        linreg = LinearRegression()
        linreg.fit(X_train, y_train)

        # GridSearchCV
        #parameters_lin = {"fit_intercept" : [True, False], "normalize" : [True, False], "copy_X" : [True, False]}
        #grid_linreg = GridSearchCV(linreg, parameters_lin, verbose=1 , scoring = "r2")
        #grid_linreg.fit(X_train, y_train)
        #print("Best LinReg Model: " + str(grid_linreg.best_estimator_))
        #print("Best Score: " + str(grid_linreg.best_score_))


        # regression coefficients
        print('Coefficients: ', linreg.coef_)
        # variance score: 1 means perfect prediction
        print('Variance score: {}'.format(linreg.score(X_test, y_test)))
        
        # plot for residual error
        # plt.style.use('fivethirtyeight')
        # plt.scatter(linreg.predict(X_train), linreg.predict(X_train) - y_train,
        #             color = "green", s = 10, label = 'Train data')
        # plt.scatter(linreg.predict(X_test), linreg.predict(X_test) - y_test,
        #             color = "blue", s = 10, label = 'Test data')
        # plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)
        # plt.legend(loc = 'upper right')
        # plt.title("Residual errors")
        # st.pyplot()




        #micasa = np.array([1,0 , 33, 0, 0, 100, 0, 1, 0, 0]).reshape((1, -1))

        print("micasa:")
        print(micasa)
        micasa.drop(['m2value'], axis=1, inplace=True)
        a = linreg.predict(micasa) 
        st.write(f'Precio Estimado xm2: {a}')
        st.write(f'Precio Estimado : {(micasa.m2*a)}')



        # # Saving Example:
        # import pickle
        # 
        # # create an sample.pkl
        # with open ('sample.pkl', 'wb') as files:
        #     pickle.dump(micasa, files)



        # # Saving Model:
        # import pickle
        # 
        # # create an iterator object with write permission - model.pkl
        # with open ('model.pkl', 'wb') as files:
        #     pickle.dump(linreg, files)
        # 
        # # load saved model
        # with open('model.pkl' , 'rb') as f:
        #     lr = pickle.load(f)
        # 
        # micasa2.drop(['m2value'], axis=1, inplace=True)
        # b = lr.predict(micasa2)
        # st.write(f'Precio Estimado xm2: {b}')
        # st.write(f'Precio Estimado : {(micasa2.m2*b)}')


    #generateModel()


    def makePrediction():

        st.write('---')
        st.subheader('Predecir Valor de Propiedad: ')
        press = False

        with st.form(key='my_form2'):
            getBarrio = st.selectbox('Barrio', options=lstBarrios)
            getM2 = st.text_input('m2')
            getRooms = st.text_input('Ambientes')
            getBathrooms = st.text_input('Baños')
            getYears = st.text_input('Años')
            press = st.form_submit_button(label='Aplicar')

        if press:
            
            urlLocal = 'http://127.0.0.1:8000/predict'

            with st.echo():
                url = 'https://propiedadesba.herokuapp.com/predict'
                
                payload = json.dumps({
                    'barrio': getBarrio,
                    'm2': getM2,
                    'rooms': getRooms,
                    'bathrooms': getBathrooms,
                    'years': getYears
                })

                headers = {
                'Content-Type': 'application/json'
                }

                response = requests.request("POST", url, headers=headers, data=payload)
                data = response.json()

            #st.write(response.text)
            st.write(f'Precio Estimado x m2: **{data["prediction"][0]:.2f}** USD')
            st.write(f'Precio Estimado : **{( int(getM2) * data["prediction"][0]):.2f}** USD')
            press = False

    makePrediction()


    def line(n):
        for i in range(n):
            st.markdown('')

    about =  """
    [![Victor0](https://img.shields.io/badge/python-gray.svg?colorA=grey&colorB=white&logo=python)](https://www.python.org/)
    [![Victor1](https://img.shields.io/badge/pandas-gray.svg?colorA=grey&colorB=white&logo=pandas)](https://pandas.pydata.org/) 
    [![Victor2](https://img.shields.io/badge/fastapi-gray.svg?colorA=grey&colorB=white&logo=FastAPI)](https://fastapi.tiangolo.com/) 
    [![Victor3](https://img.shields.io/badge/streamlit-gray.svg?colorA=grey&colorB=white&logo=streamlit)](https://streamlit.io/) 
    """

    line(20)
    st.info(about)