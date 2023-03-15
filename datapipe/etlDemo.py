"""
*** Modo de Ejecucion ***
streamlit run etlDemo.py 
"""


import os
import sys
import petl
from petl.util.base import Table

import configparser
import requests
import datetime
import json
import decimal

import streamlit as st
import calplot as cal

st.set_option('deprecation.showPyplotGlobalUse', False)

def app():
    def line(n):
        for i in range(n):
            st.markdown('')


    config = configparser.ConfigParser()
    try:
        config.read('./datapipe/etlDemo.ini')
    except Exception as e:
        print('error reading INI file:' + str(e) )
        sys.exit()

    url = config['CONFIG']['url']
    startdate = config['CONFIG']['startdate']
    end='&end_date='
    enddate = config['CONFIG']['enddate']



    st.title('ETL from API: plot a Time Series')
    st.markdown('Using Bank of Canada [API](https://www.bankofcanada.ca/valet/docs)🔗 to get the dialy exchange rate between USD and CAD.')
    st.markdown('Set start date and end date to get a JSON message, parse it and generate the plot for the time series.')
    st.markdown('time series forecasting coming soon.')
    line(2)


    with st.form(key='my_form1'):
        with st.expander("📅 Select Dates From/To   :", expanded=False):
            startdate = st.date_input("Begin", datetime.datetime.strptime(startdate, '%Y-%m-%d'))  #corrigir x fecha de CONFIG
            enddate = st.date_input("End", datetime.datetime.strptime(enddate, '%Y-%m-%d'))      #corrigir x fecha de CONFIG
        st.form_submit_button(label='Aplicar')
    line(2)


    try:
        BOCResp = requests.get(url+str(startdate)+end+str(enddate))
        done = False
    except Exception as e:
        print('error reading url:' + str(e) )
        done = True
        #sys.exit()

	#print(BOCResp.text)
    st.text_area('LOG:',BOCResp.text, height=240)

    BOCRates = []
    BOCDates = []

    if (done == False):
        if (BOCResp.status_code == 200):
            BOCRaw = json.loads(BOCResp.text)

            for row in BOCRaw['observations']:
                BOCDates.append(datetime.datetime.strptime(row['d'], '%Y-%m-%d'))
                BOCRates.append(decimal.Decimal(row['FXUSDCAD']['v']))

            #print(BOCRates)

            exchRates = petl.fromcolumns([BOCDates,BOCRates], header=['date', 'rate'] )
            #print(exchRates)

            df = petl.todataframe(exchRates)


            agree = st.checkbox('Check Dataframes:')

            if agree:
                col1, col2 = st.columns(2)
                with col1:
                    st.write('**PETL** FORMAT:')
                    st.write(exchRates)
                with col2:
                    st.write('**PANDAS** FORMAT:')
                    st.write(df.head())
                st.write("---")

            import seaborn as sns
            import matplotlib.pyplot as plt

            # st.set_option('deprecation.showPyplotGlobalUse', False)
            # sns.set(context="paper", font="monospace")
            # sns.set_theme(style="darkgrid")
            # plt.title('Time Series from selected period:')
            # 
            # plt.xticks(rotation=45)
            # # Plot the responses for different events and regions
            # sns.lineplot(x="date", y="rate",
            #             #hue="rate", style="event",
            #             data=df)
            # st.pyplot()




            # import altair as alt
            # source = df
            # bar = alt.Chart(source).mark_line().encode(
            #     x='date:O',
            #     y='rate:Q'
            # )
            # rule = alt.Chart(source).mark_rule(color='red').encode(
            #     y='mean(rate):Q'
            # )
            # ch = (bar ).properties(width=600)
            # st.write(ch)


            import plotly.graph_objects as go
            fig = go.Figure([go.Scatter(x=df['date'], y=df['rate'])])
            fig.update_layout(title="Dialy Exchange Rate")
            st.write(fig)


            st.write('---')
            df["rate"] = df.rate.astype(float)
            vmin = df['rate'].min()
            vmax = df['rate'].max()

            #Set orderdate as index
            df.set_index('date', inplace = True)
            pl1 = cal.calplot(data = df['rate'], cmap = 'Blues', figsize = (16, 8), suptitle = "Dialy Exchange Rate", vmin = vmin, vmax = vmax)
            st.pyplot()




    def sep():
        st.markdown('<hr style="height:2px;border-width:0;color:gray;background-color:darkblue">', unsafe_allow_html=True)

    about =  """
    [![Victor0](https://img.shields.io/badge/python-gray.svg?colorA=grey&colorB=white&logo=python)](https://www.python.org/)
    [![Victor1](https://img.shields.io/badge/pandas-gray.svg?colorA=grey&colorB=white&logo=pandas)](https://pandas.pydata.org/) 
    [![Victor2](https://img.shields.io/badge/plotly-gray.svg?colorA=grey&colorB=white&logo=plotly)](https://plotly.com/) 
    [![Victor3](https://img.shields.io/badge/streamlit-gray.svg?colorA=grey&colorB=white&logo=streamlit)](https://streamlit.io/) 
    """

    #sep()
    #st.markdown(about)

    line(20)
    st.info(about)
