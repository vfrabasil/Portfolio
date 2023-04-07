import streamlit as st

# custom made multiapp:
from multiapp import MultiApp

# import your app modules here:
from home import home
#1 from heatmap import heatmap
from dataexplorer import explorer
from stormbound import storm
#1 from NLP import ner
from titanic import titanic
from geoespacial import flujo
from datapipe import etlDemo
from propiedades import prop

#st.markdown("""
## Multi-Page App
#This multi-page app is using the [streamlit-multiapps](https://github.com/upraneelnihar/streamlit-multiapps) 
#""")

# st.set_page_config(page_title='Victor Frabasil\'s portfolio' ,page_icon='📊')
app = MultiApp()


# Add all your application here
app.add_app("🏠 HOME", home.app)
#1 app.add_app("🎨 Python Heatmaps (Visualization Libraries)", heatmap.app)
app.add_app("🔬 Transactional Data Explorer", explorer.app)
app.add_app("🎮 Game Statistics (EDA/Interactive Dashboards)", storm.app)
#1 app.add_app("📑 Text Language Processing (NLP/spaCy)", ner.app)
app.add_app("🚢 Titanic Survival Analysis and Prediction (EDA/ML/Feature Eng.)", titanic.app)
app.add_app("📈 Variacion Cotizacion USD (Data Pipeline/ETL/API/Time Series forecasting)", etlDemo.app)
app.add_app("🏢 Venta de Propiedades en Buenos Aires (EDA/ML/FastAPI)", prop.app)
app.add_app("🌎 Data Geolocation (pydeck)", flujo.app)

#Transaction analysis App. This app uses two .csv files to create a ratio difference matrix.
#Try different libraries and palettes. Some libraries use a matrix to generate the heatmap, therefore the original dataset is pivoted to generate it. As a preview, we will show the original version of the dataset, and the version converted into a matrix.
#Exploratory Data Analisys for natural language processing (NLP) using spaCy.
#Using scrapping from Stormbound game dataset to perform an EDA analysis with its units types.

# The main app
app.run()



#app.add_app("Data", data.app)