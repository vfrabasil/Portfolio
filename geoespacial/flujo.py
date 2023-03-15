import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk


#@st.experimental_memo
def load_data(nrows):
    data = pd.read_csv("./geoespacial/nuevas-estaciones-bicicletas-publicas.csv", nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis="columns", inplace=True)

    data['wkt'] = data['wkt'].str.replace('POINT ','')
    data['wkt'] = data['wkt'].str.replace('(','')
    data['wkt'] = data['wkt'].str.replace(')','')


    data[['lon', 'lat']] = data['wkt'].str.split(' ', 1, expand=True)
    data["lat"] = pd.to_numeric(data["lat"])
    data["lon"] = pd.to_numeric(data["lon"])
    return data


def app():
    data = load_data(100000)



    def title(s):
        st.markdown(f"<h1 style='text-align: center; color: darkblue;'>{s}</h1>", unsafe_allow_html=True)
    def sep():
        st.markdown('<hr style="height:2px;border-width:0;color:gray;background-color:darkblue">', unsafe_allow_html=True)

    st.markdown("<style> .css-fg4pbf {background: #d7ddde;}</style>",unsafe_allow_html=True )
    title("🚴 Estaciones bicicletas publicas en BUENOS AIRES 🚴")
    sep()

    data.drop(['wkt', 'id'], axis=1, inplace=True)
    st.write(data.style.set_properties(**{'background-color': 'steelblue',
                            'color': 'white',
                            'border-color': 'white'}))

    data.dropna(inplace=True) 
    data.drop(['codigo','nombre','tipo','horario','anclajes_t'], axis=1, inplace=True)

    def map(data, lat, lon, zoom):

        view_state = pdk.ViewState(
            longitude=lon,
            latitude=lat,
            zoom=zoom,
            min_zoom=5,
            max_zoom=15,
            pitch=40.5,
            bearing=-27.36)

        layer= pdk.Layer(
                    #"HexagonLayer",
                    type='IconLayer',
                    data=data,
                    get_icon='icon_data',
                    get_size=4,
                    pickable=True,
                    size_scale=15,
                    get_position=["lon", "lat"],
                )
            
        tooltip = {
        "html": "<b>Direccion:</b> {ubicacion} <br/> ",
        "style": {
                "backgroundColor": "steelblue",
                "color": "white"
        }
        }


        r = pdk.Deck(layers=[layer], initial_view_state=view_state, map_style="mapbox://styles/mapbox/light-v9", tooltip=tooltip)
        st.write(r)


    sep()
    zoom_level = 12
    midpoint = (np.average(data["lat"]), np.average(data["lon"]))

    icon_data = {
        "url": "https://img.icons8.com/plasticine/100/000000/marker.png",
        "width": 128,
        "height":128,
        "anchorY": 128
    }
    data['icon_data']= None
    for i in data.index:
        data['icon_data'][i] = icon_data



    map(data, midpoint[0], midpoint[1], 11)
    sep()
    st.write('Origen de los datos: https://data.buenosaires.gob.ar/dataset/estaciones-bicicletas-publicas/resource/juqdkmgo-1011-resource ')

