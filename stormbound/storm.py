from altair.vegalite.v4.schema.channels import Tooltip
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import altair as alt
import squarify


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix , accuracy_score, precision_score, recall_score
from PIL import Image

#Modo de Ejecucion:
# streamlit run storm.py --server.maxUploadSize=1024

def title(s):
    st.text("")
    st.markdown(f"<h1 style='text-align: center; color: green;'>{s}</h1>", unsafe_allow_html=True)
    st.markdown('<hr style="height:2px;border-width:0;color:gray;background-color:green">', unsafe_allow_html=True)
    st.text("")

def subtitle(s):
    st.text("")
    st.markdown(f'<p style="background-color:#0066cc;color:#33ff33;font-size:24px;border-radius:2%;">{s}</p>', unsafe_allow_html=True)
    st.text("")


def eda(df):
    ''' check what data types we have: '''

    subtitle('Describe data information:')
    #st.write(df.dtypes)
    st.write(df.describe())
    st.write(f'Dataframe size: {df.shape}')

def clean_and_split(df):
    legendary_df = df[df['is_legendary'] == 1]
    normal_df = df[df['is_legendary'] == 0].sample(75)
    legendary_df.fillna(legendary_df.mean(),inplace=True)
    normal_df.fillna(normal_df.mean(),inplace=True)
    feature_list = ['weight_kg' , 'height_m' , 'sp_attack' , 'attack' , 'sp_defense' , 'defense' , 'speed' , 'hp' , 'is_legendary']
    sub_df = pd.concat([legendary_df,normal_df])[feature_list]
    X = sub_df.loc[:, sub_df.columns != 'is_legendary']
    Y = sub_df['is_legendary']
    X_train, X_test , y_train , y_test = train_test_split(X ,Y ,random_state=1 ,test_size= 0.2 ,shuffle=True,stratify=Y)
    return X_train , X_test , y_train , y_test


def fRarity(x):
    if x == 'Common':
        return 0
    elif x == 'Rare':
        return 1
    elif x == 'Epic':
        return 2
    elif x == 'Legendary':
        return 3
    return 4

def fFaction(x):
    if x == 'Tribes of Shadowfen':
        return 0
    elif x == 'Swarm of the East':
        return 1
    elif x == 'Ironclad Union':
        return 2
    elif x == 'Winter Pact':
        return 3
    return 4

def fCard(x):
    if x == 'Unit':
        return 0
    elif x == 'Structure':
        return 1
    elif x == 'Spell':
        return 2
    return 3


def dataClean(df):
    ''' First we need to do some minor cleanups and encoding the categorical variables. Pandas makes this straightforward: '''

    #cleanup_nums = {"Rarity":     {"Legendary": 3, "Epic": 2, "Rare": 1, "Common": 0},
    #                "num_cylinders": {"four": 4, "six": 6, "five": 5, "eight": 8,
    #                              "two": 2, "twelve": 12, "three":3 }}

    #st.write(pd.Series(df.Rarity.value_counts(), ).to_dict() )
    #st.write(dict(zip(df.Rarity, df.Rarity.value_counts())))

    df['iRarity'] = df['Rarity'].apply(fRarity)
    df['iFaction'] = df['Faction'].apply(fFaction)
    df['iCard'] = df['Card Type'].apply(fCard)
    df['Move'] = df['Move'].fillna(0)
    df['Strength'] = df['Strength'].fillna(0)
    df['Unit Type'] = np.where(df['Unit Type'] == ' ', df['Card Type'], df['Unit Type'])


def app():

    # Load Data
    title('STORMBOUND')

    col1, col2, col3 = st.columns(3)


    img1 = Image.open("./stormbound/image1.png")
    #img1 = Image.open('./Floor_plans/3-marla.png')
    img2 = Image.open("./stormbound/image2.png")    

    col1.image(img1, use_column_width=None, width = 200 )
    col2.subheader("DATA ANALISYS")
    col3.image(img2, use_column_width=None, width = 200 )
    


    df = pd.read_csv('./stormbound/stormbound_card_list.csv', sep=',')

    #df2 = pd.read_html('https://stormboundkingdomwars.fandom.com/wiki/Cards')


    st.write(df)

    # Transform Data
    subtitle('Units list:')
    dataClean(df)
    st.dataframe(df)

    # Basic Info
    num_total = len(df)
    num_legendary = len(df[df['Rarity'] == 'Legendary'])
    num_epic = len(df[df['Rarity'] == 'Epic'])
    num_rare = len(df[df['Rarity'] == 'Rare'])
    num_common = len(df[df['Rarity'] == 'Common'])


    subtitle('Basic statistics:')

    st.write('''
    Number of Cards: {}
    '''.format(num_total))

    st.write('''
    Number of Features :{}
    '''.format(df.shape[1]))

    st.write('''
    Number of Legendary Cards: {}
    '''.format(num_legendary))

    st.write('''
    Number of Epic Cards: {}
    '''.format(num_epic))

    st.write('''
    Number of Rare Cards: {}
    '''.format(num_rare))

    st.write('''
    Number of Common Cards: {}
    '''.format(num_common))

    st.markdown('##')
    eda(df)


    #TODO: combobox to select Rarity:
    subtitle(f'Rarity cards distribution based on Faction: ')
    with st.expander("Selection Rarity:", expanded=True):
        selection = st.multiselect("Opcion", df['Rarity'].unique() )
    if len(selection) > 0:
        legendary_df = df.loc[df['Rarity'].isin(selection) ]
        fig1 = plt.figure()
        ax = sns.countplot(data=legendary_df , x = 'Faction', order=legendary_df['Faction'].value_counts().index)
        plt.xticks(rotation=45)
        st.pyplot(fig1)


    subtitle('CountPlot: Unit Types')
    fig1 = plt.figure()
    g = sns.countplot(data=df , x = 'Unit Type', order=df['Unit Type'].value_counts().index)
    #plt.xticks(rotation=45)
    g.set_xticklabels(g.get_xticklabels(), rotation=90, fontsize=5)
    #g.set_yticklabels(g.get_yticklabels(), fontsize=5)

    st.pyplot(fig1)


    subtitle('MarkPoint: Strength vs Mana Cost')
    #fig2 = plt.figure()
    #sns.scatterplot(data=df , x = 'Mana' , y = 'Strength' , hue='Rarity')
    #st.pyplot(fig2)


    #st.subheader('Mark Point by Rarity')
    #brush = alt.selection_interval()
    #points = alt.Chart(df).mark_point().encode(
    #    x='Mana:Q',
    #    y='Strength:Q',
    #    #shape='Rarity',
    #    color=alt.condition(brush, 'Rarity:N', alt.value('lightgray'))
    #).add_selection(
    #    brush
    #)
    #bars = alt.Chart(df).mark_bar().encode(
    #    y='Rarity:N',
    #    color='Rarity:N',
    #    x='count(Rarity):Q'
    #).transform_filter(
    #    brush
    #)
    #st.altair_chart(points & bars)


    st.subheader('By Rarity (interactive)')
    brush = alt.selection_interval()
    points = alt.Chart(df).mark_point().encode(
        x='Mana:Q',
        y='Strength:Q',
        tooltip=['Mana','Strength','Faction'],
        #shape='Rarity',
        color=alt.condition(brush, 'Rarity:N', alt.value('lightgray'))
    ).add_selection(
        brush
    )



    bars = alt.Chart(df).mark_bar().encode(
        y='Rarity:N',
        #color='Rarity:N',
        #tooltip=['Faction', 'count(Faction)'],
        x='count(Rarity):Q'
    ).transform_filter(
        brush
    )

    text1 = bars.mark_text(
        align='left',
        baseline='middle',
        dx=10, # update
        color='white'

    ).transform_aggregate(
        count='count()',
        groupby=['Rarity']

    ).encode(
        x=alt.value(0), # update
        text=alt.Text('count:Q')
    )




    st.altair_chart(points & (bars + text1))




    st.subheader('By Faction (interactive)')
    brush = alt.selection_interval()
    points = alt.Chart(df).mark_point().encode(
        x='Mana:Q',
        y='Strength:Q',
        #shape='Rarity',
        tooltip=['Mana','Strength','Faction'],
        color=alt.condition(brush, 'Faction:N', alt.value('lightgray'))
    ).add_selection(
        brush
    )
    bars = alt.Chart(df).mark_bar().encode(
        y='Faction:N',
        color='Faction:N',
        x='count(Faction):Q',
        tooltip=['Faction', 'count(Faction)']
    ).transform_filter(
        brush
    )
    st.altair_chart(points & bars)


    st.subheader('Count of units types by Rarity')
    chart = alt.Chart(df).mark_bar().encode(
        x='Unit Type', #alt.X('Unit Type', bin=alt.Bin(maxbins=30)),
        y='count()',
        color='Rarity',
        tooltip=['Unit Type','Rarity','count()']
    )
    st.altair_chart(chart)


    st.subheader('Count of units types by Faction')
    chart = alt.Chart(df).mark_bar().encode(
        x='Unit Type', #alt.X('Unit Type', bin=alt.Bin(maxbins=30)),
        y='count()',
        color='Faction',
        tooltip=['Unit Type','Faction','count()']
    )
    st.altair_chart(chart)


    # Chart variando SHAPE
    #chart = alt.Chart(df).mark_point().encode(
    #    x='Mana',
    #    y='Strength',
    #    shape='Rarity',
    #    color='Faction'
    #)
    #st.altair_chart(chart)


    #subtitle('Correlation between features (Legendary only)')
    #fig3 = plt.figure()
    #sns.heatmap(legendary_df[['Mana','Strength','Move']].corr())
    #st.pyplot(fig3)



    #chart = alt.Chart(df).mark_tick().encode(
    #    x='Strength'
    #)
    #st.altair_chart(chart)



    subtitle('Heatmap: Correlation between features')
    fig3 = plt.figure()
    sns.heatmap(df[['Mana','Strength','Move', 'iRarity', 'iFaction', 'iCard']].corr(),  annot=True)
    st.pyplot(fig3)



    ##colors=[cmap(norm(i)) for i in df.Volume]
    #plt.figure(figsize=(8,6))
    ##title="Tree map"
    #plt.title(title, size=18)
    #squarify.plot(df['Unit Type'], label=df['Unit Type'],
    #            text_kwargs={'color':'blue', 'size':12} )
    #            #,color=colors)
    #plt.axis('off')
    #plt.savefig('fig.png')
    #plt.show()

    subtitle('Tree Map: Unit types')

    #a = titanic.groupby('class')[['survived']].sum().index.get_level_values(0).tolist()
    #print(a)
    #d = titanic.groupby('class')[['survived']].sum().reset_index().survived.values.tolist()
    #print(d)

    #l = df.groupby('Unit Type')[['Name']].count().index.get_level_values(0).tolist()
    #s = df.groupby('Unit Type')[['Name']].count().reset_index().Name.values.tolist()
    df1 =  df.groupby('Unit Type')[['Name']].count().sort_values(by=['Name'] , ascending=False)
    l = df1.index.get_level_values(0).tolist()
    s = df1.Name.values.tolist()
    st.write(df1 )
    fig, ax = plt.subplots()
    squarify.plot(s, label=l , text_kwargs={'color':'black', 'size':4},  alpha=.8 ) #,  pad=True ) 
    plt.axis('off')
    plt.show()
    st.pyplot(fig)




# -----------------------------------------------------------------------------
#if __name__ == '__main__':
#	main()
# -----------------------------------------------------------------------------