"""
--------------------------------------------
Titanic - Machine Learning from Disaster
Kaggle Competition
--------------------------------------------
0 Load your Data
1 Load and Explore your Data
2 Exploratory Data Analysis (EDA)
3 Making Decisions
4 Cleaning & Missing Data Handling
5 Feature Engineering **
6 Prepare our data for ML
7 Machine Learning (ML)
--------------------------------------------

*** Modo de Ejecucion ***
streamlit run titanic.py

*** From Heroku ***
TODO

v21.09 SEP 2022
Author: @VictorFrabasil
"""


# Data Analysis
from math import nan
from operator import index
import numpy as np
import pandas as pd
import random
import math

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

# ML Models
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Framework
import streamlit as st
import streamlit.components.v1 as components



# Global Configuration:
#pd.options.display.float_format = "{:.2f}".format
#pd.set_option('display.float_format', lambda x: '%.2f' % x)


# Setting Cache for dataset:
#@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_dataset():
    trainDf = pd.read_csv('./titanic/train.csv')
    testDf= pd.read_csv('./titanic/test.csv')
    return trainDf, testDf

# Display Function:
def sep():
    st.markdown('<hr style="height:2px;border-width:0;color:gray;background-color:green">', unsafe_allow_html=True)

def seps():
    st.sidebar.markdown('<hr style="height:2px;border-width:0;color:gray;background-color:green">', unsafe_allow_html=True)


# Main Function:
def app():
#def main():

    # 0 Load your Data --------------------------------------------
    train_data = pd.read_csv('./titanic/train.csv')
    test_data = pd.read_csv('./titanic/test.csv')
    datasets = [train_data, test_data]


    st.markdown("<h1 style='text-align: center; color: green;'>TITANIC PREDICTION</h1>", unsafe_allow_html=True)
    #st.sidebar.image(mylogo, width=256)
    page = st.sidebar.selectbox('Option:', ['Explore Data', 'Filtering Data', 'EDA', 'Feature Engineering', 'Cleaning',  'Machine Learning'])
    sep()


    if page == 'Explore Data':
        titanicExplore(train_data, test_data )    
    if page == 'Filtering Data':
        titanicFilter(train_data, test_data )            
    if page == 'EDA':
        titanicEDA(train_data, test_data ) 
    if page == 'Cleaning':
        titanicCleaning(train_data, test_data ) 
    if page == 'Feature Engineering':
        titanicFeature(train_data, test_data ) 
    if page == 'Machine Learning':
        titanicML(train_data, test_data ) 


# 1 Explore your Data --------------------------------------------
def titanicExplore(train_data, test_data ):
    #st.write('--------------------------------------------')
    st.write('##### ▶️ Train/Test SHAPE:')
    with st.echo():
        # Start viewing the shape of both DF:
        st.write(f' **Train Shape:** {train_data.shape}' )
        st.write(f' **Test Shape:** {test_data.shape}' )


    #st.write('##### Train/Test DTYPES:')
    #with st.echo():
    #    # variable types for all columns:
    #    st.write(f' {train_data.dtypes}'  )
    #    st.write(f' {test_data.dtypes}'  )


    #def wrap_by_word(s, n):
    #    a = s.split()
    #    ret = ''
    #    st.write(len(a))
    #    for i in range(0, len(a), n):
    #        ret += ' '.join(a[i:i+n]) + '\n'
    #    return ret
    #st.write(wrap_by_word(f'{test_data.dtypes}', 2) )


    #st.write('--------------------------------------------')
    st.write('##')
    st.write('##### ▶️ Train/Test HEAD data:')
    with st.echo():
        # Small sample of first rows:
        st.write(train_data.head())
        st.write(test_data.head())

    #st.write('--------------------------------------------')
    # this pandas function doesn’t return anything back to Python (which is why you get a value of None), it only prints
    print('Train/Test INFO:')
    print(train_data.info())
    print(test_data.info())

    #with st.echo():
    #    import io 
    #    buffer = io.StringIO() 
    #    train_data.info(buf=buffer)
    #    s = buffer.getvalue() 
    #    with open('df_info.txt', 'w', encoding='utf-8') as f:
    #        f.write(s) 
      
    st.write('##')
    st.write('##### ▶️ Train INFO data:')
    st.image('./titanic/info.png', width=400)

    #st.write('--------------------------------------------')
    st.write('##')
    st.write('##### ▶️ Train/Test DESCRIBE numerical data (Statistical Analysis):')
    if st.checkbox('Insights'):
        with st.echo():
            # Describing the numerical values:
            st.write(train_data.describe(exclude=['O']).T.style.background_gradient(subset=["max","min", "50%"],cmap='ocean_r'))
            st.write(test_data.describe(exclude=['O']).T.style.background_gradient(subset=["max", "min", "50%"],cmap='ocean_r'))
    else:
        with st.echo():
            # Describing the numerical values:
            st.write(train_data.describe(exclude=['O']))
            st.write(test_data.describe(exclude=['O']))



    #st.write('--------------------------------------------')
    #st.write('##')
    #st.write('##### Train/Test DESCRIBE categorical data:')
    #with st.echo():
    #    st.write(train_data.describe(include=['O']))
    #    st.write(test_data.describe(include=['O']))
    print(train_data.describe(include = 'object'))
    print(test_data.describe(include = 'object'))


    st.write('##')
    st.write('##### ▶️ Uniques values per column:')
    with st.echo():
        # Counting the uniques values of each column:
        col1, col2 = st.columns(2)
        with col1:
            st.write('*Train Set:*')
            st.write(train_data.nunique())
        with col2:
            st.write('*Test Set:*')
            st.write(test_data.nunique())


    def highlight(x):
        c1 = 'background-color: yellow'
        #empty DataFrame of styles
        df1 = pd.DataFrame('', index=x.index, columns=x.columns)
        #set new columns by condition
        df1.loc[(x['count'] > 0), 'count'] = c1
        return df1

    st.write('##')
    st.write('##### ▶️ Counting NAN rows:')    
    with st.echo():
        col1, col2 = st.columns(2)
        with col1:
            st.write('*Train Set:*')
            st.write(train_data.isnull().sum().to_frame().rename(columns={0: 'count'}).style.apply(highlight, axis=None) )
        with col2:
            st.write('*Test Set:*')
            st.write(test_data.isnull().sum().to_frame().rename(columns={0: 'count'}).style.apply(highlight, axis=None) )


    with st.echo():
        figsize=(5 ,4)
        figcorr, ax = plt.subplots()
        sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False,cmap='inferno')
        ax.set_title('Nan Detected in train data:')
        st.pyplot(figcorr)

    st.write('##')
    st.write('##### ▶️ Categorical Variables:')   

    st.write('###### 1 - SEX:')   
    with st.echo():
        st.write(train_data['Sex'].value_counts(dropna=False))
        
        survived_sex = train_data.groupby('Sex')['Survived'].mean() 
        ch = alt.Chart(survived_sex.reset_index()).mark_bar().encode(
            x=alt.X('Survived:Q', title='Survived Probability'),
            y='Sex:O',
            tooltip='Survived'
        )
        st.altair_chart(ch) # probability of survived by SEX

    st.write('###### 2 - EMBARKED:') 
    with st.echo():
        st.write(train_data['Embarked'].value_counts(dropna=False))

        survived_embarked = train_data.groupby('Embarked')['Survived'].mean() 
        ch = alt.Chart(survived_embarked.reset_index()).mark_bar().encode(
            x=alt.X('Survived:Q', title='Survived Probability'),
            y='Embarked:O',
            tooltip='Survived'
        )
        st.altair_chart(ch) # probability of survived by EMBARKED port



    st.write('##')
    st.write('##### ▶️ Continous Variables:')  

    st.write('###### 1 - AGE:') 
    with st.echo():
        ch = alt.Chart(train_data).mark_bar().encode(
            alt.X('Age:Q'), #, bin=alt.Bin(extent=[0, 100], step=5)),
            alt.Y('count()'), 
            tooltip='count():Q'
        )
        st.altair_chart(ch) # AGE Histogram

    with st.echo():
        # From describe() we know that age range is 0,42..80
        bins = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]
        age_range = pd.cut(train_data['Age'], bins = bins, labels = bins[:-1] ) # divide into groups
        survived_age = train_data.groupby(age_range)['Survived'].mean()
        ch = alt.Chart(survived_age.reset_index()).mark_bar().encode(
            x=alt.X('Survived:Q', title='Survived Probability'),
            y=alt.Y('Age:O', title='Age (step = 5)'),
            tooltip='Survived'
        )
        st.altair_chart(ch) # probability of survived by AGE group

    st.write('###### 2 - SIBSP:') 
    with st.echo():
        ch = alt.Chart(train_data).mark_bar().encode(
            alt.X('SibSp:O'),
            alt.Y('count()'), 
            tooltip='count():Q'
        )
        st.altair_chart(ch) # SibSp Count

    with st.echo():
        survived_SibSp = train_data.groupby('SibSp')['Survived'].mean()
        ch = alt.Chart(survived_SibSp.reset_index()).mark_bar().encode(
            x=alt.X('Survived:Q', title='Survived Probability'),
            y=alt.Y('SibSp:O', title='SibSp'),
            tooltip='Survived'
        )
        st.altair_chart(ch) # probability of survived by SibSp

    #st.write(train_data.loc[train_data['SibSp'] == 8] )


    st.write('###### 3 - PARCH:') 
    with st.echo():
        ch = alt.Chart(train_data).mark_bar().encode(
            alt.X('Parch:O'),
            alt.Y('count()'), 
            tooltip='count():Q'
        )
        st.altair_chart(ch) # Parch Count

    with st.echo():
        survived_Parch = train_data.groupby('Parch')['Survived'].mean()
        ch = alt.Chart(survived_Parch.reset_index()).mark_bar().encode(
            x=alt.X('Survived:Q', title='Survived Probability'),
            y=alt.Y('Parch:O', title='Parch'),
            tooltip='Survived'
        )
        st.altair_chart(ch) # probability of survived by Parch


    st.write('###### 4 - FARE:') 
    with st.echo():
        ch = alt.Chart(train_data).mark_bar().encode(
            alt.X('Fare:Q', bin=alt.Bin(extent=[0, 520], step=20)),
            alt.Y('count()'), 
            tooltip='count():Q'
        )
        st.altair_chart(ch) # Fare Histogram

    with st.echo():
        # From describe() we know that fare range is 0..512,32
        bins = [0,5,10,20,50,100,150,200,250,300,350,400,450,500,550]
        fare_range = pd.cut(train_data['Fare'], bins = bins, labels = bins[:-1] ) # divide into groups
        survived_fare = train_data.groupby(fare_range)['Survived'].mean()
        ch = alt.Chart(survived_fare.reset_index()).mark_bar().encode(
            x=alt.X('Survived:Q', title='Survived Probability'),
            y=alt.Y('Fare:O', title='Fare (step = 5)'),
            tooltip='Survived'
        )
        st.altair_chart(ch) # probability of survived by Fare group


    #st.write('##')
    #st.write('##### ▶️ Train NAN rows:')
    #with st.echo():
    #    # Highlight the NULLS:
    #    st.write(train_data[train_data.isnull().any(axis=1)].style.highlight_null('yellow'))
    ## Only columns
    ##st.write(train_data.loc[:, train_data.isna().any()])

    #st.write('##')
    #st.write('##### ▶️ Duplicated Rows:')
    #with st.echo():
    #    # Search Duplicates:
    #    st.write(train_data[train_data.duplicated(keep='last')])
    #    st.write(test_data[test_data.duplicated(keep='last')])



# 1.2 Exploratory Data Analysis (EDA) --------------------------------------------
def titanicFilter(train_data, test_data ):
    st.write('##')
    st.write('##### ▶️ Filtering to Analysis:')  


    ratioMinAge = train_data['Age'].min()
    ratioMaxAge = train_data['Age'].max()
    ratioMinFare = train_data['Fare'].min()
    ratioMaxFare = train_data['Fare'].max()

    with st.form(key='my_form1'):

        selMinAge, selMaxAge = st.slider( "Age Range", float(ratioMinAge), float(ratioMaxAge), (float(ratioMinAge), float(ratioMaxAge)) )

        selMinFare, selMaxFare = st.slider( "Fare Range", float(ratioMinFare), float(ratioMaxFare), (float(ratioMinFare), float(ratioMaxFare)) )

        with st.expander("📋 SibSp/Parch:", expanded=False):
            optionSib = st.multiselect("by SIBSP", options=train_data['SibSp'].unique() )
            optionPar = st.multiselect("by PARCH", options=train_data['Parch'].unique() )
            if len(optionSib) > 0:
                train_data = train_data.loc[train_data['SibSp'].isin(optionSib) ]
            if len(optionPar) > 0:
                train_data = train_data.loc[train_data['Parch'].isin(optionPar) ]

        with st.expander("📋 Embarked:", expanded=False):
            optionEmb = st.multiselect("Embarked:", options=train_data['Embarked'].unique() )
            if len(optionEmb) > 0:
                train_data = train_data.loc[train_data['Embarked'].isin(optionEmb) ]

        with st.expander("📋 Sex:", expanded=False):
            optionSex = st.multiselect("Sex:", options=train_data['Sex'].unique() )
            if len(optionSex) > 0:
                train_data = train_data.loc[train_data['Sex'].isin(optionSex) ]


        with st.expander("📋 Name:", expanded=False):
            myquery = st.text_input('Name Contains ...')
            if myquery:
                print(f'\'{myquery}\'')
                print(myquery)
                train_data = train_data[train_data['Name'].str.contains(pat=myquery, case=True)]

        with st.expander("📋 Ticket:", expanded=False):
            myqueryT = st.text_input('Ticket Contains ...')
            if myqueryT:
                train_data = train_data[train_data['Ticket'].str.contains(myqueryT, case=False)]

        with st.expander("📋 Cabin:", expanded=False):
            myqueryC = st.text_input('Cabin Contains ...')
            if myqueryC:
                train_data = train_data[train_data['Cabin'].str.contains(myqueryC, case=False, na=False)] 


        st.form_submit_button(label='Aplicar')

    train_data = train_data[(train_data['Age'] >= selMinAge) & (train_data['Age'] <= selMaxAge)]
    train_data = train_data[(train_data['Fare'] >= selMinFare) & (train_data['Fare'] <= selMaxFare)]

    st.write('##')
    st.write('##### ▶️ Result:') 
    st.write(train_data)


    sep()
    st.write('##')
    st.write('##### ▶️ Interactive Chart:')
    train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch']

    embDict = {'C' : 'Cherbourg','Q' : 'Queenstown', 'S' : 'Southampton' }
    survDict = {0 : 'Died' , 1 : 'Survived'}
    train_data_labeled = train_data.copy()
    train_data_labeled['Embarked'] = train_data_labeled['Embarked'].map(embDict)
    train_data_labeled['Survived'] = train_data_labeled['Survived'].map(survDict)


    yVar = st.radio("Select Y variable:", ('PassengerId', 'Fare'))


    with st.echo():
        brush = alt.selection_interval()
        points = alt.Chart(train_data_labeled).mark_circle(size=60, opacity=0.50).encode(
            alt.X('Age:Q' ),
            alt.Y(yVar ),
            #shape='Sex:N',
            color=alt.condition(brush, 'Survived:N', alt.value('lightgray')),
            tooltip=['PassengerId', 'Name', 'Pclass']
        ).add_selection(
            brush
        )
        
        bars = alt.Chart(train_data_labeled).mark_bar().encode(
            alt.Column('Embarked'),
            alt.Y('count(Survived)', axis=alt.Axis(grid=False) ), #,  stack='normalize' ),
            color='Survived:N',
            x='Sex:N',
            tooltip= ['Embarked', 'count(Sex)', 'count(Survived)']
        ).transform_filter(
            brush
        )

        st.altair_chart( points & bars)




# 2 Exploratory Data Analysis (EDA) --------------------------------------------
def titanicEDA(train_data, test_data ):

    st.write('##')
    st.write('##### ▶️ Target Variable:')
    with st.echo():
        st.write(train_data['Survived'].value_counts()) # show number of samples for each value
        st.write( f'MEAN: **{np.mean(train_data.Survived)}**' ) # an unbalanced dataset
        st.write( f'( %: {np.mean(train_data.Survived)* 100:.2f} Survived) ' )


    st.write('##')
    st.write('##### ▶️ Correlation Features:')
    heatCmap = sns.diverging_palette(20, 220, n=200)
    with st.echo():
        figsize=(10, 8)
        figcorr, ax = plt.subplots(figsize=figsize)
        sns.heatmap(train_data.corr(), annot=True, cmap=heatCmap);
        ax.set_title('Correlation Features')
        st.pyplot(figcorr)

    #cm = sns.color_palette('Blues', as_cmap=True)
    cm = sns.diverging_palette(20, 200, n=100, as_cmap=True)
    with st.expander('Insights :', expanded=False):
        df_corr = pd.DataFrame(train_data.corr().Survived)
        st.write( abs(df_corr).sort_values(by='Survived', ascending=False).style.background_gradient(cmap=cm))
        st.write('**PassengerId**  has no correlation with any other features.')
        st.write('**Pclass** is the most correlated numeric feature vs Survived ')
        st.write('**Pclass** and **Fare** have significant negative correlation value (0.55) *(high Fare -> first class)*')
        st.write('**SibSp** and **Parch** have significant positive correlation value (0.41) between them')


    st.write('##')
    st.write('##')
    st.write('##### ▶️ SEX Distribution:')
    # Create colors
    a, b =[plt.cm.Reds, plt.cm.Blues]
    with st.echo():
        sexDf = train_data['Sex'].groupby(train_data['Sex']).count()
        fig1, ax1 = plt.subplots()
        ax1.pie(sexDf, labels=sexDf, startangle=90, autopct='%1.1f%%', colors=[a(0.6),b(0.6)])  
        ax1.axis('equal') 
        ax1.title.set_text('Embarked Sex Distribution:') 
        plt.legend(sexDf.index)
        # draw center circle
        centre_circle = plt.Circle((0, 0), 0.4, fc='white')   
        fig1 = plt.gcf()
        fig1.gca().add_artist(centre_circle)
        st.pyplot(fig1)
       
    DrawDonnutSex(train_data)


    #ChSexPlot = alt.Chart(train_data).mark_bar(
    #    opacity=0.6,
    #).encode(
    #    alt.X('Sex:N'),   #, bin=alt.Bin(extent=[0, 100], step=5)),
    #    alt.Y('count()'), #, stack=None),
    #    alt.Color('Sex:N')
    #)
    #st.altair_chart(ChSexPlot)



    with st.expander('Insights :', expanded=False):
        #st.write('** Insight: **')
        Men            = train_data[train_data['Sex'] == 'male']['Sex'].count()
        survived_men   = train_data[(train_data['Sex'] == 'male') & (train_data['Survived'] == 1)]['Sex'].count()
        Women          = train_data[train_data['Sex'] == 'female']['Sex'].count()
        survived_women = train_data[(train_data['Sex'] == 'female') & (train_data['Survived'] == 1)]['Sex'].count()
        st.write(train_data.groupby(by='Sex')['Survived'].value_counts())
        st.write(f'Male survived: {survived_men} from {Men} ( **{ str(round(100*survived_men / Men, 2))}** % )' )
        st.write(f'Female survived: {survived_women} from {Women} ( **{ str(round(100*survived_women / Women, 2))}** % )' )



    st.write('##')
    st.write('##')
    st.write('##### ▶️ EMBARKED: How many embarked in each port and survived:')
    DrawDonnutEmb(train_data)
    with st.expander('Embarked Detail :', expanded=False):
        embDict = {'C' : 'Cherbourg','Q' : 'Queenstown', 'S' : 'Southampton' }
        survDict = {0 : 'Died' , 1 : 'Survived'}
        train_data_labels = train_data.copy()
        train_data_labels['Embarked'] = train_data_labels['Embarked'].map(embDict)
        train_data_labels['Survived'] = train_data_labels['Survived'].map(survDict)

        DrawEmb(train_data_labels)
        st.write('EMBARKED by Sex:')
        DrawEmbComposite(train_data_labels, 'Sex:O', 'count()', 'Sex', 'zero')
        st.write('EMBARKED by Survived:')
        DrawEmbComposite(train_data_labels, 'Survived:O', 'count()', 'Survived', 'zero')
        st.write('EMBARKED Composite:')
        DrawEmbComposite(train_data_labels, 'Sex:O', 'count(Survived)', 'Survived', 'normalize')

        st.write('##')
        st.write(train_data.groupby(by='Sex')['Survived'].value_counts())

        sMen           = train_data[(train_data['Sex'] == 'male') & (train_data['Embarked'] == 'S')]['Sex'].count()
        sSurvived_men  = train_data[(train_data['Sex'] == 'male') & (train_data['Embarked'] == 'S')  & (train_data['Survived'] == 1)]['Sex'].count()
        st.write(f'Men survived embarked in Southampton only {sSurvived_men} from {sMen} ( **{ str(round(100*sSurvived_men / sMen, 2))}** % )' )

        sMen           = train_data[(train_data['Sex'] == 'male') & (train_data['Embarked'] == 'Q')]['Sex'].count()
        sSurvived_men  = train_data[(train_data['Sex'] == 'male') & (train_data['Embarked'] == 'Q')  & (train_data['Survived'] == 1)]['Sex'].count()
        st.write(f'Men survived embarked in Queenstown only {sSurvived_men} from {sMen} ( **{ str(round(100*sSurvived_men / sMen, 2))}** % )' )

        sMen           = train_data[(train_data['Sex'] == 'male') & (train_data['Embarked'] == 'C')]['Sex'].count()
        sSurvived_men  = train_data[(train_data['Sex'] == 'male') & (train_data['Embarked'] == 'C')  & (train_data['Survived'] == 1)]['Sex'].count()
        st.write(f'Men survived embarked in Cherbourg only {sSurvived_men} from {sMen} ( **{ str(round(100*sSurvived_men / sMen, 2))}** % )' )


    st.write('##')
    st.write('##')
    st.write('##### ▶️ PCLASS: How many survived in each class:')
    DrawDonnutPclass(train_data)
    with st.expander('Insights :', expanded=False):
        not_survived = train_data[(train_data['Pclass'] == 3) & (train_data['Survived'] == 0)]['Pclass'].count()
        totalP3 = train_data[ train_data['Pclass'] == 3 ]['Pclass'].count()
        st.write(train_data.groupby(by='Pclass')['Survived'].value_counts())
        st.write(f'PClass **3** passenger died: {not_survived} of total { totalP3 } (**{not_survived/totalP3 * 100:.2f} **) %' )



    st.write('##')
    st.write('##')
    st.write('##### ▶️ Survived status based in diferent features:')
    start = ','
    end = '.'
    cm = sns.light_palette("seagreen", reverse=False,  as_cmap=True)
    st.write('** Parch: ** (parents travelled with + number of childrens)')
    with st.echo():
        st.write(train_data[ ['Parch', 'Survived'] ].groupby('Parch', as_index=False).mean()
                                                    .sort_values(by='Survived', ascending=False)
                                                    .style.background_gradient(cmap=cm, subset=['Survived']) )
    st.write('** SibSp: ** (sisters + brothers + spouse)')
    with st.echo():
        st.write(train_data[ ['SibSp', 'Survived'] ].groupby('SibSp', as_index=False).mean()
                                                    .sort_values(by='Survived', ascending=False)
                                                    .style.background_gradient(cmap=cm, subset=['Survived']) )

    st.write('** Title: ** (extracted from Name column)')
    #st.write(train_data[ ['Title', 'Survived'] ].groupby('Title', as_index=False)
    #                                            .mean()
    #                                            .sort_values(by='Survived', ascending=False) )
    #st.write(train_data[ ['Title', 'Survived'] ].groupby('Title', as_index=False)
    #                                            .agg({'Title': 'count', 'Survived':'mean'})
    #                                            .sort_values(by='Survived', ascending=False) )
    with st.echo():
        train_data['Title']=train_data['Name'].map(lambda s: s[s.find(start)+len(start):s.find(end)] )
        st.write(train_data[ ['Title', 'Survived'] ].groupby('Title', as_index=False)
                                                .agg( Count=pd.NamedAgg(column='Title', aggfunc='count'),
                                                      SurvivedMean=pd.NamedAgg(column='Survived', aggfunc='mean'))
                                                .sort_values(by='SurvivedMean', ascending=False)
                                                .style.background_gradient(cmap=cm, subset=['SurvivedMean']) )





# 3 Data Cleaning --------------------------------------------
def titanicCleaning(train, test ):

    train_data = train.copy()
    test_data = test.copy()

    #st.write('##### 0 - Feature Eng.:')
    def findString(source, substrings):
        for substring in substrings:
            if source.find(substring) != -1:
                return substring
        return np.nan

    # Title: new column from Name
    start = ','
    end = '.'
    train_data['Title']=train_data['Name'].map(lambda s: s[s.find(start)+len(start):s.find(end)] )
    test_data['Title']=test_data['Name'].map(lambda s: s[s.find(start)+len(start):s.find(end)] )


    # Deck: new column from Cabin
    train_data['Cabin'] = train_data['Cabin'].fillna("Unknown")
    test_data['Cabin'] = test_data['Cabin'].fillna("Unknown")
    cabins = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'U']
    train_data['Deck'] = train_data['Cabin'].astype(str).map(lambda x: findString(x, cabins))
    test_data['Deck'] = test_data['Cabin'].astype(str).map(lambda x: findString(x, cabins))


    st.write('##### 1 - Drop useless variables:')
    st.write('We have seen that PASSENGERID variable has a small correlation with the target variable so we drop it')
    with st.echo():
        train_data = train_data.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1)
        test_data = test_data.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1)


    st.write('##### 2 - Missing data:')

    with st.echo():
        st.write(train_data['Title'].value_counts())
        st.write(train_data.groupby('Title')['Age'].mean())

        
    #st.write(train_data.groupby('Title')['Age'].mean())
    #st.write(train_data.groupby('Title')['Age'].std())

    #ageMean = train_data.groupby('Title')['Age'].mean()
    #ageStd  = train_data.groupby('Title')['Age'].std()
    #ageNull = train_data["Age"].isnull().sum()
    #ageStd = ageStd.fillna(1)
    #ageStd = ageStd.replace(to_replace = 0, value = 1)
    #st.write(ageStd)
    #st.write(ageMean)
    #st.write(ageStd)
    #st.write(ageNull)

    ## Random numbers:  mean +/- std 
    #rand_age = np.random.randint(ageMean - ageStd, ageMean + ageStd, size = ageNull)
    #st.write(rand_age)

    ## fill NaN values in Age column with random values generated
    #age_slice = train_data["Age"].copy()
    #age_slice[np.isnan(age_slice)] = rand_age
    #train_data["Age"] = age_slice
    #train_data["Age"] = train_data["Age"].astype(int)
    #st.write(train_data.head(20))

    #train_data["Age"] = train_data["Age"].fillna(0)
    #train_data["Age"] = train_data["Age"].astype(int)
    #train_data["Age"] = train_data["Age"].fillna(np.random.randint(ageMean - ageStd, ageMean + ageStd) )

    st.write('##')
    st.write('create a dictionary table based on MEAN and STD of tittles:')
    with st.echo():
        ageMeanDict = train_data.groupby('Title')['Age'].mean().to_dict()
        ageStdDict = train_data.groupby('Title')['Age'].std().to_dict()
        for k,v in ageStdDict.items():
            if math.isnan(v) or v == 0:
                ageStdDict[k] = 1.0
        for k,v in ageMeanDict.items():
            ageMeanDict[k] = np.random.randint(ageMeanDict[k] - ageStdDict[k], ageMeanDict[k] + ageStdDict[k])
        st.write(ageMeanDict)

    ##train_data.loc[ train_data["Age"].isnull(), "Age"] = train_data["Title"].map(ageMeanDict)

    #train_data.loc[ train_data["Age"].isnull(), "Age"] = np.random.randint(train_data["Title"].map(ageMeanDict) - train_data["Title"].map(ageStdDict), 
    #                                                     train_data["Title"].map(ageMeanDict) + train_data["Title"].map(ageStdDict), size=1)


    train_data.loc[ train_data["Age"].isnull(), "Age"] = train_data["Title"].map(ageMeanDict)
    # Falta el +- ageStdDict

    #train_data.loc[ train_data["Age"].isnull(), "Age"] = np.random.randint(ageMean - ageStd, ageMean + ageStd)

    st.write('MAPPING:')
    with st.echo():
        st.write(train_data['Embarked'].value_counts(dropna=False))
        st.write(train_data['Embarked'].value_counts(dropna=False, normalize=True))   
        
        embDict = {'C' : 0, 'Q' : 1, 'S' : 2 }
        train_data['Embarked'] = train_data['Embarked'].fillna('S')  #falta el puerto mas popular
        train_data['Embarked'] = train_data['Embarked'].map(embDict).astype(int)

        sexDict = {'male' : 0 , 'female' : 1}
        train_data['Sex'] = train_data['Sex'].map(sexDict)


    st.write('ROUND FLOAT TO INT VALUES:')
    with st.echo():
        train_data['Age'] =  train_data['Age'].round(0).astype(int)
        train_data['Fare'] =  train_data['Fare'].round(0).astype(int)

    st.write('Values count:')
    st.write(train_data['Deck'].value_counts(dropna=False))





    
    st.write('##### 3 - Split:')
    # Split 1: into dependend and independent variables
    with st.echo():
        X = train_data.drop('Survived', axis=1)
        y = train_data['Survived']

    # Split 2: X into continous variables and categorical variables
    with st.echo():
        X_continous  = X[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]
        X_categorical = X[['Title', 'Sex', 'Deck', 'Embarked']]

    st.write('##### 4 - Dummies:')




    with st.echo():
        X_encoded = pd.get_dummies(X_categorical)
        X = pd.concat([X_continous, X_encoded],axis=1) # Concatenate both sets

    st.write('##### 4 - Final Dataframe for ML:')
    st.write(X.sample(10))
    st.write(X.shape)

# 4 Feaure Eng. --------------------------------------------
def titanicFeature(train_data, test_data ):

    df = train_data.copy()
    #a = 'Different {} Grouped_Value'.format('Age')
    a = 'AgeGroup'

    st.write('##')
    st.write('##')
    st.write('##### ▶️Create new columns based on features:')
    with st.echo():
        # AgeGroup: New column from Age
        intervals = [0, 1, 4, 9, 15, 19, 29, 44, 60, 80]
        labels = ['Infant', 'Toddler', 'Child', 'Teenager', 'Young Adult', 'Adult', 'Mid-Age', 'Middle Senior', 'Elderly']
        df[a] = pd.cut(x = df['Age'], bins = intervals, labels = labels, include_lowest=True)

    with st.echo():
        chAge = alt.Chart(df).mark_bar().encode(
            alt.X('AgeGroup', sort=alt.EncodingSortField(field='Age') ),
            alt.Y('count()'), 
            alt.Color('AgeGroup'),
            tooltip='count()'
        )
        st.altair_chart(chAge)

    def findString(source, substrings):
        for substring in substrings:
            if source.find(substring) != -1:
                return substring
        return np.nan


    # the Name column is not being used,
    # we can use it to extract the title from the name.
    with st.echo():
        # Title: new column from Name
        start = ','
        end = '.'
        df['Title']=df['Name'].map(lambda s: s[s.find(start)+len(start):s.find(end)] )
    
    with st.echo():
        chTitle = alt.Chart(df).mark_bar().encode(
            alt.X('Title'),
            alt.Y('count()'), 
            alt.Color('Title'),
            tooltip=['Title', 'count()']
        )
        st.altair_chart(chTitle)




    # Other way:
    # titleList=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
    #             'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess', 'Don', 'Jonkheer']
    # df['Title']=df['Name'].map(lambda x: findString(x, titleList))

    
    #replacing all titles with mr, mrs, miss, master
    #def replace_titles(x):
    #    title=x['Title']
    #    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
    #        return 'Mr'
    #    elif title in ['Countess', 'Mme']:
    #        return 'Mrs'
    #    elif title in ['Mlle', 'Ms']:
    #        return 'Miss'
    #    elif title =='Dr':
    #        if x['Sex']=='Male':
    #            return 'Mr'
    #        else:
    #            return 'Mrs'
    #    else:
    #        return title
    #df['Title']=df.apply(replace_titles, axis=1)


    # Only 1st class passengers have cabins the rest are 'Unknown'. 
    # Cabin number looks like 'Cnnn', the first letter refers to the deck
    # Turning cabin number into Deck
    with st.echo():
        # Deck: new column from Cabin
        cabins = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
        df['Deck'] = df['Cabin'].astype(str).map(lambda x: findString(x, cabins))

    with st.echo():
        chDeck = alt.Chart(df).mark_bar().encode(
            alt.X('Deck'),
            alt.Y('count()'), 
            alt.Color('Deck'),
            tooltip=['Deck', 'count()']
        )
        st.altair_chart(chDeck)


    # People traveling alone did better or worse?
    # Using linear combinations of features.
    # In decision tree is hard to model this relationships. 
    # In alinear regression maybe unnecessary.
    with st.echo():
        # FamilySize: new column from SibSp and Parch
        df['FamilySize']=df['SibSp']+df['Parch']

    with st.echo():
        chTitle = alt.Chart(df).mark_bar().encode(
            alt.X('FamilySize:O'),
            alt.Y('count()'), 
            alt.Color('FamilySize:N'),
            tooltip=['FamilySize', 'count()']
        )
        st.altair_chart(chTitle)


    with st.echo():
        #FareByPerson: new column from Fare and FamilySize
        df['FareByPerson']=df['Fare']/(df['FamilySize']+1)

    with st.expander('Insights :', expanded=False):
        st.write(f'Nulls value in **AGE** Feaure: {df.Age.isna().sum()} ({df.Age.isna().sum() / df.Age.count() * 100:.2f} %) ' )
        most = df[(df[a] == 'Adult' ) | (df[a] == 'Mid-Age')]['Age'].count()
        st.write(f'Adults and Mid-Age represent: **{most / (df.Age.count() + df.Age.isna().sum()) * 100:.2f}** % ')
        st.write(f'Cabins is only assigned to passenger of Pclass **1** and the most assigned Deck is the **C**')
        st.write(f'Most of the passenger travel alone (Family size of **0**)')

        st.write('##')
        st.write('Sample after Feature Eng.: ')
        st.write(df.sample(10))



    #st.write('##### ▶️ Correlation Features II (after Feature Eng.):')
    #heatCmap = sns.diverging_palette(20, 220, n=200)
    #with st.echo():
    #    figsize=(10, 8)
    #    figcorr, ax = plt.subplots(figsize=figsize)
    #    sns.heatmap(df.corr(), annot=True, cmap=heatCmap);
    #    ax.set_title('Correlation Features')
    #    st.pyplot(figcorr)



# 5 ML Models --------------------------------------------
def titanicML(train_data, test_data ):
    st.write('##')
    st.write('##### ▶️ML Models test:')    

    train_data = train_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    test_data = test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    embDict = {'C' : '0','Q' : '1', 'S' : '2' }
    sexDict = {'female' : '0','male' : '1' }

    train_data['Embarked'] = train_data['Embarked'].map(embDict)
    train_data['Sex'] = train_data['Sex'].map(sexDict)
    test_data['Embarked'] = test_data['Embarked'].map(embDict)
    test_data['Sex'] = test_data['Sex'].map(sexDict)


    # Handling missing values
    train_data.loc[train_data['Embarked'].isnull(), 'Embarked'] = '2'
    train_data.loc[train_data['Fare'].isnull(), 'Fare'] = train_data['Fare'].median()
    train_data.loc[train_data['Age'].isnull(), 'Age'] = train_data['Age'].median()

    test_data.loc[test_data['Embarked'].isnull(), 'Embarked'] = '2'
    test_data.loc[test_data['Fare'].isnull(), 'Fare'] = test_data['Fare'].median()
    test_data.loc[test_data['Age'].isnull(), 'Age'] = test_data['Age'].median()
    test_data['Fare'] = test_data['Fare'].astype(int)

    def highlight(x):
        c1 = 'background-color: yellow'
        #empty DataFrame of styles
        df1 = pd.DataFrame('', index=x.index, columns=x.columns)
        #set new columns by condition
        df1.loc[(x['count'] > 0), 'count'] = c1
        return df1


    with st.echo():
        st.write(train_data.isnull().sum().to_frame().rename(columns={0: 'count'}).style.apply(highlight, axis=None) )

    with st.echo():
        #Model preparation:
        X = train_data.drop(['Survived'], axis=1)
        y = train_data['Survived']
        X_test  = test_data.copy()


    st.subheader('Logistic Regression')
    with st.echo():
        log_model = LogisticRegression()
        log_model.fit(X, y)
        log_model_score = round(log_model.score(X, y) * 100, 2)
        st.write('%' + str(log_model_score))

    st.subheader('DecisionTreeClassifier')
    with st.echo():
        tree_model = DecisionTreeClassifier()
        tree_model.fit(X, y)
        tree_model_score = round(tree_model.score(X, y) * 100, 2)
        st.write('%' + str(tree_model_score))


        #forest_model = RandomForestClassifier()
        #forest_model.fit(X, y)
        #Y_prediction = forest_model.predict(X_test)
        #forest_model_score = round(forest_model.score(X, y) * 100, 2)
        #st.write('%' + str(forest_model_score))
        #st.write('Prediction:' + str(Y_prediction))
        
    st.subheader('RandomForestClassifier')
    with st.echo():
        random_forest = RandomForestClassifier(n_estimators=100)        
        random_forest.fit(X, y)
        Y_prediction = random_forest.predict(X_test)
        random_forest.score(X, y)
        acc_random_forest = round(random_forest.score(X, y) * 100, 2)
        st.write('%' + str(acc_random_forest))

    st.subheader('SVC')
    with st.echo():
        SVC_model = SVC()
        SVC_model.fit(X, y)
        SVC_model_score = round(SVC_model.score(X, y) * 100, 2)
        st.write('%' + str(SVC_model_score))

    st.subheader('LinearSVC')
    with st.echo():
        Linear_SVC_model = LinearSVC()
        Linear_SVC_model.fit(X, y)
        Linear_SVC_model_score = round(Linear_SVC_model.score(X, y) * 100, 2)
        st.write('%' + str(Linear_SVC_model_score))

    st.subheader('GaussianNB')
    with st.echo():
        nb_model = GaussianNB()
        nb_model.fit(X, y)
        nb_model_score = round(nb_model.score(X, y) * 100, 2)
        st.write('%' + str(nb_model_score))

    st.subheader('KNeighborsClassifier')
    with st.echo():
        knn_model = KNeighborsClassifier()
        knn_model.fit(X, y)
        knn_model_score = round(knn_model.score(X, y) * 100, 2)
        st.write('%' + str(knn_model_score))

    st.subheader('Perceptron')
    with st.echo():
        Perceptron_model = Perceptron()
        Perceptron_model.fit(X, y)
        Perceptron_model_score = round(Perceptron_model.score(X, y) * 100, 2)
        st.write('%' + str(Perceptron_model_score))

    st.subheader('SGDClassifier')
    with st.echo():
        SGD_model = SGDClassifier()
        SGD_model.fit(X, y)
        SGD_model_score = round(SGD_model.score(X, y) * 100, 2)
        st.write('%' + str(SGD_model_score))

# Drawing Functions  --------------------------------------------
def DrawDonnutEmb(train_data):
    group_names    = ['Southampton', 'Cherbourg', 'Queenstown']
    group_size     = list(train_data['Embarked'].value_counts())
    subgroup_names = ['Died', 'Survived']*3
    subgroup_size  = list(train_data.groupby(by='Embarked')['Survived'].value_counts())

    # Create colors
    a, b, c=[plt.cm.Blues, plt.cm.Reds, plt.cm.Purples]

    fig, ax = plt.subplots(figsize=(7,10))
    ax.axis('equal')

    plt.title('Percentage of survival \nbased on Embarked Port', fontsize = 20)

    # First Ring (outside)
    plt.pie(group_size, radius=1.5, labels=group_names, 
            colors=[a(0.6), b(0.6), c(0.6)], startangle=90)

    # Second Ring (Inside)
    plt.pie(subgroup_size, radius=1.3-0.3, 
                    labels=subgroup_names, labeldistance=1, 
                    colors=[b(0.5), b(0.4), c(0.5), c(0.4), a(0.5), a(0.4)], startangle=-9,
                    autopct = '%.2f%%', rotatelabels=True, pctdistance=0.75)

    # draw circle
    centre_circle = plt.Circle((0, 0), 0.5, fc='white')
    fig = plt.gcf()
    
    # Adding Circle in Pie chart
    fig.gca().add_artist(centre_circle)
    plt.legend(group_names, loc='lower left')
    st.pyplot(fig)
    #plt.show()


def DrawDonnutSex(train_data):
    group_names    = train_data['Sex'].unique()
    group_size     = list(train_data['Sex'].value_counts())
    subgroup_names = ['Died', 'Survived']*2
    subgroup_size  = list(train_data.groupby(by=['Sex','Survived'])['Survived'].value_counts())

    # Create colors
    a, b =[plt.cm.Blues, plt.cm.Reds ]
    fig, ax = plt.subplots(figsize=(7,10))
    ax.axis('equal')
    plt.title('Percentage of survival \nbased on passengers sex', fontsize = 20)

    # First Ring (outside)
    plt.pie(group_size, radius=1.5, labels=group_size, 
            colors=[a(0.6), b(0.6)], startangle=90)

    # Second Ring (Inside)
    plt.pie(subgroup_size, radius=1.3-0.3, 
                    labels=subgroup_names, labeldistance=1, 
                    colors=[b(0.5), b(0.4), a(0.5), a(0.4)], startangle=-37,
                    autopct = '%.2f%%', rotatelabels=True, pctdistance=0.75)

    # draw circle
    centre_circle = plt.Circle((0, 0), 0.5, fc='white')
    fig = plt.gcf()
    
    # Adding Circle in Pie chart
    fig.gca().add_artist(centre_circle)
    plt.legend(group_names, loc='lower left')
    st.pyplot(fig)
    #plt.show()


def DrawDonnutPclass(train_data):
    group_names    = train_data['Pclass'].unique()
    group_size     = list(train_data['Pclass'].value_counts())
    subgroup_names = ['Died', 'Survived']*3
    subgroup_size  = list(train_data.groupby(by=['Pclass','Survived'])['Survived'].value_counts())

    # Create colors
    a, b, c =[plt.cm.Blues, plt.cm.Reds, plt.cm.Greens ]
    fig, ax = plt.subplots(figsize=(7,10))
    ax.axis('equal')
    plt.title('Percentage of survival \nbased on Pclass', fontsize = 20)

    # First Ring (outside)
    plt.pie(group_size, radius=1.5, labels=group_size, 
            colors=[a(0.6), b(0.6), c(0.6)], startangle=90)

    # Second Ring (Inside)
    plt.pie(subgroup_size, radius=1.3-0.3, 
                    labels=subgroup_names, labeldistance=1, 
                    colors=[b(0.5), b(0.4), c(0.5), c(0.4),  a(0.5), a(0.4)], startangle=-71.5,
                    autopct = '%.2f%%', rotatelabels=True, pctdistance=0.75)

    # draw circle
    centre_circle = plt.Circle((0, 0), 0.5, fc='white')
    fig = plt.gcf()
    
    # Adding Circle in Pie chart
    fig.gca().add_artist(centre_circle)
    plt.legend(group_names, loc='lower left', title='Pclass:')
    st.pyplot(fig)
    #plt.show()



def DrawEmb(train_data_labels):
    with st.echo():
        chEmb = alt.Chart(train_data_labels).mark_bar().encode(
            alt.X('Embarked'),
            alt.Y('count()'), 
            tooltip='count()'
        ).properties(
            height=400,
            width=200
        )   
        st.altair_chart(chEmb)

#def DrawEmbSex(train_data):
#    with st.echo():
#        #Embarked by sex
#        chEmbSex = alt.Chart(train_data).mark_bar().encode(
#            alt.Column('Embarked'), 
#            alt.X('Sex'),
#            alt.Y('count()', axis=alt.Axis(grid=False)), 
#            alt.Color('Sex'),
#            tooltip='count()'
#        )
#        st.altair_chart(chEmbSex)
#
#def DrawEmbSurv(train_data_labels):
#    with st.echo():
#        # Embarked by survived
#        chEmbSurv = alt.Chart(train_data_labels).mark_bar().encode(
#            alt.Column('Embarked'), 
#            alt.Color('Survived'),
#            alt.X('Survived:O'),
#            alt.Y('count()', axis=alt.Axis(grid=False)), 
#            tooltip='count()'
#        )
#        st.altair_chart(chEmbSurv)
#
#def DrawEmbCompo(train_data_labels):
#    with st.echo():
#        # Embarked Composite Chart
#        chEmbCompo = alt.Chart(train_data_labels).mark_bar().encode(
#            alt.Column('Embarked'), 
#            alt.X('Sex:O'),
#            alt.Y('count(Survived)', axis=alt.Axis(grid=False),  stack='normalize' ),
#            alt.Color('Survived'),
#            tooltip='count()'
#        )
#        st.altair_chart(chEmbCompo)




def DrawEmbComposite(train_data_labels, x, y, c, s):
    # Embarked Composite Chart
    chEmbCompo = alt.Chart(train_data_labels).mark_bar().encode(
        alt.Column('Embarked'), 
        alt.X(x),
        alt.Y(y, axis=alt.Axis(grid=False),  stack=s ),
        alt.Color(c),
        tooltip='count()'
    )
    st.altair_chart(chEmbCompo)

# -----------------------------------------------------------------------------
#if __name__ == '__main__':
#	main()
# -----------------------------------------------------------------------------


