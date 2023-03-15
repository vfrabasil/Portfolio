  
"""
*** Streamlit HeapMap Data Explorer ***
App to analyze the transaction data obtained on two different days and evaluate the variation in the values.

*** Modo de Ejecucion ***
streamlit run ner2.py

*** From Heroku ***
https://link-heatmap.herokuapp.com/

v21.08 August 2022
Author: @VictorFrabasil
"""



from typing import List, Sequence, Tuple, Optional

import pandas as pd
import streamlit as st
import spacy
from spacy import displacy


from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

NER_ATTRS = ["text", "label_", "start", "end", "start_char", "end_char"]


def sep():
    st.markdown('<hr style="height:2px;border-width:0;color:gray;background-color:green">', unsafe_allow_html=True)

def seps():
    st.sidebar.markdown('<hr style="height:2px;border-width:0;color:gray;background-color:green">', unsafe_allow_html=True)


def get_html(html: str):
    """Convert HTML so it can be rendered."""
    WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
    # Newlines seem to mess with the rendering
    html = html.replace("\n", " ")
    return WRAPPER.format(html)

def visualize_ner(
    doc: spacy.tokens.Doc,
    *,
    labels: Sequence[str] = tuple(),
    attrs: List[str] = NER_ATTRS,
    show_table: bool = True,
    title: Optional[str] = "Named Entities",
    sidebar_title: Optional[str] = "Named Entities",
    key=None,  # add key as parameter
) -> None:
    """Visualizer for named entities."""
    seps()
    if title:
        st.header(title)
    if sidebar_title:
        st.sidebar.header(sidebar_title)
    label_select = st.sidebar.multiselect(
        "Entity labels selection:", options=labels, default=list(labels), key=key # add key now
    )
    seps()


    colors = {"PERSON": "linear-gradient(90deg, #aa9cfc, #fc9ce7)"}
    #colors ={"DATE": "linear-gradient(90deg, #ca9cde, #ff9ce7)", "PERSON": "radial-gradient(white,red)" }
    options = {"ents": label_select, "colors": colors}
    #options={"ents": label_select}
    #displacy.serve(doc, style="ent", options=options)



    html = displacy.render(doc, style="ent", options=options)
    style = "<style>mark.entity { display: inline-block }</style>"
    st.write(f"{style}{get_html(html)}", unsafe_allow_html=True)
    if show_table:
        data = [
            [str(getattr(ent, attr)) for attr in attrs]
            for ent in doc.ents
            if ent.label_ in labels
        ]
        df = pd.DataFrame(data, columns=attrs)
        st.dataframe(df)


def app():
#def main():

    nlp = spacy.load("en_core_web_sm")
    #nlp = spacy.load("es_core_news_sm")
    sampleEs = "En la parte inferior del escalón, hacia la derecha, vi una pequeña esfera tornasolada, de casi intolerable fulgor. Al principio la creí giratoria; luego comprendí que ese movimiento era una ilusión producida por los vertiginosos espectáculos que encerraba. El diámetro del Aleph sería de dos o tres centímetros, pero el espacio cósmico estaba ahí, sin disminución de tamaño. Cada cosa (la luna del espejo, digamos) era infinitas cosas, porque yo claramente la veía desde todos los puntos del universo. Vi el populoso mar, vi el alba y la tarde, vi las muchedumbres de América, vi una plateada telaraña en el centro de una negra pirámide, vi un laberinto roto (era Londres), vi interminables ojos inmediatos escrutándose en mí como en un espejo, vi todos los espejos del planeta y ninguno me reflejó, vi en un traspatio de la calle Soler las mismas baldosas que hace treinta años vi en el zaguán de una casa en Frey Bentos, vi racimos, nieve, tabaco, vetas de metal, vapor de agua, vi convexos desiertos ecuatoriales y cada uno de sus granos de arena, vi en Inverness a una mujer que no olvidaré, vi la violenta cabellera, el altivo cuerpo, vi un cáncer de pecho, vi un círculo de tierra seca en una vereda, donde antes hubo un árbol, vi una quinta de Adrogué, un ejemplar de la primera versión inglesa de Plinio, la de Philemont Holland, vi a un tiempo cada letra de cada página (de chico yo solía maravillarme de que las letras de un volumen cerrado no se mezclaran y perdieran en el decurso de la noche), vi la noche y el día contemporáneo, vi un poniente en Querétaro que parecía reflejar el color de una rosa en Bengala, vi mi dormitorio sin nadie, vi en un gabinete de Alkmaar un globo terráqueo entre dos espejos que lo multiplicaban sin fin, vi caballos de crin arremolinada, en una playa del Mar Caspio en el alba, vi la delicada osadura de una mano, vi a los sobrevivientes de una batalla, enviando tarjetas postales, vi en un escaparate de Mirzapur una baraja española, vi las sombras oblicuas de unos helechos en el suelo de un invernáculo, vi tigres, émbolos, bisontes, marejadas y ejércitos, vi todas las hormigas que hay en la tierra, vi un astrolabio persa, vi en un cajón del escritorio (y la letra me hizo temblar) cartas obscenas, increíbles, precisas, que Beatriz había dirigido a Carlos Argentino, vi un adorado monumento en la Chacarita, vi la reliquia atroz de lo que deliciosamente había sido Beatriz Viterbo, vi la circulación de mi propia sangre, vi el engranaje del amor y la modificación de la muerte, vi el Aleph, desde todos los puntos, vi en el Aleph la tierra, vi mi cara y mis vísceras, vi tu cara, y sentí vértigo y lloré, porque mis ojos habían visto ese objeto secreto y conjetural, cuyo nombre usurpan los hombres, pero que ningún hombre ha mirado: el inconcebible universo."
    sampleE2 = "According to Greek mythology, humans were originally created with four arms, four legs and a head with two faces. \
    Fearing their power, Zeus split them into two separate parts, condemning them to spend their lives in search of their other halves.\
    ... and when one of them meets the other half, the actual half of himself, whether he be a lover of youth or a lover of another \
    sort, the pair are lost in an amazement of love and friendship and intimacy and one will not be out of the other's sight, as I may say, even for a moment. \
    Plato, The Symposium"
    sampleEn4 = "At my age, one should be aware of one's limits, and this knowledge may make for happiness. When I was young, I thought of literature as a game of skillful and surprising variations; now that I have found my own voice, I feel that tinkering and tampering neither greatly improve nor greatly spoil my drafts. This, of course, is a sin against one of the main tendencies of letters in this century--the vanity of overwriting-- ... I suppose my best work is over. This gives me a certain quiet satisfaction and ease. And yet I do not feel I have written myself out. In a way, youthfulness seems closer to me today than when I was a young man. I no longer regard happiness as unattainable; once, long ago, I did. Now I know that it may occur at any moment but that it should never be sought after. As to failure or fame, they are quite irrelevant and I never bother about them. What I'm out for now is peace, the enjoyment of thinking and of friendship, and, though it may be too ambitious, a sense of loving and of being loved. Jorge Luis Borges, The Aleph and Other Stories"
    sampleEn3 = "China Evergrande Group is unlikely to receive direct government support and is on the brink of defaulting on upcoming debt payments, S&P Global Ratings said. The distressed developer’s troubles could further hit investor confidence in China’s property sector and junk-rated credit markets more broadly, according to an S&P report dated Sept. 20."
    sampleEn5 = "The Indian Space Research Organisation or is the national space agency of India, headquartered in Bengaluru. It operates under Department of Space which is directly overseen by the Prime Minister of India while Chairman of ISRO acts as executive of DOS as well."
    sampleEn = "Apple Inc. is an American multinational technology company that specializes in consumer electronics, computer software, and online services. Apple is the world's largest technology company by revenue (totaling $274.5 billion in 2020) and, since January 2021, the world's most valuable company. As of 2021, Apple is the world's fourth-largest PC vendor by unit sales, and fourth-largest smartphone manufacturer. It is one of the Big Five American information technology companies, along with Amazon, Google, Microsoft, and Facebook.\
    Apple was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976 to develop and sell Wozniak's Apple I personal computer. It was incorporated by Jobs and Wozniak as Apple Computer, Inc. in 1977, and sales of its computers, including the Apple II, grew quickly. It went public in 1980 to instant financial success. Source: Wikipedia"


    st.sidebar.image('./NLP/logohd.jpg', width=256)
    st.title('Exploratory Data for NLP using spaCy 📜')

    st.write("##")
    input_data = st.text_area('Input text to analyze:', sampleEn, height=400)
    doc1 = nlp(input_data)





    sep()
    st.subheader('🔹 NER visualize:') 
    #Named Entity Recognition
    visualize_ner(doc1, labels=nlp.get_pipe("ner").labels, key='1')

    sep()
    st.subheader('🔹 Basic Statistics:')
    st.write(f'◾️ Total entities detected in text: {len(doc1.ents)}')
    labels = [x.label_ for x in doc1.ents]
    counter=Counter(labels)
    st.write(f'◾️ Labels in text: {counter}')

    # all tokens that arent stop words or punctuations
    words = [token.text
            for token in doc1
            if not token.is_stop and not token.is_punct]
    st.write(f'◾️ Total tokens detected in text (<> punctuations): {len(words)}')

    # noun tokens that arent stop words or punctuations
    nouns = [token.text
            for token in doc1
            if (not token.is_stop and
                not token.is_punct and
                token.pos_ == "NOUN")]
    st.write(f'◾️ Total nouns tokens detected in text: {len(nouns)}')

    # five most common tokens
    word_freq = Counter(words)
    common_words = word_freq.most_common(5)
    st.write("◾️ five most common tokens:")
    st.write(common_words)

    # five most common noun tokens
    noun_freq = Counter(nouns)
    common_nouns = noun_freq.most_common(5)
    st.write("◾️ five most common noun tokens:")
    st.write(common_nouns)



    sep()
    st.subheader('🔹 Words count in Entities :')
    most=counter.most_common()
    x, y= [], []
    if len(most) > 0:
        for word,count in most[:40]:
            #if (word not in stop):
            x.append(word)
            y.append(count)
        fig, ax = plt.subplots()
        sns.barplot(x=y,y=x)
        st.write( fig )




    sep()
    st.subheader('🔹 Five most common words :')
    word_freq = Counter(words).most_common(5)
    x, y = zip(*word_freq)
    fig, ax = plt.subplots()
    plt.bar(x, y)
    st.write( fig )



    sep()
    st.subheader('🔹 Count speech tags:') 
    # Returns integers that map to parts of speech
    counts_dict = doc1.count_by(spacy.attrs.IDS['POS'])
    # Print the human readable part of speech tags
    cols = ['tag', 'count']
    lst = []
    for pos, count in counts_dict.items():
        human_readable_tag = doc1.vocab[pos].text   
        lst.append([human_readable_tag, count])
    df1 = pd.DataFrame(lst, columns=cols)
    st.write(df1)





    sep()
    st.subheader('🔹 Most common tokens per entity:') 
    makes = nlp.get_pipe("ner").labels
    make_choice = st.selectbox('Select entity to count:', makes)
    def ner(text,ent=make_choice):
        doc=nlp(text)
        return [X.text for X in doc.ents if X.label_ == ent]

    if len(make_choice) > 0:
        #words = [word.text for word in nlp(input_data)]
        #st.write(words)

        cols = ['word']
        wordDf = []
        wordDf = pd.DataFrame(words, columns=cols) 
        #st.write(wordDf)
        gpe=wordDf['word'].apply(lambda x: ner(x))
        gpe=[i for x in gpe for i in x]
        st.write(gpe)
        if len(gpe) > 0:
            counter=Counter(gpe)
            x,y=map(list,zip(*counter.most_common()))
            fig, ax = plt.subplots()
            sns.barplot(y,x)
            st.write( fig )




    stopwords = set(STOPWORDS)
    def show_wordcloud(data):
        wordcloud = WordCloud(
            background_color='black',
            stopwords=stopwords,
            max_words=100,
            max_font_size=30,
            scale=3,
            random_state=1)
        wordcloud=wordcloud.generate(str(data))
        fig = plt.figure(1, figsize=(12, 12))
        plt.axis('off')
        plt.imshow(wordcloud)
        plt.show()
        st.write( fig )

    sep()
    st.subheader('🔹 WordCloud:')
    show_wordcloud(doc1)


