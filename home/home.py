import streamlit as st
import streamlit.components.v1 as components


def title(s):
    st.markdown(f"<h1 style='text-align: center; color: darkblue;'>{s}</h1>", unsafe_allow_html=True)
    #st.markdown('<hr style="height:2px;border-width:0;color:gray;background-color:green">', unsafe_allow_html=True)

def subtitle(s):
    st.markdown(f"<h3 style='text-align: center; color:#0077B5;'>{s}</h3>", unsafe_allow_html=True)
    st.text("")

def subsubtitle(s):
    st.markdown(f"<h4 style='text-align: center; color: darkblue;'>{s}</h4>", unsafe_allow_html=True)
    st.text("")    

def subtitle2(s):
    st.text("")
    st.markdown(f'<p style="background-color:#0066cc;color:#33ff33;font-size:24px;border-radius:2%;">{s}</p>', unsafe_allow_html=True)
    st.text("")

def sep():
    st.markdown('<hr style="height:2px;border-width:0;color:gray;background-color:darkblue">', unsafe_allow_html=True)

def seps():
    st.sidebar.markdown('<hr style="height:2px;border-width:0;color:gray;background-color:white">', unsafe_allow_html=True)




def app():

    #MAIN PAGE SETTINGS:

    title('VICTOR FRABASIL')
    #subsubtitle('🔹 portfolio 🔹')
    subtitle('Software Engineer 🔹 Python/C++ Developer 🔹 Data Analyst ')
    st.write("##")
    #sep()

    #SIDEBAR SETTINGS:
    #st.sidebar.image('./home/Perfil.png', width=256)
    #st.sidebar.write("##") 
    #seps()

    st.markdown("<style> .css-fg4pbf {background: #a7bbcc;}</style>",unsafe_allow_html=True )
    #st.markdown("<style> .css-1d391kg {background-color: #4a6982;}</style>",unsafe_allow_html=True )
    st.markdown("<style> .css-zbg2rx {background-color: #4a6982;}</style>",unsafe_allow_html=True )
    

    skill_col_size = 5
#    embed_component= {'linkedin':"""<script src="https://platform.linkedin.com/badges/js/profile.js" async defer type="text/javascript"></script>
#            <div class="badge-base LI-profile-badge" data-locale="en_US" data-size="medium" data-theme="light" data-type="VERTICAL" data-vanity="mehulgupta7991" data-version="v1"><a class="badge-base__link LI-simple-link" href="https://in.linkedin.com/in/victorfrabasil?trk=profile-badge"></a></div>""", 'medium':"""<div id="medium-widget"></div>
#                <script src="https://medium-widget.pixelpoint.io/widget.js"></script>
#                <script>MediumWidget.Init({renderTo: '#medium-widget', params: {"resource":"https://medium.com/data-science-in-your-pocket","postsPerLine":3,"limit":9,"picture":"big","fields":["description","claps","publishAt"],"ratio":"landscape"}})</script>"""}

    embed_component= {'linkedin':"""<script src="https://platform.linkedin.com/badges/js/profile.js" async defer type="text/javascript"></script>
        <div class="badge-base LI-profile-badge" data-locale="en_US" data-size="medium" data-theme="light" data-type="VERTICAL" data-vanity="victorfrabasil" data-version="v1"><a class="badge-base__link LI-simple-link" href="https://ar.linkedin.com/in/victorfrabasil?trk=profile-badge"></a></div>"""}


    about_me =  """
    ## Connect:
    [![Victor1](https://img.shields.io/badge/Linkedin-victorfrabasil-gray.svg?colorA=gray&colorB=white&logo=linkedin)](https://linkedin.com/in/victorfrabasil)
    \n
    [![Victor2](https://img.shields.io/badge/Github-vfrabasil-gray.svg?colorA=gray&colorB=white&logo=github)](https://www.github.com/vfrabasil/)
    \n
    [![Victor3](https://img.shields.io/badge/Medium-@vfrabasil-gray.svg?colorA=gray&colorB=white&logo=medium)](https://medium.com/@vfrabasil)
    \n
    [![Victor4](https://img.shields.io/badge/Kaggle-victorfrabasil-gray.svg?colorA=gray&colorB=white&logo=kaggle)](https://kaggle.com/victorfrabasil)
    """


    #\n
    #[![Victor3](https://img.shields.io/badge/Gmail-vfrabasil-gray.svg?colorA=gray&colorB=white&logo=gmail)](https://gmail.com)         


    with st.sidebar:
        components.html(embed_component['linkedin'],height=310)
        seps()        
        st.sidebar.markdown(about_me)
        st.write("[![mailto](https://img.shields.io/badge/Gmail-vfrabasil@gmail.com-gray.svg?logo=Gmail&colorA=gray&colorB=white&link=mailto:vfrabasil@gmail.com)](mailto:vfrabasil@gmail.com)")
        #st.write("&nbsp[![Connect](https://img.shields.io/badge/-Beytullah_Ali_Göyem-blue?style=flat-square&logo=Linkedin&logoColor=white&link=https://tr.linkedin.com/in/beytullah-ali-g%C3%B6yem-461749152)](https://tr.linkedin.com/in/beytullah-ali-g%C3%B6yem-461749152) &nbsp[![mailto](https://img.shields.io/badge/-beytullahali.goyem@gmail.com-c14438?style=flat-square&logo=Gmail&logoColor=white&link=mailto:beytullahali.goyem@gmail.com)](mailto:beytullahali.goyem@gmail.com)&nbsp[![Follow](https://img.shields.io/twitter/follow/baligoyem?style=social)](https://twitter.com/baligoyem)")
        #st.sidebar.markdown('<a class="mailto" href="mailto:vfrabasil@gmail.com" style="color:black;">📧</a>', unsafe_allow_html=True)

    seps()
    pdfFileObj = open('pdfs/Frabasil Victor Resume.pdf', 'rb')
    st.sidebar.download_button('📥 Resume / CV',pdfFileObj,file_name='Frabasil Victor Resume.pdf',mime='pdf', help='Download Resume')




    #st.sidebar.write("** CONTACT: **")   
    #st.sidebar.write("◾️ vfrabasil@gmail.com")
    #st.sidebar.write("◾️ linkedin.com/in/victorfrabasil")   
    #st.sidebar.write("◾️ github.com/vfrabasil")   
    #st.sidebar.write("##")   
    seps()