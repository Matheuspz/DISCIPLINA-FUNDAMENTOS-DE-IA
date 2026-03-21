import streamlit as st
# import os
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate

#Criar o titulo
st.title("My ChatBot")
st.markdown("---")

with st.sidebar:
    st.markdown("##  Configurações")
    st.markdown("---")

    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="",
        help="Crie sua chave em console.groq.com/keys"
    )
    contexto = st.text_area(
        "Contexto do Assitente",
        value="Você é um jogador italiano proficional de jogo da velha chamado Mario, você sabe falar portugues, porém com um sotaque italiano",
        height=160,
        disabled=True,
    )
    #  Escolha de modelos
    modelo = st.selectbox(
        "Modelo Groq",
        [
            "llama-3.3-70b-versatile",
            "llama3-8b-8192",
            "mixtral-8x7b-32768"
        ]
    )
    # Ajuste de temperatura
    temperatura = st.slider(
        "Temperatura (criatividade)",
        0.0, 1.0, 0.3
    )
    # Criação da memória
    st.markdown("---")

    if st.button(" Limpar conversa"):
        st.session_state.mensagens = []
        st.rerun()
    # Nome do desenvolvedor(a)
    st.markdown("---")
    st.markdown(
        "<small>Fundamentos de IA · Univille<br>Matheus Pesch Zenere</small>",
        unsafe_allow_html=True
    )


# FUNÇÃO DO BOT
def resposta_bot(mensagens):

    chat = ChatGroq(
        model=modelo,
        api_key=groq_key,
        temperature=temperatura
    )

    mensagens_modelo = [("system", contexto)]
    mensagens_modelo += mensagens

    prompt = ChatPromptTemplate.from_messages(mensagens_modelo)

    chain = prompt | chat

    resposta = chain.invoke({})

    return resposta.content


# MEMÓRIA DO CHAT

if "mensagens" not in st.session_state:
    st.session_state.mensagens = [
        ("assistant", "Olá!  Sou o assistente de IA da Univille. Como posso ajudar?")
    ]


# MOSTRAR HISTÓRICO

for role, content in st.session_state.mensagens:

    with st.chat_message(role):
        st.markdown(content)


# INPUT DO USUÁRIO
pergunta = st.chat_input("Digite sua pergunta...")


if pergunta:

    if not groq_key:
        st.warning(" Informe sua Groq API Key no menu lateral.")
        st.stop()

    st.session_state.mensagens.append(("user", pergunta))

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):

            resposta = resposta_bot(st.session_state.mensagens)

            st.markdown(resposta)

    st.session_state.mensagens.append(("assistant", resposta))