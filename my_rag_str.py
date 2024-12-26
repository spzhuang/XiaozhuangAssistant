# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 13:40:00 2024

@author: spzhuang
"""

import streamlit as st
from streamlit_chat import message
import yaml

from langchain.schema import HumanMessage,AIMessage
from my_rag_v1 import Conversation
import tempfile 
import os
current_dir = os.getcwd()
tempdir = os.path.join(os.getcwd(),"tempdir")
os.makedirs(name=tempdir,exist_ok=True)
import shutil


with open('config.yml','r') as file:
    config = yaml.safe_load(file)
   
if 'chat_content' not in st.session_state:
    st.session_state.chat_content = ''
if 'chat_disable' not in st.session_state:
    st.session_state.chat_disable = True
if 'sidebar_button1_disable'  not in st.session_state:
    st.session_state.sidebar_button1_disable=True
if "sidebar_exbutton_isclick" not in st.session_state:
    st.session_state.sidebar_exbutton_isclick=False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {k:[] for k in config["user"]}
if "turn" not in st.session_state:
    st.session_state.turn = {k:0 for k in config["user"]}
if "complete_current_turn" not in st.session_state:
    st.session_state.complete_current_turn = False
if "debug" not in st.session_state:
    st.session_state.debug = False
if "clear_button_disable" not in st.session_state:
    st.session_state.clear_button_disable = True
if "api_key" not in st.session_state:
    st.session_state.api_key = None
if "selectbox0" not in st.session_state:
    st.session_state.selectbox0=None
if "selectbox1" not in st.session_state:
    st.session_state.selectbox1=None
if "selectbox2" not in st.session_state:
    st.session_state.selectbox2=None
# if "file_upload_state" not in st.session_state:
#     st.session_state.file_upload_state = None
if "sidebar_textinput_islegal" not in st.session_state:
    st.session_state.sidebar_textinput_islegal = False
if "sidebar_textinput_show" not in st.session_state:
    st.session_state.sidebar_textinput_show = True


st.title("小庄助手欢迎您")
st.markdown(""" 
            <div style='text-align: right;'> 
                <h3 style='color: gray; font-size: 16px'>个性化聊天伙伴或阅读助手</h3> 
            </div> """, unsafe_allow_html=True)

def validate_user(username):
    if username in config['user']:
        config_llm_info = [st.session_state.selectbox0,
                           st.session_state.selectbox1,
                           st.session_state.api_key]
        config_embedding_info = [st.session_state.selectbox0,
                           st.session_state.selectbox2,
                           st.session_state.api_key]
        
        st.session_state.chat_disable=False
        st.session_state.sidebar_textinput_islegal=True

        st.session_state.T = Conversation(llm_info=config_llm_info,
                                          embedding_info=config_embedding_info,
                                          memory_max_length=2000)
        if st.session_state.file_uploader:
            st.session_state.sidebar_button1_disable = False
    else:
        st.session_state.sidebar_button1_disable = True

def prompt_handle(query): 
    
    current_user = st.session_state["sidebar_text_input"]
    current_history = st.session_state["chat_history"][current_user]
    st.session_state.T.ask(query)
    completion = st.session_state.T.completion
    # completion = [i.content for i in completion]
    st.session_state.turn[current_user] += 1
    completion.append(st.session_state.turn[current_user])
    current_history.append(completion)
    st.session_state.complete_current_turn = True    

def sidebar_button():
    st.session_state.sidebar_exbutton_isclick=True
    file = st.session_state["file_uploader"]
    with tempfile.NamedTemporaryFile(delete=False,dir=tempdir) as f:
        f.write(file.getbuffer())
        f_path = f.name
    with st.session_state["sidebar_spinner"],st.spinner("initiazing"):
        st.session_state.T.init_system(file_path=f_path,chunksize=500,chunkoverlap=50)
    st.session_state.clear_button_disable = False

def upload_change():
    # st.session_state.file_upload_state=True
    if not st.session_state.sidebar_textinput_show:
        st.session_state.sidebar_button1_disable=False

def message2string(info1,info2,idkey):
    for info in [info1,info2]:
        if isinstance(info,HumanMessage):
            info2str = info.content
            is_user = True
            message(info2str,is_user=is_user,key="human_"+str(idkey))
        elif isinstance(info,AIMessage):
            info2str = info.content
            is_user = False
            message(info2str,is_user=is_user,key="ai_"+str(idkey))

def sidebar_apikey(value):
    st.session_state.api_key = value
    if (st.session_state.selectbox1 and st.session_state.selectbox2 and st.session_state.selectbox0): 
        st.session_state.sidebar_textinput_show=False
    else:
        st.session_state.sidebar_textinput_show=True

def sidebar_clear_tempfile_dir():
    if os.path.exists(tempdir):
        for item in os.listdir(tempdir):
            item_path = os.path.join(tempdir, item)
            try:
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f"删除 {item_path} 时出错: {e} 😱")
    else:
        print(f"目录 {tempdir} 不存在！😱")
    st.session_state.clear_button_disable = True
    st.session_state.sidebar_exbutton_isclick = False


st.sidebar.title("上传你的PDF") 
uploaded_file = st.sidebar.file_uploader("",type=["pdf"],key="file_uploader",on_change=upload_change) # 文件展示 



st.sidebar.button(label="解析文档",on_click=sidebar_button,disabled=st.session_state.sidebar_button1_disable)



st.session_state["sidebar_spinner"] = st.sidebar.empty()
st.session_state["sidebar_spinner2"] = st.sidebar.empty()


usr_name = st.sidebar.text_input(label="请输入您的用户名：",placeholder="spzhuang",
                                 key="sidebar_text_input",on_change=lambda:validate_user(st.session_state["sidebar_text_input"]),
                                 disabled=st.session_state.sidebar_textinput_show)




placeholder = st.sidebar.empty() # 创建占位符

if st.session_state.sidebar_exbutton_isclick:
    st.session_state["sidebar_spinner2"].write(":green[文档解析成功😊]")

if usr_name:
    if st.session_state.sidebar_textinput_islegal == False:
        placeholder.write(f":red[用户{usr_name}没有权限,请联系管理员]")
    else:
        placeholder.write(f":green[用户{usr_name}已登录,您可开始对话啦😊]")


st.sidebar.selectbox(label="factor-name",options=["zhipu"],key="selectbox0",index=0)
st.sidebar.selectbox(label="model-name",options=["glm-4-plus"],key="selectbox1",index=0)
st.sidebar.selectbox(label="embedding-name",options=["embedding-3"],key="selectbox2",index=0)

st.sidebar.text_input(label="apikey",type="password",key="apikey_input",on_change=lambda :sidebar_apikey(st.session_state.apikey_input))
st.sidebar.button(label="清除临时文件",on_click=sidebar_clear_tempfile_dir,disabled=st.session_state.clear_button_disable)

# 正文
st.session_state["container"] = st.container()

st.chat_input("你想知道什么",
              key='chat_input',
              on_submit=lambda: prompt_handle(st.session_state['chat_input']),
              disabled=st.session_state.chat_disable)

if st.session_state.complete_current_turn:
    current_user = st.session_state["sidebar_text_input"]
    # current_user_turn = st.session_state.turn[current_user]
    with st.session_state["container"]:
        for human,ai,turn_ in st.session_state.chat_history[current_user]:
            message2string(human,ai,turn_)  
    if st.session_state.debug:
        st.sidebar.write(st.session_state.T.history_context)
        st.sidebar.write(len(st.session_state.T.history_context))
            
# st.write(st.session_state["chat_history"])











