# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 13:40:00 2024

@author: spzhuang
"""

import streamlit as st
state = st.session_state
# from streamlit_chat import message
import yaml
from util import decode_string
from langchain.schema import HumanMessage,AIMessage
from my_rag_agent_v1 import Conversation
import tempfile 
import os,re
current_dir = os.getcwd()
tempdir = os.path.join(os.getcwd(),"tempdir")
os.makedirs(name=tempdir,exist_ok=True)
import shutil
import time
from streamlit.delta_generator import DeltaGenerator
from langchain.callbacks import StreamingStdOutCallbackHandler

class Mycallback(StreamingStdOutCallbackHandler):

    def __init__(self,placeholder1:DeltaGenerator=None,placeholder2:DeltaGenerator=None):
        self.token1 = ""
        self.token2 = ""
        self.flag = 0
        self.placeholder1 = placeholder1
        self.placeholder2 = placeholder2
       
    def on_llm_new_token(self, token: str,  **kwargs) -> None:

        if "Final" not in token and self.flag==0:
            self.token1 += token
            self.placeholder1.text(self.token1)
        else:
            if "Final" in token or self.flag>0:
                self.flag += 1
                if self.flag<=3:
                    self.token1 += token
                    self.placeholder1.text(self.token1)
                else:
                    self.token2 += token
                    self.placeholder2.text(self.token2)
            



with open('config.yml','r') as file:
    config = yaml.safe_load(file)
bing_url = config["search"]["url"]
bing_api = config["search"]["key"]

if 'chat_state' not in state:
    state.chat_state = {"聊天输入框":{"聊天输入":None,"隐藏":True,"完成对话轮次":False},
                        "聊天历史":{k:[] for k in config["user"]}}

if 'sidebar_widget_state' not in state:
    state.sidebar_widget_state = {"解析按钮":{"隐藏":True,"是否点击":False},
                                  "用户名输入框":{"隐藏":True,"是否合法":False,"内容":None},
                                  "清除按钮":{"隐藏":True},
                                  "文件上传器":None}

if "llm_info" not in state:
    state.llm_info={"factor":None,"model_name":None,"embedding_name":None,"api_key":None}

if "search_info" not in state:
    state.search_info = {"search_url":"https://api.bing.microsoft.com/v7.0/search","bing_search_api_key":None}
    

if "sign_info" not in state:
    state.sign_info = {"产商":None,
                       "模型":None,
                       "embedding":None,
                       "模型apikey":None,
                       "配置":{"是否正确":None,"说明":None}}


if "history" not in state:
    state.history = []

if "answer" not in state:
    state.answer = []



if "count" not in state:
    state.count = 0


def validate_user(username):
    state.sidebar_widget_state["用户名输入框"]["内容"] = username
    if username in config['user']:
        config_llm_info = [state.sign_info["产商"],
                           state.sign_info["模型"],
                           state.sign_info["模型apikey"]]
        config_embedding_info = [state.sign_info["产商"],
                           state.sign_info["embedding"],
                           state.sign_info["模型apikey"]]
        
        state.chat_state["聊天输入框"]["隐藏"]=False
        state.sidebar_widget_state["用户名输入框"]["是否合法"]=True

        state.agent = Conversation(llm_info=config_llm_info,
                                   embedding_info=config_embedding_info,
                                   memory_max_token=2000)
        if state.sign_info["配置"]["是否正确"] is None or state.sign_info["配置"]["是否正确"]==False:
            try:
                response = state.agent.llm.invoke("你好")
                if response:
                    state.sign_info["配置"]["是否正确"] = True
                    state.sign_info["配置"]["说明"] = "已验证,配置正确"
                else:
                    state.sign_info["配置"]["是否正确"] = False
                    state.sign_info["配置"]["说明"] = "配置错误,请检查sign_info设置信息"
            except:
                state.sign_info["配置"]["是否正确"] = False
                state.sign_info["配置"]["说明"] = "网络错误或者配置错误,请检查sign_info设置信息"
            
        if state.sidebar_widget_state["文件上传器"] and state.sign_info["配置"]["是否正确"] == True:
            state.sidebar_widget_state["解析按钮"]["隐藏"] = False
        
        current_user = username
        if state.chat_state["聊天历史"][current_user]:
            with st.session_state["container"]:
                for user,message in state.chat_state["聊天历史"].get(current_user):
                    if user=="ai":
                        with st.chat_message(name=user):
                            st.write(message)
                    else:
                        with st.chat_message(name=current_user,avatar="小庄AI.png"):
                            st.write(message)
        
    else:
        state.sidebar_widget_state["解析按钮"]["隐藏"] = True
        state.sidebar_widget_state["用户名输入框"]["是否合法"]=False




def analysis_button():
    state.sidebar_widget_state["解析按钮"]["是否点击"]=True
    files = state["file_uploader"]
    tempfile_path = []
    for i in files:
        with tempfile.NamedTemporaryFile(delete=False,dir=tempdir) as fb:
            fb.write(i.getbuffer())
            tempfile_path.append(fb.name)
    with st.session_state["sidebar_spinner"],st.spinner("initiazing"):
        state.agent.init_system(file_paths=tempfile_path)
    state.sidebar_widget_state["清除按钮"]["隐藏"] = False

def upload_change():
     
    state.sidebar_widget_state["文件上传器"] = state["file_uploader"]
    if state.sidebar_widget_state["用户名输入框"]["是否合法"]==True:
        state.sidebar_widget_state["解析按钮"]["隐藏"] = False


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
    state.sidebar_widget_state["清除按钮"]["隐藏"] = True
    state.sidebar_widget_state["解析按钮"]["是否点击"] = False

def model_apikey(value):
    if (state.selectbox1 and state.selectbox2 and state.selectbox0): 
        state.sidebar_widget_state["用户名输入框"]["隐藏"]=False
        state.sign_info["产商"] = state.selectbox0
        state.sign_info["模型"] = state.selectbox1
        state.sign_info["embedding"] = state.selectbox2
        state.sign_info["模型apikey"] = value
        state.selected_page = "Chat"
        state.sign_info["配置"]["是否正确"] = None
        state.text_input = None
    else:
        state.sidebar_widget_state["用户名输入框"]["隐藏"]=True


def prompt_handle(query): 
    
    # state.history.append((query,"系统回复"))

    
    current_user = state.sidebar_widget_state["用户名输入框"]["内容"]
    

            
    with st.session_state["container"]:
        
        for user,message in state.chat_state["聊天历史"].get(current_user):
            if user=="ai":
                with st.chat_message(name=user):
                    st.write(message)
            else:
                with st.chat_message(name=current_user,avatar="小庄AI.png"):
                    st.write(message)
            
        with st.chat_message(name=current_user):
            st.write(query)
            state.chat_state["聊天历史"][current_user].append(["user",query])
        with st.chat_message(name="ai",avatar="小庄AI.png"):
            with st.status("正在思考"):
                place1 = st.empty()
            place2 = st.empty()
            callback = Mycallback(placeholder1=place1,placeholder2=place2)
            state.agent.add_streamlit_callback(callback)
            stream_ans = state.agent.ask(query)
            response = []
            for j in stream_ans:
                response.append(j)
                
            # st.markdown(response[-1]["output"])
            state.chat_state["聊天历史"][current_user].append(["ai",response[-1]["output"]])


def select_page(value):
    if value=="signinfo":
        state.apikey_input = state.sign_info["模型apikey"]

def chat():
    st.session_state["container"] = st.container(height=600)
    # current_user = state.sidebar_widget_state["用户名输入框"]["内容"]
    # with st.session_state["container"]:
    #     for message in state.history:
    #         if message[0] == "user":
    #             with st.chat_message(name=current_user):
    #                 st.markdown(message[1])
    #         elif message[0]=="ai":
    #             with st.chat_message(name="ai",avatar="小庄AI.png"):
    #                 st.markdown(message[1])
    
    st.chat_input("你想知道什么",
                  key='chat_input',
                  on_submit=lambda: prompt_handle(st.session_state['chat_input']),
                  disabled=state["chat_state"]["聊天输入框"]["隐藏"])
    

def sign():

    st.selectbox(label="factor-name",options=["zhipu"],key="selectbox0",index=0)
    st.selectbox(label="model-name",options=["glm-4-plus"],key="selectbox1",index=0)
    st.selectbox(label="embedding-name",options=["embedding-3"],key="selectbox2",index=0)
    st.text_input(label="apikey",type="password",key="apikey_input",
                  on_change=lambda:model_apikey(st.session_state.apikey_input))
    
# def count2():
#     state.count += 1


pages = {"Chat":chat,"signinfo":sign}

def main():
    # st.title("小庄助手欢迎您")
    st.set_page_config(page_title="小庄助手欢迎您",page_icon="小庄AI.png")
    col1,col2 = st.columns(2)
    with col1:
        st.image("小庄AI.png")
    with col2:
        st.write("# 小庄助手欢迎您")
    
    st.markdown(""" 
                <div style='text-align: right;'> 
                    <h3 style='color: gray; font-size: 16px'>个性化聊天伙伴或阅读助手-Agent版</h3> 
                </div> """, unsafe_allow_html=True)
    selected_page = st.sidebar.selectbox("❗请先在signinfo页面输入信息", options=list(pages.keys()),
                                         key="selected_page",on_change=lambda:select_page(state.selected_page))
    st.sidebar.title("上传你的PDF")
    st.sidebar.file_uploader("",type=["pdf"],key="file_uploader",on_change=upload_change,accept_multiple_files=True) # 文件展示 

    st.sidebar.button(label="解析文档",on_click=analysis_button,disabled=state.sidebar_widget_state["解析按钮"]["隐藏"])
    state["sidebar_spinner"] = st.sidebar.empty()
    state["sidebar_spinner2"] = st.sidebar.empty()
    usr_name = st.sidebar.text_input(label="请输入您的用户名：",placeholder="输入之后按回车",
                                     key="text_input",on_change=lambda:validate_user(state["text_input"]),
                                     disabled=state.sidebar_widget_state["用户名输入框"]["隐藏"])
    placeholder = st.sidebar.empty() # 创建占位符
    if usr_name:
        if state.sidebar_widget_state["用户名输入框"]["是否合法"] == False:
            placeholder.write(f":red[用户{usr_name}没有权限,请联系管理员]")
        else:
            if state.sign_info["配置"]["是否正确"] == False:
                placeholder.write(f":red[用户{usr_name}配置不正确: {state.sign_info['配置']['说明']}😔]")
            elif state.sign_info["配置"]["是否正确"] == True:
                placeholder.write(f":green[用户{usr_name}已登录,您可开始对话啦😊]")
    
    if state.sidebar_widget_state["解析按钮"]["是否点击"]==True:
        state["sidebar_spinner2"].write(":green[文档解析成功😊]")

    st.sidebar.button(label="清除临时文件",on_click=sidebar_clear_tempfile_dir,
                      disabled=state.sidebar_widget_state["清除按钮"]["隐藏"])

    pages[selected_page]()
    # click = st.sidebar.button(label="计数器",on_click=count2)
    # if click:
    #     with st.session_state["container"]:
    #         st.write(state.count)

if __name__ == "__main__":
    main()


# def message2string(info1,info2,idkey):
#     for info in [info1,info2]:
#         if isinstance(info,HumanMessage):
#             info2str = info.content
#             is_user = True
#             message(info2str,is_user=is_user,key="human_"+str(idkey))
#         elif isinstance(info,AIMessage):
#             info2str = info.content
#             is_user = False
#             message(info2str,is_user=is_user,key="ai_"+str(idkey))




