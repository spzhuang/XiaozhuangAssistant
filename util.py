# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 00:25:28 2024

@author: admin
"""
from langchain.schema import AIMessage,HumanMessage
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.embeddings import ZhipuAIEmbeddings

template = """你叫小庄助手,是一个幽默风趣的AI助手,喜欢在回答中增加表情符号使得聊天更加
有趣.你的任务是陪伴用户聊天或者根据文档回答问题.如果用户的提问你不清楚答案,只要诚实的回答
你不知道即可.并且,你还要注意使用用户使用的语言回答问题.你可充分利用chat_history以及context回答用户的query.
如果用户提出的问题需要使用数学公式回答,请使用'$'符号包裹行内数学公式以及'$$'符号包括单行数学公式.

context: {context}

history: {history}

query: {query}

answer:
"""

summary_template = """请将history的内容进行压缩,要求字数尽可能少于400字.
history: {history}
output: 
"""

query_about_pdf = """请判断用户提出的问题与文档的标题以及摘要的相关性,如果相关性较大,请你返回一个接近1的浮点数,
越是相关则越接近1,如果相关性较小,请你返回一个接近0的数,越是不相关,则越接近0.不需要说不相干的话.
query: {query}

title and abstract: 
{title_abstract}

----
score: 
"""


def message2str(message):
    if isinstance(message,HumanMessage):
        return f"\n human: {message.content} \n"
    elif isinstance(message,AIMessage):
        return f"AI: {message.content} \n"
    else:
        raise(ValueError,"message2str 只支持HumanMessage或AIMessage")
        
support_llm={"zhipu":ChatZhipuAI}
support_embedding = {"zhipu":ZhipuAIEmbeddings}