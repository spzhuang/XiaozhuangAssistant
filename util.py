# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 00:25:28 2024

@author: admin
"""
from langchain.schema import AIMessage,HumanMessage
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables.utils import AddableDict
from langchain_core.outputs import LLMResult
from typing import (List,Union,Dict,Any,TYPE_CHECKING,
    AsyncGenerator,Callable,Final,Generator,Iterable,cast,)
from langchain.prompts import StringPromptTemplate
from langchain.tools import Tool
from langchain.agents import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish, OutputParserException
import re

from streamlit.delta_generator import DeltaGenerator
import inspect
from streamlit import dataframe_util, type_util
from streamlit.errors import StreamlitAPIException
from streamlit.runtime.metrics_util import gather_metrics
_TEXT_CURSOR: Final = " ▏"

support_llm={"zhipu":ChatZhipuAI}
support_embedding = {"zhipu":ZhipuAIEmbeddings}


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


agent_template_zh = """
你是小庄助手,一个幽默风趣的聊天助手,与人们闲聊或者回答用户的问题.
你可以综合考虑使用下面的工具以及先前的对话历史进行回答:
    
tools: {tools}

回答需要遵循下面的格式:
    
Question: 你需要回答的用户输入问题
Thought: 你应该总是需要思考如何去做
Action: 你所采取的行动, 它应该属于某个 [{tool_names}]
Action Input: 给Action的输入
Observation: action的结果
...(这样的Thought/Action/Action Input/Observation可以重复最多10次,如果仍然没有答案,就告诉用户很抱歉,我不知道答案)
Thought: 我现在知道了最终的答案了
Final Answer: 用户问题的最终答案

开始吧! 记住给出的最终答案可以使用一些表情符号以体现幽默风趣.

先前的对话历史
{history}

新的问题: {input}
{agent_scratchpad}
"""



def message2str(message):
    if isinstance(message,HumanMessage):
        return f"\n human: {message.content} \n"
    elif isinstance(message,AIMessage):
        return f"AI: {message.content} \n"
    else:
        raise(ValueError,"message2str 只支持HumanMessage或AIMessage")
        
def get_title_abstract(pdf_content:str):
    "为了得到pdf的摘要与标题"
    abstract = re.findall(pattern=r"Abstract([\w \n\.\S]+)Introduction",string=pdf_content)[0]
    abstract = abstract.strip()
    if abstract.endswith("1."):
        abstract = abstract[:-2]
    abstract = abstract.strip()
    abstract = abstract[:abstract.rfind(".")+1]
    abstract_index = pdf_content.find("Abstract")
    title_include = pdf_content[:abstract_index]
    title_include = title_include.split("\n")
    ind = 0
    while ind<len(title_include):
        if len(title_include[ind]) <= 1:
            title_include.remove(title_include[ind])
        else:
            ind += 1
    title = title_include[:2]
    if title[1].startswith(":"):
        title = title[0] + title[1]
    else:
        title = title[0]
    
    title_abstract = "Title: "+ title + "\nAbstract: " + abstract
    return title,abstract,title_abstract        
   
def direct(query:str)->str:
    """
    如果问题不需要使用工具,那么直接回答
    """
    return ""


def get_date_now(query:str)->str:
    """
    返回当前的日期与时间
    """
    import datetime
    return str(datetime.datetime.now())


def encode_string(passwords:str,d=3)->str:
    passlis = list(passwords)
    passmap = map(lambda x:ord(x)-d,passlis)
    output = map(chr,passmap)
    return "".join(list(output))

def decode_string(passwords:str,d=3)->str:
    passlis = list(passwords)
    passmap = map(lambda x:ord(x)+d,passlis)
    output = map(chr,passmap)
    return "".join(list(output))




class MyChunkedCallbackHandler(BaseCallbackHandler):
    def __init__(self, chunk_size=5):
        self.chunk_size = chunk_size
        self.current_chunk = []
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.current_chunk.append(token) # 每次生成新的token, 就将新的tokentoken加入到self.current_chunk中, 如果超过了chunk_size, 则打印
        if len(self.current_chunk) >= self.chunk_size:
            print("".join(self.current_chunk),end="")
            self.current_chunk = []
    def on_llm_end(self,response:LLMResult,**kwargs) -> None:
        if self.current_chunk:
            print("".join(self.current_chunk))
            self.current_chunk = []


class LoggingHandler2(BaseCallbackHandler):

    def __init__(self):
        self.isprint = 0
        self.last_token = ""
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:

        #　print(token,end="")

        if self.isprint >= 3:
            if ":" in self.last_token:
                if token.startswith(" "):
                    print(token.lstrip(),end="")
                else:
                    print(token,end="")
                self.last_token = token
            else:
                print(token,end="")
                
        if "Final" in token:
            self.isprint += 1
            self.last_token = token
        if "Answer" in token and "Final" in self.last_token:
            self.isprint += 1
            self.last_token = token
        if ":" in token and "Answer" in self.last_token:
            self.isprint += 1
            self.last_token = token

    def on_chain_end(self,outputs:Dict[str,Any], **kwargs)->None:
        self.isprint = 0
        self.last_token = ""

class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]
    def format(self, **kwargs)->str:
        intermediate_steps = kwargs.pop("intermediate_steps") # intermediate_steps = (AgentAction, Observation)
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation:{observation}\nThought:"
        kwargs["agent_scratchpad"] = thoughts # 设置agent_scratchpad变量的值
        kwargs["tools"] = "\n".join([f"{i.name}:{i.description}" for i in self.tools])
        kwargs["tool_names"] = ", ".join([i.name for i in self.tools])
        return self.template.format(**kwargs)

class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output:str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # return values 通常是一个单键"output"的字典, 不推荐对output的值做任何尝试
                return_values = {"output": llm_output.split("Final Answer:")[-1].strip()},
                log = llm_output
            )
        # 解析action的输入与输出
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex,llm_output,re.DOTALL) #re.DOTALL 当使用 re.DOTALL标志时 .也会匹配换行符
        if not match:
            if "Action Input" not in llm_output:
                llm_output += "\nAction: DirectAnswer"
                llm_output += "\nAction Input: "
                match = re.search(regex,llm_output,re.DOTALL) #re.DOTALL 当使用 re.DOTALL标志时 .也会匹配换行符
                if not match:
                    raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # 返回Action与Action输入
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

class Retriever:
    
    def __init__(self,vector_store,abstract):
        self.vector_db = vector_store
        self.abstract = abstract
        self.titles = "\n".join(list(self.abstract.keys()))
    def relevance_search(self,query:str,title:str="",k:int=4):
        if len(title)>0:
            retriever = self.vector_db.as_retriever(search_kwargs={"k":k,"filter":{"title":title}})
        else:
            retriever = self.vector_db.as_retriever(search_kwargs={"k":k})
        similar_doc = retriever.get_relevant_documents(query)

        # relevant_doc = [i.page_content for i in similar_doc]
        # title_doc = [i.metadata['title'] for i in similar_doc]
        if len(title)>0:
            res = f"title = {title}\nabstract = {self.abstract[title]}"
        else:
            res = ""
            for k,v in self.abstract.items():
                res = res + f"Title: {k}\n"
                res = res + f"Abstract: {v}\n\n"
            
        res += "relevant document:"
        for inde,doc in enumerate(similar_doc):
            res += f"'id' = {inde}\n'source' = {doc.metadata['title']}\n'document' = {doc.page_content}\n{'='*10}\n"
        res = res.strip()
        return res
    
    def tool_description(self):
        descript = f"""如果用户询问,请调用此工具,而不是直接回答.如果用户明示了文献的标题,
        则传入title参数,否则不用传入,这意味着此工具会提供所有文献的摘要与标题.
        你可用下面的文献标题判断用户询问的内容是否与此工具有关.\n{self.titles}"""
        return descript





