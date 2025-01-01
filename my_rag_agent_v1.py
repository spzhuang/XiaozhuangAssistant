# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 12:51:39 2024
小庄助理Agent版本
@author: admin
"""

# 1. import 必要包
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationSummaryBufferMemory,ConversationBufferMemory,ChatMessageHistory
from langchain.schema import Document,AIMessage,HumanMessage,SystemMessage,messages_to_dict
from langchain.prompts import PromptTemplate,SystemMessagePromptTemplate,StringPromptTemplate
from langchain.chains import ConversationalRetrievalChain,RetrievalQA,ConversationChain
from langchain_core.runnables import RunnablePassthrough 
import re,os,yaml

from typing import List, Dict, Union
from langchain.agents import AgentExecutor, LLMSingleActionAgent, AgentOutputParser

from langchain.utilities import BingSearchAPIWrapper
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import Tool,tool
from util import get_title_abstract,agent_template_zh,direct,query_about_pdf,decode_string,LoggingHandler2
from util import CustomPromptTemplate,CustomOutputParser,Retriever,support_llm,support_embedding,get_date_now
from pydantic import BaseModel, Field, validator
from calculate import X_LLMMathChain

from langchain.callbacks import StreamingStdOutCallbackHandler
from streamlit.delta_generator import DeltaGenerator

# import ipdb
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'


with open('config.yml','r') as file:
    config = yaml.safe_load(file)
bing_url = config["search"]["url"]
bing_api = decode_string(config["search"]["key"])
search = BingSearchAPIWrapper(bing_subscription_key=bing_api,bing_search_url=bing_url,k=5) 



class Conversation:
    
    # relevant_prompt = PromptTemplate(template=query_about_pdf,input_variables=["query","title_abstract"])
    
    def __init__(self,llm_info,embedding_info,memory_max_token = 2000,chunksize=1000,chunkoverlap=50):
        """
        llm_info = (factory,model_name,api_key):(str,str,str) 用来确定llm的api调用信息
        embeddings_info = (factory,embedding_name,api_key):(str,str,str) 用来确定embedding的api调用信息
        """
        self.llm = support_llm[llm_info[0]](model=llm_info[1],api_key=llm_info[2],streaming=True)
        llm_math_chain = X_LLMMathChain.from_llm(llm=self.llm, verbose=False)
        
        self.embedding = support_embedding[embedding_info[0]](model=embedding_info[1],api_key=embedding_info[2])

        self.memory = ConversationSummaryBufferMemory(llm=self.llm,max_token_limit=memory_max_token)
        self.completion = None
        
        self.text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n","."],
                                                      chunk_size=chunksize,
                                                      chunk_overlap=chunkoverlap)
        self._tools = [Tool(name="Search", func=search.run, description="当你需要回答当前事件的时候使用"),
                       Tool(name="Calculate",func=llm_math_chain.run, description="当你需要做数学计算时使用"),
                       Tool(name="DirectAnswer",func=direct,description="如果问题不需要使用工具,那么直接回答"),
                       Tool(name="GetDatetime",func=get_date_now,description="此工具返回当前的日期与时间,如果用户的问题与当前的时间日期有关,比如:\n某地今天天气;"
                      "某地最新热点;某地明天天气等等你应该"
                      "先使用此工具得到当前的日期与时间,再根据现在的日期或者时间进行搜索")
                        ]
        self._tool_names = [i.name for i in self._tools]
        self.prompt = CustomPromptTemplate(template = agent_template_zh,tools = self._tools,
                                      input_variables = ["input","intermediate_steps","history"])
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
        self.db = None
        self.title_abstract = None
        self.output_parser = CustomOutputParser()
        agent = LLMSingleActionAgent(
            llm_chain=self.chain,
            output_parser=self.output_parser,
            stop=["\nObservation:"],
            allowed_tools=self._tool_names
        )
        self.agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=self._tools, 
                                                                 verbose=False,memory=self.memory)
        # self.agent_executor = agent_executor.with_config(callbacks=[LoggingHandler2()])
        # if callback:
        #     self.callback_handler = callback
        #     self.agent_executor = agent_executor.with_config(callbacks=[self.callback_handler])
        
    def init_system(self,file_paths:List[os.PathLike]=None,tub_size=50):
        
        if file_paths is not None:
            documents = []
            titles = []
            abstracts = dict()
            titles_abstracts = []
            for i in file_paths:
                pdf_file = PDFMinerLoader(file_path=i)
                cur_document = pdf_file.load()
                title,abstract,title_abstract = get_title_abstract(cur_document[0].page_content)
                titles.append(title)
                abstracts[title] = abstract
                titles_abstracts.append(title_abstract)
                cur_document[0].metadata["title"] = title
                documents += cur_document
            titles2str = ""
            for k,c in enumerate(titles):
                titles2str += "Title " + str(k) + " = "  + c + "\n"
            

            self.titles8abstracts = '\n\n'.join(titles_abstracts)
            self.titles = titles2str
            self.abstracts = abstracts

            documents_split = self.text_splitter.split_documents(documents)
            for i,v in enumerate(documents_split):
                documents_split[i].metadata["sort"] = i
                
            documents_count = len(documents_split)
            tub = (documents_count//50) + 1 # 将分档分成若干个桶， 每个桶大小不超过50
            tub_range = [i*50 for i in range(tub+1)]

            
            if tub==1:
                vector_db = FAISS.from_documents(documents_split,self.embedding)
            else:
                vector_db = FAISS.from_documents(documents_split[:tub_range[1]],self.embedding)
            for ind,val in enumerate(tub_range[1:-1]):
                k1 = val
                k2 = tub_range[ind+2]
                vector_db.add_documents(documents_split[k1:k2])
            self.db = vector_db
        else:
            self.db = None
        if self.db:
            self.clean()
            retriever = Retriever(self.db,self.abstracts)
            retriever_tool = Tool(name="Retriever",func=retriever.relevance_search,
                                  description=retriever.tool_description())
            self._tools.insert(2,retriever_tool)
            self._tool_names.insert(2,retriever_tool.name)
            self.prompt = CustomPromptTemplate(template = agent_template_zh,tools = self._tools,
                                          input_variables = ["input","intermediate_steps","history"])
            self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
            agent = LLMSingleActionAgent(
                llm_chain=self.chain,
                output_parser=self.output_parser,
                stop=["\nObservation:"],
                allowed_tools=self._tool_names
            )
            self.agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=self._tools, 
                                                                     verbose=False,memory=self.memory)
    def ask(self,query):
        response = self.agent_executor.stream(query)
        return response
    
    def clean(self)->None:
        self.chain = None
        self.prompt = None
        
    def add_streamlit_callback(self,callback=None)->None:
        self.callback_handler = callback
        self.agent_executor = self.agent_executor.with_config(callbacks=[self.callback_handler])
    








