# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 12:51:39 2024

@author: spzhuang
"""


from langchain.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ChatMessageHistory
from langchain.schema import HumanMessage,SystemMessage
from langchain.prompts import PromptTemplate
from util import template,summary_template,message2str,support_llm,support_embedding,query_about_pdf
import os,re

class Conversation:
    
    summary_prompt = PromptTemplate(template=summary_template,input_variables=["history"])
    prompts = PromptTemplate(template=template,input_variables=["context", "history", "query"])
    relevant_prompt = PromptTemplate(template=query_about_pdf,input_variables=["query","title_abstract"])

    def __init__(self,llm_info,embedding_info,memory_max_length = 2000):
        """
        llm_info = (factory,model_name,api_key):(str,str,str) 用来确定llm的api调用信息
        embeddings_info = (factory,embedding_name,api_key):(str,str,str) 用来确定embedding的api调用信息
        """
        self.llm = support_llm[llm_info[0]](model=llm_info[1],api_key=llm_info[2])
        self.embedding = support_embedding[embedding_info[0]](model=embedding_info[1],api_key=embedding_info[2])
        self.history = ChatMessageHistory()
        self.memory_max_length = memory_max_length
        self.history_context = ''
        self.db = None
        self.chain = None
        self.completion = None
        self.title_abstract = None
        self.chain = Conversation.prompts | self.llm
    def init_system(self,file_path:os.PathLike=None,chunksize=500,chunkoverlap=50,tub_size=50):
        
        if file_path is not None:
            pdf_file = PDFMinerLoader(file_path=file_path)
            documents = pdf_file.load()
            
            abstract = re.findall(pattern=r"Abstract([\w \n\.\S]+)Introduction",string=documents[0].page_content)
            abstract = abstract[0].strip()
            abstract = abstract[:abstract.rfind(".")+1]
            title = documents[0].page_content[:documents[0].page_content.find("\n")]
            title_abstract = "Title: "+ title + "\n\nAbstract: " + abstract
            self.title_abstract = title_abstract
            
            textsplitter = RecursiveCharacterTextSplitter(separators=["\n\n","."],
                                                          chunk_size=chunksize,
                                                          chunk_overlap=chunkoverlap)
            documents_split = textsplitter.split_documents(documents)
            for i,v in enumerate(documents_split):
                documents_split[i].metadata["sort"] = i
            documents_count = len(documents_split)
            tub = (documents_count//tub_size) + 1 # 将分档分成若干个桶， 每个桶大小不超过50
            tub_range = [i*tub_size for i in range(tub+1)]
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
        
        
    def ask(self,query):
        
        score = self.llm.invoke(Conversation.relevant_prompt.format(query=query,title_abstract=self.title_abstract))
        score = float(re.findall(r"[0-9\.]+",score.content)[0])
        
        if self.db:
            if score>0.5:
                similar_doc = self.db.similarity_search_with_relevance_scores(query, k=5)
                relevant_score = max([i[1] for i in similar_doc])
                if relevant_score < 0.15:
                    context = self.title_abstract
                else:
                    context = [i[0].page_content for i in similar_doc]
                    context = '\n'.join(context)
            else:
                context = ""
        else:
            context = ""
        
        human_message = HumanMessage(query)
        system_message = SystemMessage(context)
        
        if self.completion:        
            self.history.add_user_message(self.completion[0])
            self.history.add_ai_message(self.completion[1])
            
            last_two = self.history.messages[-2:]
            last_two = map(message2str,self.history.messages[-2:])
            for i in last_two:
                self.history_context = self.history_context + i
            if len(self.history_context) > self.memory_max_length:
                compress_history = self.llm.invoke(Conversation.summary_prompt.format(history=self.history_context))
                self.history_context = f"coversation summary: {compress_history.content}\n"
        else:
            pass
        
        response = self.chain.invoke({"history":self.history_context,"context":system_message,
                                 "query":human_message}) 
        self.completion = [human_message,response]
        return response
    
    def clean(self)->None:
        self.db = None
        self.chain = None


    










