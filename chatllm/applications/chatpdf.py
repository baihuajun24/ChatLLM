#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : AI.  @by PyCharm
# @File         : ChatPDF
# @Time         : 2023/4/21 11:44
# @Author       : betterme
# @WeChat       : meutils
# @Software     : PyCharm
# @Description  :
import glob
from pathlib import Path
from meutils.pipe import *
from meutils.office_automation.pdf import extract_text, pdf2text

from chatllm.utils import textsplitter
from chatllm.applications.chatann import ChatANN


class ChatPDF(ChatANN):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_index(self, file_or_text, textsplitter=textsplitter):  # todo 多篇 增加 parser loader

        texts = extract_text(file_or_text)
        texts = textsplitter(texts)
        return super().create_index(texts)
    
    def create_index_list(self, file_list, textsplitter=textsplitter):  # todo 多篇 增加 parser loader
        text_array = list()
        for file in file_list:
            texts = extract_text(file)
            texts = textsplitter(texts)
            text_array.extend(texts)
        return super().create_index(text_array)


if __name__ == '__main__':
    # from chatllm.applications.chatpdf import ChatPDF
    qa = ChatPDF(encode_model='nghuyong/ernie-3.0-nano-zh')  # 自动建索引
    # qa.create_index('../../data/财报.pdf')
    # qa.load_llm4chat(model_name_or_path='/Users/betterme/PycharmProjects/AI/CHAT_MODEL/chatglm', device='cpu')
    # qa.create_index('data/财报.pdf')
    # sources = ['data/接种率文献/应用Kaplan-Meie...市首剂含麻疹成份疫苗接种率_王萌.pdf', 'data/接种率文献/卡普兰-迈耶(Kaplan...成分疫苗接种率评估中的应用_姜晓飞.pdf']
    # sources = 'data/接种率文献/卡普兰-迈耶(Kaplan...成分疫苗接种率评估中的应用_姜晓飞.pdf'
    # directory = 'data/接种率文献/'  # Change to your directory
    # directory = 'D:/huajun/chatGLM/chatGLM/track2-data/track2-问答式科研知识库'
    # directory = '../chatGLM/track2-data/track2-问答式科研知识库'
    # sources = glob.glob(directory + '*.pdf')
    sources = ['../chatGLM/track2-data/track2-问答式科研知识库/1.pdf', '../chatGLM/track2-data/track2-问答式科研知识库/2.pdf']
    print(sources)
    query = 'LLama和GPT两个模型在技术路线上有何不同'
    # query = '延迟接种属于疫苗犹豫吗？'

    # for understanding how code works
    # bytes_array = Path(sources).read_bytes()
    # texts = extract_text(bytes_array)
    # texts = textsplitter(texts)
    # for i, text in enumerate(texts):
    #     print(f"split text {i}: {text}")

    # qa.create_index(sources)
    qa.create_index_list(sources)
    qa.load_llm4chat(model_name_or_path='D:\huajun\chatGLM\chatGLM\chatglm-6B', device='cuda')

    # list(qa(query='东北证券主营业务', topk=1, threshold=0.8))
    list(qa(query=query, topk=3, threshold=0.8))

    # 召回结果
    print(f"Ask sources are: {sources}")
    print(f"query is {query}")
    print(qa.recall)
    print(type(qa.recall))
    print(qa.recall['text'])
