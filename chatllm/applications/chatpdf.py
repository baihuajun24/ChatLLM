#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : AI.  @by PyCharm
# @File         : ChatPDF
# @Time         : 2023/4/21 11:44
# @Author       : betterme
# @WeChat       : meutils
# @Software     : PyCharm
# @Description  :

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


if __name__ == '__main__':
    # filename = '../../data/财报.pdf'
    # bytes_array = Path(filename).read_bytes()
    # texts = extract_text(bytes_array)
    # texts = textsplitter(texts)
    # print(texts)
    from chatllm.applications.chatpdf import ChatPDF

    qa = ChatPDF(encode_model='nghuyong/ernie-3.0-nano-zh')  # 自动建索引
    # qa.create_index('../../data/财报.pdf')
    # qa.load_llm4chat(model_name_or_path='/Users/betterme/PycharmProjects/AI/CHAT_MODEL/chatglm', device='cpu')
    # qa.create_index('data/财报.pdf')
    source = 'data/接种率文献/应用Kaplan-Meie...市首剂含麻疹成份疫苗接种率_王萌.pdf'
    query = '延迟接种属于疫苗犹豫吗？'
    qa.create_index(source)
    # AssertionError: Torch not compiled with CUDA enabled
    qa.load_llm4chat(model_name_or_path='D:\huajun\chatGLM\chatGLM\chatglm-6B', device='cpu')

    # list(qa(query='东北证券主营业务', topk=1, threshold=0.8))
    list(qa(query=query, topk=1, threshold=0.8))

    # 召回结果
    print(f"Ask source is: {source}")
    print(f"query is {query}")
    print(qa.recall)
