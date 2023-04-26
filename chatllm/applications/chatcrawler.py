#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : AI.  @by PyCharm
# @File         : crawler4qa
# @Time         : 2023/4/24 18:17
# @Author       : betterme
# @WeChat       : meutils
# @Software     : PyCharm
# @Description  :


from meutils.pipe import *
from meutils.request_utils.crawler import Crawler

from chatllm.applications import ChatBase


class Crawler4QA(ChatBase):

    def __init__(self, chat_func):
        super().__init__(chat_func)

    def qa(self, query,
           url="https://top.baidu.com/board?tab=realtime",
           xpath='//*[@id="sanRoot"]/main/div[2]/div/div[2]/div[*]/div[2]/a/div[1]//text()', **kwargs):
        knowledge_base = Crawler(url).xpath(xpath)  # 爬虫获取知识库

        return self._qa(query, knowledge_base, **kwargs)


if __name__ == '__main__':
    from chatllm.utils import load_llm4chat

    chat_func = load_llm4chat(
        model_name_or_path="/Users/betterme/PycharmProjects/AI/CHAT_MODEL/chatglm",
        device='mps',
    )

    qa = Crawler4QA(chat_func=chat_func)

    for i, _ in qa(query='提取人名'):
        print(i, flush=True)
        sys.stdout.flush()

    pprint(qa.knowledge_base)