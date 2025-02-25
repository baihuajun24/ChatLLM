#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : AI.  @by PyCharm
# @File         : ann4qa
# @Time         : 2023/4/24 18:10
# @Author       : betterme
# @WeChat       : meutils
# @Software     : PyCharm
# @Description  :

from meutils.pipe import *
from chatllm.applications import ChatBase

from docarray import DocList, BaseDoc
from docarray.typing import TorchTensor
from docarray.utils.find import find
from docarray.utils.filter import filter_docs
from sentence_transformers import SentenceTransformer


class ChatANN(ChatBase):

    def __init__(self, backend='in_memory', encode_model="nghuyong/ernie-3.0-nano-zh", **kwargs):
        """
        :param backend:
            'in_memory' # todo: 支持更多后端
        :param encode_model:
            "nghuyong/ernie-3.0-nano-zh"
            "shibing624/text2vec-base-chinese"
            "GanymedeNil/text2vec-large-chinese"
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.backend = backend
        self.encode = SentenceTransformer(encode_model, device='cpu').encode  # 加缓存，可重新set

        # create index
        self.index = None

        # 召回结果df
        self.recall = pd.DataFrame({'id': [], 'text': [], 'score': []})

    def qa(self, query, topk=3, threshold=0.66, **kwargs):
        df = self.find(query, topk, threshold)
        if len(df) == 0:
            logger.warning('召回内容为空!!!')

        knowledge_base = '\n'.join(df.text)
        return self._qa(query, knowledge_base, **kwargs)

    def find(self, query, topk=5, threshold=0.66):  # 返回df
        v = self.encode(query, convert_to_numpy=False)

        if self.backend == 'in_memory':
            # r = self.index.find(TorchTensor(v), topk)
            r = self.index.find(v, topk)
            self.recall = (
                # r.documents.to_dataframe() # bug
                # pd.DataFrame(json.loads(r.documents.to_json()))
                r.documents.to_dataframe()
                .assign(score=r.scores)
                .query(f'score > {threshold}')
            )

        return self.recall

    def create_index(self, texts):
        tensors = self.encode(texts, show_progress_bar=True, convert_to_numpy=False)

        class Document(BaseDoc):
            text: str
            embedding: TorchTensor

        self.index = DocList[Document]()
        self.index.extend([Document(text=text, embedding=tensor) for text, tensor in zip(texts, tensors)])
        self.index.find = lambda query, topk=3: find(self.index, query, limit=topk, search_field='embedding')

        return self.index


if __name__ == '__main__':

    qa = ChatANN(encode_model="nghuyong/ernie-3.0-nano-zh")
    qa.load_llm4chat(model_name_or_path="/Users/betterme/PycharmProjects/AI/CHAT_MODEL/chatglm")
    qa.create_index(['周杰伦'] * 10)

    for i, _ in qa(query='有几个周杰伦'):
        pass
    print(qa.recall)
