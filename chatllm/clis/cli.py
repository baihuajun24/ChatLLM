#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : MeUtils.
# @File         : __init__.py
# @Time         : 2021/1/31 10:20 下午
# @Author       : yuanjie
# @Email        : meutils@qq.com
# @Software     : PyCharm
# @Description  : python meutils/clis/__init__.py
import os

from meutils.pipe import *

cli = typer.Typer(name="ChatLLM CLI")

if LOCAL_HOST.startswith('10.219'):
    MODEL_PATH = "/Users/betterme/PycharmProjects/AI/CHAT_MODEL/chatglm"


def f(a=1, **kw):
    print(a)
    print(kw)


@cli.command(help="help")  # help会覆盖docstring
def clitest(**kwargs): # 不支持 **kwargs
    f(**kwargs)


@cli.command()  # help会覆盖docstring
def webui(name: str = 'chatpdf', port=8501):
    """
        chatllm-run webui --name chatpdf --port 8501
    """
    main = get_resolve_path(f'../webui/{name}.py', __file__)
    os.system(f'streamlit run {main} --server.port {port}')


@cli.command()  # help会覆盖docstring
def flask_api(model_name_or_path=None, host='127.0.0.1', port=8000, path='/'):
    """
        chatllm-run flask-api --model_name_or_path <MODEL_PATH> --host 127.0.0.1 --port 8000
    """
    from chatllm.applications import ChatBase

    qa = ChatBase()
    qa.load_llm4chat(model_name_or_path or MODEL_PATH)
    qa.run_serving(host, port, path)


if __name__ == '__main__':
    cli()
