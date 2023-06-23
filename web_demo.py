# encoding=utf-8
'''
Author: xupingmao
email: 578749341@qq.com
Date: 2023-04-15 10:23:28
LastEditors: xupingmao
LastEditTime: 2023-06-23 12:51:17
FilePath: \ChatGLM-app\web_demo.py
Description: 描述
'''

from transformers import AutoModel, AutoTokenizer
import gradio as gr
import mdtex2html
import os
import torch
import sys
from torch.cuda import get_device_properties
from app_conf import get_src_root_dir


sys.path.append("./ChatGLM-6B")

# Fix错误： RuntimeError: Library cudart is not initialized
torch_dir = os.path.dirname(torch.__file__)
os.environ['PATH'] = os.environ.get("PATH", "") + os.pathsep + os.path.join(torch_dir, "lib")
default_model_path = os.path.join(get_src_root_dir(), "THUDM/chatglm-6b")
default_model_path = os.path.abspath(default_model_path)

class CmdOptions:
    precision = None


class WebDemo:
    tokenizer = None
    model = None


def load_model(path = default_model_path):
    print(f"load model {path} ...")
    """加载模型, 模型用的是transformers的框架"""
    cmd_opts = CmdOptions()
    if cmd_opts.precision is None:
        total_vram_in_gb = get_device_properties(0).total_memory / 1e9
        print(f'GPU memory: {total_vram_in_gb:.2f} GB')

        if total_vram_in_gb > 30:
            cmd_opts.precision = 'fp32'
        elif total_vram_in_gb > 13:
            cmd_opts.precision = 'fp16'
        elif total_vram_in_gb > 10:
            cmd_opts.precision = 'int8'
        else:
            cmd_opts.precision = 'int4'

        print(f'Choosing precision {cmd_opts.precision} according to your VRAM.'
                f' If you want to decide precision yourself,'
                f' please add argument --precision when launching the application.')

    print(f"using precision {cmd_opts.precision}")

    kw = dict(trust_remote_code=True, revision="main")
    WebDemo.tokenizer = AutoTokenizer.from_pretrained(path, **kw)
    model = AutoModel.from_pretrained(path, **kw)

    if cmd_opts.precision == "fp16":
        model = model.half().cuda()
    elif cmd_opts.precision == "int4":
        model = model.half().quantize(4).cuda()
    elif cmd_opts.precision == "int8":
        model = model.half().quantize(8).cuda()
    elif cmd_opts.precision == "fp32":
        model = model.float()

    model = model.eval()
    WebDemo.model = model
    return model

"""Override Chatbot.postprocess"""

model = load_model()

def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def predict(input, chatbot, max_length, top_p, temperature, history):
    """处理对话的函数"""
    chatbot.append((parse_text(input), ""))
    for response, history in model.stream_chat(WebDemo.tokenizer, input, history, max_length=max_length, top_p=top_p,
                                               temperature=temperature):
        chatbot[-1] = (parse_text(input), parse_text(response))       

        yield chatbot, history


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], []


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">ChatGLM</h1>""")

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                    container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(0, 4096, value=2048, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

    history = gr.State([])

    submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history], [chatbot, history],
                    show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

demo.queue().launch(share=False, inbrowser=True)
