import gradio as gr
import os
import shutil

from chains.local_doc_qa import LocalDocQA
from configs.model_config import *
import models.shared as shared
import nltk

nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path

use_hybrid_search_list = ["否", "是"]

def get_vs_list():
    lst_default = ["新建知识库"]
    if not os.path.exists(VS_ROOT_PATH):
        return lst_default
    lst = os.listdir(VS_ROOT_PATH)
    if not lst:
        return lst_default
    lst.sort()
    return lst_default + lst

def get_hybrid_chose(vs_name):
    if not os.path.exists(VS_ROOT_PATH):
        return use_hybrid_search_list
    lst = os.listdir(VS_ROOT_PATH + "/" + vs_name)
    if not lst:
        return use_hybrid_search_list
    f = open(os.path.join(VS_ROOT_PATH, vs_name, "flag"), "r")
    flag = f.readline()
    f.close()
    if str(flag).strip() == "0":
        return ["否"]
    else:
        return ["是"]
    
def get_default_hybrid_chose(vs_name):
    if not os.path.exists(VS_ROOT_PATH):
        return USE_HYBRID_SEARCH
    lst = os.listdir(VS_ROOT_PATH + "/" + vs_name)
    if not lst:
        return USE_HYBRID_SEARCH
    f = open(os.path.join(VS_ROOT_PATH, vs_name, "flag"), "r")
    flag = f.readline()
    f.close()
    if str(flag).strip() == "0":
        return "否"
    else:
        return "是"

# 将文本编码为向量
embedding_model_dict_list = list(embedding_model_dict.keys())

# 获取模型集合，通过API_KEY调用的就只有通义千问模型
llm_model_dict_list = list(llm_model_key_dict.keys())

local_doc_qa = LocalDocQA()

flag_csv_logger = gr.CSVLogger()


def get_answer(query: str, vs_path, history, mode, use_hybrid_search, score_threshold=VECTOR_SEARCH_SCORE_THRESHOLD,
               vector_search_top_k=VECTOR_SEARCH_TOP_K, chunk_conent: bool = True,
               chunk_size=CHUNK_SIZE, streaming: bool = STREAMING):
    
    if query is None or len(query.strip()) == 0:
        query = "你好"
        
    if len(history) > 120:
        del history[0:100]
        
    if mode == "带Session的对话":
        # get session_id: TODO: get by client
        session_id = SESSION_ID
        # 1. get prompt
        resp = ""
        prompt = local_doc_qa.get_prompt_by_tair_session(query=query, session_id=session_id)
        for answer_result in local_doc_qa.llm.generatorAnswer(prompt=prompt, history=history,
                                                      streaming=streaming):
            resp = answer_result.llm_output["answer"]
            history = answer_result.history
            history[-1][0] = query
            history[-1][-1] = resp
            yield history, ""
        local_doc_qa.insert_tair_session(query=query, resp=resp, session_id=session_id)    
    elif mode == "知识库问答" and vs_path is not None and os.path.exists(vs_path):
        use_hybrid = False
        if use_hybrid_search == "是":
            use_hybrid = True
        for resp, history in local_doc_qa.get_knowledge_based_answer_tair(
                query=query, vs_path=vs_path, chat_history=history, use_hybrid_search = use_hybrid, streaming=streaming):
            source = "\n\n"
            source += "".join(
                [f"""<details> <summary>出处 [{i + 1}] {os.path.split(doc.metadata["source"])[-1]}</summary>\n"""
                 f"""{doc.page_content}\n"""
                 f"""</details>"""
                 for i, doc in
                 enumerate(resp["source_documents"])])
            history[-1][-1] += source
            yield history, ""
    else:
        for answer_result in local_doc_qa.llm.generatorAnswer(prompt=query, history=history,
                                                              streaming=streaming):
            resp = answer_result.llm_output["answer"]
            history = answer_result.history
            history[-1][-1] = resp
            yield history, ""
    logger.info(f"flagging: username={FLAG_USER_NAME},query={query},vs_path={vs_path},mode={mode},history={history}")
    flag_csv_logger.flag([query, vs_path, history, mode], username=FLAG_USER_NAME)

def init_model():
    llm_model_ins = shared.loaderLLM()
    llm_model_ins.set_history_len(LLM_HISTORY_LEN)
    try:
        local_doc_qa.init_cfg(llm_model=llm_model_ins)
        generator = local_doc_qa.llm.generatorAnswer("你好")
        for answer_result in generator:
            print(answer_result.llm_output)
        reply = """模型已成功加载，可以开始对话，或从右侧选择模式后开始对话"""
        logger.info(reply)
        return reply
    except Exception as e:
        logger.error(e)
        reply = """模型未成功加载，请到页面左上角"模型配置"选项卡中重新选择后点击"加载模型"按钮"""
        if str(e) == "Unknown platform: darwin":
            logger.info("该报错可能因为您使用的是 macOS 操作系统，需先下载模型至本地后执行 Web UI，具体方法请参考项目 README 中本地部署方法及常见问题："
                        " https://github.com/imClumsyPanda/langchain-ChatGLM")
        else:
            logger.info(reply)
        return reply

def reinit_model(llm_model, embedding_model, llm_history_len, top_k,
                 history):
    try:
        llm_model_ins = shared.loaderLLM(llm_model)
        llm_model_ins.history_len = llm_history_len
        local_doc_qa.init_cfg(llm_model=llm_model_ins,
                              embedding_model=embedding_model,
                              top_k=top_k)
        model_status = """模型已成功重新加载，可以开始对话，或从右侧选择模式后开始对话"""
        logger.info(model_status)
    except Exception as e:
        logger.error(e)
        model_status = """模型未成功重新加载，请到页面左上角"模型配置"选项卡中重新选择后点击"加载模型"按钮"""
        logger.info(model_status)
    return history + [[None, model_status]]

def get_vector_store(vs_id, files, sentence_size, history, one_conent, one_content_segmentation):
    vs_path = os.path.join(VS_ROOT_PATH, vs_id)
    filelist = []
    if local_doc_qa.llm and local_doc_qa.embeddings:
        if isinstance(files, list):
            for file in files:
                filename = os.path.split(file.name)[-1]
                shutil.move(file.name, os.path.join(UPLOAD_ROOT_PATH, vs_id, filename))
                filelist.append(os.path.join(UPLOAD_ROOT_PATH, vs_id, filename))
            #vs_path, loaded_files = local_doc_qa.init_knowledge_vector_store(filelist, vs_path, sentence_size)
            vs_path, loaded_files = local_doc_qa.init_knowledge_vector_store_tair(filelist, vs_path, sentence_size)
        else:
            vs_path, loaded_files = local_doc_qa.one_knowledge_add(vs_path, files, one_conent, one_content_segmentation,
                                                                   sentence_size)
        if len(loaded_files):
            file_status = f"已添加 {'、'.join([os.path.split(i)[-1] for i in loaded_files if i])} 内容至知识库，并已加载知识库，请开始提问"
        else:
            file_status = "文件未成功加载，请重新上传文件"
    else:
        file_status = "模型未完成加载，请先在加载模型后再导入文件"
        vs_path = None
    logger.info(file_status)
    return vs_path, None, history + [[None, file_status]]


def change_vs_name_input(vs_id, history):
    if vs_id == "新建知识库" or vs_id is None:
        return gr.update(visible=True), gr.update(choices = use_hybrid_search_list), gr.update(visible=True), gr.update(visible=False), None, history
    else:
        vs_path = os.path.join(VS_ROOT_PATH, vs_id)
        file_status = f"已加载知识库{vs_id}，请开始提问"
        return gr.update(visible=False), gr.update(
                choices=get_hybrid_chose(vs_id), value=get_default_hybrid_chose(vs_id)
            ),gr.update(visible=False), gr.update(visible=True), \
               vs_path, history + [[None, file_status]]

def change_mode(mode, history):
    if mode == "知识库问答":
        return gr.update(visible=True), gr.update(visible=False), history
        # + [[None, "【注意】：您已进入知识库问答模式，您输入的任何查询都将进行知识库查询，然后会自动整理知识库关联内容进入模型查询！！！"]]
    else:
        return gr.update(visible=False), gr.update(visible=False), history


def change_chunk_conent(mode, label_conent, history):
    conent = ""
    if "chunk_conent" in label_conent:
        conent = "搜索结果上下文关联"
    elif "one_content_segmentation" in label_conent:  # 这里没用上，可以先留着
        conent = "内容分段入库"

    if mode:
        return gr.update(visible=True), history + [[None, f"【已开启{conent}】"]]
    else:
        return gr.update(visible=False), history + [[None, f"【已关闭{conent}】"]]


def add_vs_name(vs_name, use_hybrid_search, chatbot):
    if vs_name in get_vs_list():
        vs_status = "与已有知识库名称冲突，请重新选择其他名称后提交"
        chatbot = chatbot + [[None, vs_status]]
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(
            visible=False), chatbot
    else:
        # 新建上传文件存储路径
        if not os.path.exists(os.path.join(UPLOAD_ROOT_PATH, vs_name)):
            os.makedirs(os.path.join(UPLOAD_ROOT_PATH, vs_name))
        # 新建向量库存储路径
        if not os.path.exists(os.path.join(VS_ROOT_PATH, vs_name)):
            os.makedirs(os.path.join(VS_ROOT_PATH, vs_name))
            f = open(os.path.join(VS_ROOT_PATH, vs_name, "flag"), "w")
            if str(use_hybrid_search).strip() == "是":
                f.write("1")
            else:
                f.write("0")
            f.close()
        vs_status = f"""已新增知识库"{vs_name}",将在上传文件并载入成功后进行存储。请在开始对话前，先完成文件上传。 """
        chatbot = chatbot + [[None, vs_status]]
        return gr.update(visible=True, choices=get_vs_list(), value=vs_name
                         ), gr.update(visible=False), gr.update(
                                choices=get_hybrid_chose(vs_name), value = get_default_hybrid_chose(vs_name)
                             ), gr.update(visible=False), gr.update(visible=True), chatbot

# 自动化加载固定文件间中文件
def reinit_vector_store(vs_id, history):
    try:
        shutil.rmtree(VS_ROOT_PATH)
        vs_path = os.path.join(VS_ROOT_PATH, vs_id)
        sentence_size = gr.Number(value=SENTENCE_SIZE, precision=0,
                                  label="文本入库分句长度限制",
                                  interactive=True, visible=True)
        vs_path, loaded_files = local_doc_qa.init_knowledge_vector_store(UPLOAD_ROOT_PATH, vs_path, sentence_size)
        model_status = """知识库构建成功"""
    except Exception as e:
        logger.error(e)
        model_status = """知识库构建未成功"""
        logger.info(model_status)
    return history + [[None, model_status]]


def refresh_vs_list():
    return gr.update(choices=get_vs_list())


block_css = """.importantButton {
    background: linear-gradient(45deg, #7e0570,#5d1c99, #6e00ff) !important;
    border: none !important;
}
.importantButton:hover {
    background: linear-gradient(45deg, #ff00e0,#8500ff, #6e00ff) !important;
    border: none !important;
}"""

webui_title = """ 
# 基于Langchain + Tair + 通义千问的Chatbot
"""
default_vs = get_vs_list()[0] if len(get_vs_list()) > 1 else "为空"
init_message = f"""本地部署开源大模型，通常对机器的配置要求较高，用户部署成本较高。本方案中，通过使用ApiKey的方式，快速调用通义千问，结合Tair向量数据库，帮助用户低成本构建专属ChatBot。\n
LLM对话：直接对话通义千问\n
知识库问答：基于选择的知识库进行特定领域的知识问答，通义千问+Tair构建私域Chatbot\n
带Session对话：借助Tair的高性能缓存历史提示信息，给大模型提示需要以 ("已知", "根据", "依据", "请根据", "请依据") 开头, 且每个提示信息缓存1小时后失效。
"""

# 初始化消息
# TODO: 修改为API_KEY方式调用
model_status = init_model()

default_theme_args = dict(
    font=["Source Sans Pro", 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
)

with gr.Blocks(css=block_css, theme=gr.themes.Default(**default_theme_args)) as demo:
    vs_path, file_status, model_status = gr.State(
        os.path.join(VS_ROOT_PATH, get_vs_list()[0]) if len(get_vs_list()) > 1 else ""), gr.State(""), gr.State(
        model_status)
    gr.Markdown(webui_title)
    with gr.Tab("对话"):
        with gr.Row():
            with gr.Column(scale=10):
                chatbot = gr.Chatbot([[None, init_message], [None, model_status.value]],
                                     elem_id="chat-box",
                                     show_label=False).style(height=750)
                query = gr.Textbox(show_label=False,
                                   placeholder="请输入提问内容，按回车进行提交").style(container=False)
            with gr.Column(scale=5):
                mode = gr.Radio(["LLM对话", "知识库问答", "带Session的对话"],
                                label="请选择使用模式",
                                value="知识库问答", )
                knowledge_set = gr.Accordion("知识库设定", visible=False)
                vs_setting = gr.Accordion("配置知识库")
                mode.change(fn=change_mode,
                            inputs=[mode, chatbot],
                            outputs=[vs_setting, knowledge_set, chatbot])
                with vs_setting:
                    vs_refresh = gr.Button("更新已有知识库选项")
                    select_vs = gr.Dropdown(get_vs_list(),
                                            label="请选择要加载的知识库",
                                            default = "新建知识库",
                                            interactive=True,
                                            value=get_vs_list()[0] if len(get_vs_list()) > 0 else None
                                            )
                    vs_name = gr.Textbox(label="请输入新建知识库名称，当前知识库命名暂不支持中文",
                                         lines=1,
                                         interactive=True,
                                         visible=True)
                    use_hybrid_search = gr.Radio(use_hybrid_search_list,
                                                label="是否开启混合检索",
                                                value=USE_HYBRID_SEARCH,
                                                interactive=True)
                    vs_add = gr.Button(value="添加至知识库选项", visible=True)
                    file2vs = gr.Column(visible=False)
                    with file2vs:
                        # load_vs = gr.Button("加载知识库")
                        gr.Markdown("向知识库中添加文件")
                        sentence_size = gr.Number(value=SENTENCE_SIZE, precision=0,
                                                  label="文本入库分句长度限制",
                                                  interactive=True, visible=True)
                        with gr.Tab("上传文件"):
                            files = gr.File(label="添加文件",
                                            file_types=['.txt', '.md', '.docx', '.pdf', '.png', '.jpg'],
                                            file_count="multiple",
                                            show_label=False)
                            load_file_button = gr.Button("上传文件并加载知识库")
                        with gr.Tab("上传文件夹"):
                            folder_files = gr.File(label="添加文件",
                                                   file_count="directory",
                                                   show_label=False)
                            load_folder_button = gr.Button("上传文件夹并加载知识库")
                    vs_refresh.click(fn=refresh_vs_list,
                                     inputs=[],
                                     outputs=select_vs)
                    vs_add.click(fn=add_vs_name,
                                 inputs=[vs_name, use_hybrid_search, chatbot],
                                 outputs=[select_vs, vs_name, use_hybrid_search, vs_add, file2vs, chatbot])
                    select_vs.change(fn=change_vs_name_input,
                                     inputs=[select_vs, chatbot],
                                     outputs=[vs_name, use_hybrid_search, vs_add, file2vs, vs_path, chatbot])
                    load_file_button.click(get_vector_store,
                                           show_progress=True,
                                           inputs=[select_vs, files, sentence_size, chatbot, vs_add, vs_add],
                                           outputs=[vs_path, files, chatbot], )
                    load_folder_button.click(get_vector_store,
                                             show_progress=True,
                                             inputs=[select_vs, folder_files, sentence_size, chatbot, vs_add,
                                                     vs_add],
                                             outputs=[vs_path, folder_files, chatbot], )
                    flag_csv_logger.setup([query, vs_path, chatbot, mode], "flagged")
                    query.submit(get_answer,
                                 [query, vs_path, chatbot, mode, use_hybrid_search],
                                 [chatbot, query])

    with gr.Tab("模型配置"):
        llm_model = gr.Radio(llm_model_dict_list,
                             label="LLM 模型",
                             value=LLM_KEY_MODEL,
                             interactive=True)
        
        llm_history_len = gr.Slider(0, 10,
                                    value=LLM_HISTORY_LEN,
                                    step=1,
                                    label="LLM 对话轮数",
                                    interactive=True)
        embedding_model = gr.Radio(embedding_model_dict_list,
                                   label="Embedding 模型",
                                   value=EMBEDDING_MODEL,
                                   interactive=True)
        top_k = gr.Slider(1, 20, value=VECTOR_SEARCH_TOP_K, step=1,
                          label="向量匹配 top k", interactive=True)
        load_model_button = gr.Button("重新加载模型")
        load_model_button.click(reinit_model, show_progress=True,
                                inputs=[llm_model, embedding_model, llm_history_len,
                                        top_k, chatbot], outputs=chatbot)
        
    demo.load(
        fn=refresh_vs_list,
        inputs=None,
        outputs=[select_vs],
        queue=True,
        show_progress=False,
    )

(demo
 .queue(concurrency_count=3)
 .launch(server_name='0.0.0.0',
         server_port=5001,
         show_api=False,
         share=False,
         inbrowser=False))
