from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Tair
from langchain.document_loaders import UnstructuredFileLoader, TextLoader
from configs.model_config import *
from textsplitter import ChineseTextSplitter
from typing import List, Tuple, Dict
from langchain.docstore.document import Document
import numpy as np
from tqdm import tqdm
from models.bash import (BaseAnswer,
                         AnswerResult)
from langchain.docstore.document import Document
from functools import lru_cache
from .tair_session import get_tair_session
import uuid

chat_session = get_tair_session(TAIR_URL)

# patch HuggingFaceEmbeddings to make it hashable
def _embeddings_hash(self):
    return hash(self.model_name)

HuggingFaceEmbeddings.__hash__ = _embeddings_hash

def load_vector_store_tair_session(session_index_name, embeddings):
    return Tair(embedding_function = embeddings, url = TAIR_URL, index_name = session_index_name)

# will keep CACHED_VS_NUM of vector store caches
@lru_cache(CACHED_VS_NUM)
def load_vector_store_tair(vs_path, embeddings):
    vs_id = str(vs_path).split(VS_ROOT_PATH + "/")[1]
    return Tair(embedding_function = embeddings, url = TAIR_URL, index_name = vs_id)

def tree(filepath, ignore_dir_names=None, ignore_file_names=None):
    """返回两个列表，第一个列表为 filepath 下全部文件的完整路径, 第二个为对应的文件名"""
    if ignore_dir_names is None:
        ignore_dir_names = []
    if ignore_file_names is None:
        ignore_file_names = []
    ret_list = []
    if isinstance(filepath, str):
        if not os.path.exists(filepath):
            print("路径不存在")
            return None, None
        elif os.path.isfile(filepath) and os.path.basename(filepath) not in ignore_file_names:
            return [filepath], [os.path.basename(filepath)]
        elif os.path.isdir(filepath) and os.path.basename(filepath) not in ignore_dir_names:
            for file in os.listdir(filepath):
                fullfilepath = os.path.join(filepath, file)
                if os.path.isfile(fullfilepath) and os.path.basename(fullfilepath) not in ignore_file_names:
                    ret_list.append(fullfilepath)
                if os.path.isdir(fullfilepath) and os.path.basename(fullfilepath) not in ignore_dir_names:
                    ret_list.extend(tree(fullfilepath, ignore_dir_names, ignore_file_names)[0])
    return ret_list, [os.path.basename(p) for p in ret_list]

def load_file(filepath, sentence_size=SENTENCE_SIZE):
    if filepath.lower().endswith(".md"):
        loader = UnstructuredFileLoader(filepath, mode="elements")
        docs = loader.load()
    elif filepath.lower().endswith(".txt"):
        loader = TextLoader(filepath, autodetect_encoding=True)
        textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
        docs = loader.load_and_split(textsplitter)
    else:
        loader = UnstructuredFileLoader(filepath, mode="elements")
        textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
        docs = loader.load_and_split(text_splitter=textsplitter)
    write_check_file(filepath, docs)
    return docs

def write_check_file(filepath, docs):
    folder_path = os.path.join(os.path.dirname(filepath), "tmp_files")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    fp = os.path.join(folder_path, 'load_file.txt')
    with open(fp, 'a+', encoding='utf-8') as fout:
        fout.write("filepath=%s,len=%s" % (filepath, len(docs)))
        fout.write('\n')
        for i in docs:
            fout.write(str(i))
            fout.write('\n')
        fout.close()

def generate_prompt(related_docs: List[str],
                    query: str,
                    prompt_template: str = PROMPT_TEMPLATE, ) -> str:
    context = "\n".join([doc.page_content for doc in related_docs])
    prompt = prompt_template.replace("{question}", query).replace("{context}", context)
    return prompt

def generate_prompt_session(related_docs: List[str],
                    query: str,
                    prompt_template: str = PROMPT_TEMPLATE_SESSION, ) -> str:
    context = "\n".join([doc.page_content for doc in related_docs])
    prompt = prompt_template.replace("{question}", query).replace("{context}", context)
    return prompt

def seperate_list(ls: List[int]) -> List[List[int]]:
    lists = []
    ls1 = [ls[0]]
    for i in range(1, len(ls)):
        if ls[i - 1] + 1 == ls[i]:
            ls1.append(ls[i])
        else:
            lists.append(ls1)
            ls1 = [ls[i]]
    lists.append(ls1)
    return lists

def search_result2docs(search_results):
    docs = []
    for result in search_results:
        doc = Document(page_content=result["snippet"] if "snippet" in result.keys() else "",
                       metadata={"source": result["link"] if "link" in result.keys() else "",
                                 "filename": result["title"] if "title" in result.keys() else ""})
        docs.append(doc)
    return docs


class LocalDocQA:
    llm: BaseAnswer = None
    embeddings: object = None
    top_k: int = VECTOR_SEARCH_TOP_K
    chunk_size: int = CHUNK_SIZE
    chunk_conent: bool = True
    score_threshold: int = VECTOR_SEARCH_SCORE_THRESHOLD

    def init_cfg(self,
                 embedding_model: str = EMBEDDING_MODEL,
                 embedding_device=EMBEDDING_DEVICE,
                 llm_model: BaseAnswer = None,
                 top_k=VECTOR_SEARCH_TOP_K,
                 ):
        self.llm = llm_model
        # mac本机启动，临时注释
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[embedding_model], model_kwargs={'device': embedding_device})
        self.top_k = top_k

    def init_knowledge_vector_store_tair(self,
                                    filepath: str or List[str],
                                    vs_path: str or os.PathLike = None,
                                    sentence_size=SENTENCE_SIZE):
        
        vs_id = str(vs_path).split(VS_ROOT_PATH + "/")[1]
        loaded_files = []
        failed_files = []
        if isinstance(filepath, str):
            if not os.path.exists(filepath):
                print("路径不存在")
                return None
            elif os.path.isfile(filepath):
                file = os.path.split(filepath)[-1]
                try:
                    docs = load_file(filepath, sentence_size)
                    logger.info(f"{file} 已成功加载")
                    loaded_files.append(filepath)
                except Exception as e:
                    logger.error(e)
                    logger.info(f"{file} 未能成功加载")
                    return None
            elif os.path.isdir(filepath):
                docs = []
                for fullfilepath, file in tqdm(zip(*tree(filepath, ignore_dir_names=['tmp_files'])), desc="加载文件"):
                    try:
                        docs += load_file(fullfilepath, sentence_size)
                        loaded_files.append(fullfilepath)
                    except Exception as e:
                        logger.error(e)
                        failed_files.append(file)

                if len(failed_files) > 0:
                    logger.info("以下文件未能成功加载：")
                    for file in failed_files:
                        logger.info(f"{file}\n")

        else:
            docs = []
            for file in filepath:
                try:
                    docs += load_file(file)
                    logger.info(f"{file} 已成功加载")
                    loaded_files.append(file)
                except Exception as e:
                    logger.error(e)
                    logger.info(f"{file} 未能成功加载")
                    
        # 是否使用混合索引
        f = open(os.path.join(VS_ROOT_PATH, vs_id, "flag"), "r")
        flag = f.readline()
        f.close()        
        if len(docs) > 0:
            logger.info("文件加载完毕，正在生成向量库")
            # 是否使用混合检索
            '''
            参考文档：https://help.aliyun.com/zh/tair/developer-reference/vector
            Tair.from_documents:本质上会检查索引是否存在，不存在则调用 TVS.CREATEINDEX 创建索引，
                                再调用 TVS.HSET 写入数据
            '''
            if str(flag).strip() == "0":
                Tair.from_documents(docs, self.embeddings, index_name=vs_id, tair_url=TAIR_URL)
            else:
                Tair.from_documents(docs, self.embeddings, index_name=vs_id, tair_url=TAIR_URL, 
                                    index_params={"lexical_algorithm":"bm25"})
            return vs_path, loaded_files
        else:
            logger.info("文件均未成功加载，请检查依赖包或替换为其他文件再次上传。")
            return None, loaded_files

    def get_knowledge_based_answer_tair(self, query, vs_path, chat_history=[], use_hybrid_search: bool = False, hybrid_search_type_flag: float = 0.5,streaming: bool = STREAMING):
        vector_store = load_vector_store_tair(vs_path, self.embeddings)
        # 混合检索参数
        kwargs = {"TEXT" : query, "hybrid_ratio" : hybrid_search_type_flag}
        related_docs_with_score = []
        '''
        参考文档：https://help.aliyun.com/zh/tair/developer-reference/vector
        similarity_search:本质上是调用的 TVS.KNNSEARCH 进行相似检索
        '''
        if use_hybrid_search:
            related_docs_with_score = vector_store.similarity_search(query, k=self.top_k, **kwargs)
        else:
            related_docs_with_score = vector_store.similarity_search(query, k=self.top_k)
        if len(related_docs_with_score)>0:
            prompt = generate_prompt(related_docs_with_score, query)
        else:
            prompt = query
        # 同问题一起Prompt提交大模型
        for answer_result in self.llm.generatorAnswer(prompt=prompt, history=chat_history,
                                                      streaming=streaming):
            resp = answer_result.llm_output["answer"]
            history = answer_result.history
            history[-1][0] = query
            history[-1][1] = resp
            response = {"query": query,
                        "result": resp,
                        "source_documents": related_docs_with_score}
            yield response, history    
            
    # tair_session
    def get_prompt_by_tair_session(self, query: str, session_id):
        # SESSION_BEGIN如("已知", "根据", "依据", "请根据", "请依据")，直接不查询，根据提示让大模型回答
        if query is not None and query.startswith(SESSION_BEGIN):
            return query
        if chat_session.not_exists_index(session_id):
            prompt = query
            return prompt
        vector_store = load_vector_store_tair_session(session_id, self.embeddings)
        related_docs_with_score = vector_store.similarity_search(query, k=self.top_k)
        if len(related_docs_with_score)>0:
            prompt = generate_prompt_session(related_docs_with_score, query)
        else:
            prompt = query
        return prompt
    
    def insert_tair_session(self, query: str, resp, session_id):
        # 仅有SESSION_BEGIN如("已知", "根据", "依据", "请根据", "请依据")开头的提示才把问题写入缓存
        if query is not None and query.startswith(SESSION_BEGIN):
            text = f"{query}"
            # 写入session缓存
            key = uuid.uuid4().hex
            Tair.from_texts([text], self.embeddings, None, session_id, "content", "metadata", tair_url=TAIR_URL, index_type="FLAT", keys=[key])
            # 设置缓存过期时间
            chat_session.tvs_hexpire(session_id, key, SESSION_HEXPIRE_TIME)