import os
import logging
import uuid

LOG_FORMAT = "%(levelname) -5s %(asctime)s" "-1d: %(message)s"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format=LOG_FORMAT)

# 替换为本地目录
embedding_model_dict = {
    "text2vec": "/embedding/text2vec-large-chinese",
}

#it can find by os
TAIR_URL = "redis://127.0.0.1:6379"
if os.getenv("TAIR_URL") is not None:
    TAIR_URL = os.getenv("TAIR_URL")
    
API_KEY='test'
if os.getenv("DASHSCOPE_API_KEY") is not None:
    API_KEY = os.getenv("DASHSCOPE_API_KEY")
    
TAIR_SESSION_INDEX_PARAMS={"distance_type" : "FLAT"}

# Embedding model name
EMBEDDING_MODEL = "text2vec"

# supported LLM models
# llm_model_dict 处理了loader的一些预设行为，如加载位置，模型名称，模型处理器实例
llm_model_dict = {
    "chatglm-6b": {
        "name": "chatglm-6b",
        "pretrained_model_name": "THUDM/chatglm-6b",
        "local_model_path": "/root/chatglm/chatglm-6b",
        "provides": "ChatGLM"
    },
}

llm_model_key_dict = {
    "qwen-v1" : {
        "name": "qwen-v1",
        "provides": "AliyunQwen"
    }
}

# LLM 名称
LLM_MODEL = "chatglm-6b"

LLM_KEY_MODEL = "qwen-v1"

# 如果你需要加载本地的model，指定这个参数  ` --no-remote-model`，或者下方参数修改为 `True`
NO_REMOTE_MODEL = False
# 量化加载8bit 模型
LOAD_IN_8BIT = False
# Load the model with bfloat16 precision. Requires NVIDIA Ampere GPU.
BF16 = False
# 本地模型存放的位置
MODEL_DIR = "model/"
# 本地lora存放的位置
LORA_DIR = "loras/"

# LLM lora path，默认为空，如果有请直接指定文件夹路径
LLM_LORA_PATH = ""
USE_LORA = True if LLM_LORA_PATH else False

# LLM streaming reponse
STREAMING = True

# Use p-tuning-v2 PrefixEncoder
USE_PTUNING_V2 = False

VS_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vector_store")

UPLOAD_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "content")

# 基于上下文的prompt模版，请务必保留"{question}"和"{context}"
PROMPT_TEMPLATE = """已知信息：
{context} 

根据上述已知信息，专业的来回答用户的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 问题是：{question}"""

PROMPT_TEMPLATE_SESSION = """已知用户以前提供的问题信息：
{context} 

根据上述已知提示，回答用户问题，可以合理进行相关扩充。如果无法从中得到答案，请自行回答用户问题，答案请使用中文。 问题是：{question}"""

# 缓存知识库数量
CACHED_VS_NUM = 1

# 文本分句长度
SENTENCE_SIZE = 100

# 匹配后单段上下文长度
CHUNK_SIZE = 250

# LLM input history length
LLM_HISTORY_LEN = 3

# return top-k text chunk from vector store
VECTOR_SEARCH_TOP_K = 2

# 知识检索内容相关度 Score, 数值范围约为0-1100，如果为0，则不生效，经测试设置为小于500时，匹配结果更精准
VECTOR_SEARCH_SCORE_THRESHOLD = 0

NLTK_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nltk_data")

FLAG_USER_NAME = uuid.uuid4().hex

EMBEDDING_DEVICE = "cpu"

# query by key
LLM_DEVICE = "cpu"

SESSION_BEGIN = ("已知", "根据", "依据", "请根据", "请依据")

SESSION_HEXPIRE_TIME = 3600

SESSION_ID = "session"

logger.info(f"""
loading model config
llm device: {LLM_DEVICE}
embedding device: {EMBEDDING_DEVICE}
dir: {os.path.dirname(os.path.dirname(__file__))}
flagging username: {FLAG_USER_NAME}
""")

# 是否开启跨域，默认为False，如果需要开启，请设置为True
# is open cross domain
OPEN_CROSS_DOMAIN = False