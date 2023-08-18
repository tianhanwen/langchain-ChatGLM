from tair import Tair as TairClient
from configs.model_config import *

class ChatBotSession():
    def __init__(self, url):
        try:
            # connect to tair from url
            self.client = TairClient.from_url(url)
        except ValueError as e:
            raise ValueError(f"Tair failed to connect: {e}")
        
    def tvs_hexpire(self, index_name, key, time = SESSION_HEXPIRE_TIME):
        self.client.tvs_hexpire(index_name, key, time)
        
    # 不存在返回True, 存在返回False
    def not_exists_index(self, index_name):
        index = self.client.tvs_get_index(index_name)
        if index is not None:
            return False
        return True

def get_tair_session(url):
    return ChatBotSession(url)