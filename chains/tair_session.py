from tair import Tair as TairClient

class ChatBotSession():
    def __init__(self, url):
        try:
            # connect to tair from url
            self.client = TairClient.from_url(url)
        except ValueError as e:
            raise ValueError(f"Tair failed to connect: {e}")
    # time out del, 5 minute
    def expires(self, index_name):
        self.client.expire(index_name, 300)
        
    # 不存在返回True, 存在返回False
    def not_exists_index(self, index_name):
        index = self.client.tvs_get_index(index_name)
        if index is not None:
            return False
        return True

def get_tair_session(url):
    return ChatBotSession(url)