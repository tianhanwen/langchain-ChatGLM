from abc import ABC, abstractmethod
from models.bash import BaseAnswer, AnswerResult
from typing import Optional, List
import dashscope
from dashscope import Generation
from http import HTTPStatus

class AliyunQwen(BaseAnswer, ABC):
    max_token: int = 1500
    history_len: int = 10
     
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    @property
    def _history_len(self) -> int:
        return self.history_len
    
    def set_history_len(self, history_len: int = 10) -> None:
        self.history_len = history_len
    
    def generatorAnswer(self, prompt: str,
                        history: List[List[str]] = [],
                        streaming: bool = False, use_history: bool = True):
        # 构造属于Qwen的history
        def parser_history(history: List[List[str]] = []):
            result_history = [] 
            if use_history == False:
                return result_history
            for tmp in history:
                if len(tmp) != 2 or tmp[0] is None or len(tmp[0].strip()) == 0:
                    continue
                qwen_history = {}
                qwen_history["user"] = tmp[0]
                qwen_history["bot"] = tmp[-1]
                result_history.append(qwen_history)
            return result_history
        if streaming:
            history += [[]]
            tmp_history = history[-self.history_len:-1] if self.history_len > 1 else []
            qwen_history = parser_history(tmp_history)
            responses=Generation.call(model=self.model_name, prompt=prompt, stream=True, history=qwen_history, max_length=self.max_token)
            for response in responses:
                answer_result = AnswerResult()
                if response.status_code==HTTPStatus.OK:
                    res = response.output.text
                    history[-1] = [prompt, res]
                    answer_result.history = history
                    answer_result.llm_output = {"answer": res}
                else:
                    print('Code: %d, status: %s, message: %s' % (response.status_code, response.code, response.message))
                    raise Exception(response.message)
                yield answer_result
        else:
            tmp_history=history[-self.history_len:] if self.history_len > 0 else [],
            qwen_history = parser_history(tmp_history)
            answer_result = AnswerResult()
            response=Generation.call(model=self.model_name, prompt=prompt, stream=False, history=qwen_history, max_length=self.max_token)
            if response.status_code==HTTPStatus.OK:
                res = response.output.text
                history += [[prompt, res]]    
                answer_result.history = history
                answer_result.llm_output = {"answer": res}
            else:
                print('Code: %d, status: %s, message: %s' % (response.status_code, response.code, response.message))
                raise Exception(response.message)
            yield answer_result