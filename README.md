本服务支持通过阿里云计算巢一键拉起：
https://computenest.console.aliyun.com/user/cn-hangzhou/serviceInstanceCreate?ServiceId=service-bad9920e33014ecf8a86
最好根据计算巢一键拉起，因为混合检索功能模块，langchain的代码有修改，正在走合入流程，当然，如果您不开启混合检索，可以忽略该提示。

1. 前期准备
   
   - 操作系统ubuntu: 22.04
   - 机器规格: 2核8G
   - python环境: python3.10
   - Tair实例: https://kvstorenext.console.aliyun.com/Tair/instance/cn-hangzhou
     - 存储介质：内存
     - 版本兼容性：Redis 6.0
     - 实例类型：高可用
     - 架构类型：不启用集群
   - 申请APIKEY并申请体验通义千问：https://help.aliyun.com/zh/dashscope/developer-reference/api-details
2. 安装依赖与参数设置
   
  ```
   pip3 install -r requirements.txt
   configs/model_config.py embedding_model_dict 替换为本地目录 
   先需要执行 git lfs clone https://huggingface.co/GanymedeNil/text2vec-large-chinese
   embedding_model_dict = {
    "text2vec": "/embedding/text2vec-large-chinese",
   }
   env文件
   export TAIR_URL=redis://default:替换为你的Tair实例账号@r-xxx.redis.rds.aliyuncs.com:6379
   export DASHSCOPE_API_KEY=替换为你的APIKEY
  ```

3. 运行
```
    ./start.sh
```