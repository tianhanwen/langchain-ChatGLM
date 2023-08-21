本服务可以在阿里云计算巢上一键拉起，如果要部署，请参考以下文档

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
   # 正在向langchain里提交代码支持混合检索, 暂时需要把安装的的site-packages/langchain/vectorstores/tair.py，替换为本项目depends/tair.py
   configs/model_config.py embedding_model_dict 
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