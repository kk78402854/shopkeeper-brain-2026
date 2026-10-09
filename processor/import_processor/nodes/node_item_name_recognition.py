"""
    @Author:MaYuhang
    @Time:2026/9/23
    @DESC:
"""
import json
import os
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from pymilvus import DataType

from common.config.milvus_config import milvus_config
from common.logging.logger import logger, node_log, step_log
from processor.import_processor import state
from processor.import_processor.state import ImportGraphState
from utils.clients.milvus_utils import get_milvus_client
from utils.escape_milvus_string_utils import escape_milvus_string
from utils.lm.embedding_utils import generate_embeddings
from utils.lm.lm_utils import get_llm_client
from utils.load_prompt import load_prompt
from utils.task_utils import add_running_task, add_done_task

# 主体识别上下文切片数：取前 K 个切片用于 LLM 识别
ITEM_NAME_CONTEXT_CHUNK_K = 5
# 主体识别上下文总字符数上限：防止上下文过长导致大模型输入超限
ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS = 2000

"""
步骤 1: 获取输入并校验
从 State 中提取 chunks 和 file_title，并进行基础校验：
- 若 chunks 为空，则尝试从 md_path 对应的 JSON 备份文件中恢复；
- 若 file_title 为空，则用 md_path 的文件名（去后缀）作为默认标题。
"""
@step_log("step_1_validata_and_get_data")
def step_1_validata_and_get_data(state)->tuple[list[dict[str,Any]],str]:
    #从State中取出chunks(可能是已切好的结果，也可能是空)
    chunks: list[dict[str,Any]] = state.get("chunks")
    #从State中取出文件标题（可能为空）
    file_title: str = state.get('file_title')
    # 从State中获取md_path
    md_path = state.get("md_path")
    md_path_obj = Path(md_path)

    #如果chunks为空，尝试从md_path同目录下的JSON备份文件中恢复
    if not chunks:
        #取出md文件路径
        md_path = state.get('md_path')
        #封装为Path对象，便于后续判断文件、取父目录、取文件名
        md_path_obj:Path = Path(md_path)
        #先判断md文件是否存在
        if md_path_obj.is_file():
            #推算出对应的JSON备份路径：同目录，同文件名，后缀为.json
            json_path_obj:Path = md_path_obj.parent / f"{md_path_obj.stem}.json"
            #再判断JSON备份文件是否存在
            if json_path_obj.is_file():
                #从JSON备份文件读取chunks，并写回State
                chunks = json.loads(json_path_obj.read_text(encoding='utf-8'))
                state['chunks'] = chunks
                logger.warning("从备份文件中读取chunks")
            else:
                #md存在但JSON备份不存在：无法恢复，直接抛异常
                logger.error(f"chunks没有值,同时也没有读取到对应json备份数据,抛出异常!")
                raise ValueError(f"chunks没有值,同时也没有读取到对应json备份数据,抛出异常!")
        else:
            # md文件也不存在：无法恢复，直接抛异常
            logger.error(f"chunks没有值,同时也没有读取到对应md_path,抛出异常!")
            raise ValueError(f"chunks没有值,同时也没有读取到对应md_path,抛出异常!")
    #如果file_title为空，用md_path的文件名(去后缀）作为默认标题
    if not file_title:
        file_title = md_path_obj.stem or "default_title"
        state['file_title'] = file_title
        logger.warning(f"file_title不存在,给与默认值:{file_title}")
    # 返回校验/恢复后的 chunks 与 file_title
    return chunks, file_title

"""
# 步骤 2: 调用大模型识别 item_name
# 从切分后的 chunks 中取前若干个作为上下文，拼接提示词后调用 LLM，
# 让模型基于 file_title 和上下文推断出条目名称（item_name）；
# 若模型返回为空，则回退使用 file_title。
"""

@step_log("step_2_call_llm_return_item_name")
def step_2_call_llm_return_item_name(chunks, file_title):
    #创建模型对象(返回一个可调用的LLM客户端)
    llm_model = get_llm_client()
    #准备提示词模板需要的参数：上下文文本初始为空
    context = ""
    #从切分后的所有chunk中遍历出前 ITEM_NAME_CONTEXT_CHUNK_K 个，用于构造上下文
    for chunk in chunks[:ITEM_NAME_CONTEXT_CHUNK_K]:
        # 目前 chunk 中有的属性：{"file_title":"","parent_title":"","title":"","part":"","content":""}
        # 拼接格式：标题: parent_title , 内容: content \n
        #         标题: parent_title , 内容: content \n
        context += f"标题：{chunk.get('parent_title')},内容：{chunk.get('content')} \n"
    #如果拼接出来的上下文长度超过 ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS ,进行截取，避免提示词过长
    context = context[:ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS]

    #调用 load_prompt 加载提示词模板，并把file_title 和 context 填充进去，拼接成完整提示词
    prompt_text = load_prompt("item_name_recognition",file_title=file_title,context=context)

    #封装消息对象：把提示词包装成 HumanMessage，作为对话输入
    message = HumanMessage(
        content=prompt_text,
    )

    #封装调用链：LLM → 字符串输出解析器，最终得到纯文本结果
    chains = llm_model | StrOutputParser()

    #调用链，获取模型返回的 item_name
    item_name = chains.invoke([message])

    #对结果进行非空判断：若模型没有识别出 item_name，则回退使用 file_title
    if not item_name:
        item_name = file_title
        logger.warning(f"没有识别出item_name,使用file_title赋值:{item_name}")
    # 返回识别结果（或回退值）
    return item_name

"""
步骤 3: 将识别出来的item_name回调到chunk中
"""
@step_log("step_3_padding_item_name_to_chunks")
def step_3_padding_item_name_to_chunks(chunks, item_name):
    # 遍历所有 chunk，为每个 chunk 补充 item_name 字段
    for chunk in chunks:
        chunk["item_name"] = item_name

"""
步骤 4: 准备 item_name 集合
确保 Milvus 中用于存储 item_name 的 collection 已存在：
- 若已存在，直接复用；
- 若不存在，则创建 schema、索引并建集合。
该集合用于存放条目名称及其向量，便于后续按条目检索。
"""
@step_log("step_4_prepared_item_name_collection")
def step_4_prepared_item_name_collection():
    #获取操作 Milvus 数据库的客服端
    milvus_client = get_milvus_client()
    #判断集合是否存在：存在则直接复用，不存在才走创建流程
    has_collection = milvus_client.has_collection(collection_name=milvus_config.item_name_collection)
    if has_collection:
        logger.info(f"{milvus_config.item_name_collection}已经存在,可以直接使用!")
        return
    logger.info(f"{milvus_config.item_name_collection}不存在,进行集合的创建!")

    #创建 schema：开启主键自增 + 允许动态字段（后续可扩展额外属性）
    schema = milvus_client.create_schema(
        auto_id=True,
        enable_dynamic_field=True
    )
    # 通过 schema 维护表中的字段（field）：
    # - pk            主键，INT64，自增
    # - file_title    文件标题，VARCHAR(512)
    # - item_name     条目名称，VARCHAR(512)
    # - dense_vector  稠密向量，FLOAT_VECTOR，维度 1024
    # - sparse_vector 稀疏向量，SPARSE_FLOAT_VECTOR
    schema.add_field(field_name="pk",datatype=DataType.INT64,is_primary=True)
    schema.add_field(field_name="file_title",datatype=DataType.VARCHAR,max_length=512)
    schema.add_field(field_name="item_name",datatype=DataType.VARCHAR,max_length=512)
    schema.add_field(field_name="dense_vector",datatype=DataType.FLOAT_VECTOR,dim=1024)
    schema.add_field(field_name="sparse_vector",datatype=DataType.SPARSE_FLOAT_VECTOR)
    #创建索引参数：索引用于提升查询性能
    index_params = milvus_client.prepare_index_params()
    #添加索引————稠密向量字段
    #使用 HNSW 图索引，COSINE 相似度；
    # M 为每个节点的最大出边数，efConstruction 为建索引时的搜索广度，数值越大越准但越慢。
    index_params.add_index(
        field_name="dense_vector",
        index_name="dense_vector_index",
        index_type="HNSW",
        metric_type="COSINE",
        params={
            "M":64,
            "efConstruction":100
        }
    )
    #添加索引————稀疏向量字段；
    #使用倒排索引 SPARSE_INVERTED_INDEX,IP 内积作为打分方式；
    #inverted_index_algo=DAAT_MAXSCORE 用于跳过大量低分文档，只计算最高分，加速检索。
    index_params.add_index(
        field_name="sparse_vector",#字段名称
        index_name="sparse_vector_index",#索引名称
        index_type="SPARSE_INVERTED_INDEX",#索引类型
        metric_type="IP",#算分方式
        params={"inverted_index_algo":"DAAT_MAXSCORE"}#跳过低分，只算高分
    )

    #创建collection：把schema 与索引参数一起传入，一次完成建表 + 建索引
    milvus_client.create_collection(
        collection_name=milvus_config.item_name_collection,
        schema=schema,
        index_params=index_params
    )
    logger.success(f"集合{milvus_config.item_name_collection}创建成功")

"""
步骤 5: 将识别出的 item_name 写入 Milvus
流程：
1. 先按 file_title 删除旧数据（保证同一文件只保留最新条目名）；
2. 对 item_name 做向量化（稠密 + 稀疏）；
3. 将 file_title、item_name 及向量写入 item_name 集合。
"""
@step_log("step_5_insert_item_name_data")
def step_5_insert_item_name_data(item_name, file_title):
    #获取Milvus 客户端
    milvus_client = get_milvus_client()
    #先删除该 file_title 下的旧数据，避免重复插入（实现“覆盖式”更新）
    milvus_client.delete(
        collection_name=milvus_config.item_name_collection,
        # 注意：  1.  等值比较用==     2.  file_tile 用引号括起来
        filter=f"file_title=='{file_title}'"
    )
    #对 item_name 做向量化处理：同时得到稠密向量和稀疏向量
    result = generate_embeddings([item_name])
    dense_vector = result.get("dense")[0]
    sparse_vector = result.get("sparse")[0]

    #将元数据与向量一并插入集合中
    milvus_client.insert(
        collection_name=milvus_config.item_name_collection,
        data=[
            {
                "file_title": file_title,
                "item_name": item_name,
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector
            }
        ]
    )
    logger.success(f"向{milvus_config.item_name_collection}插入{item_name}成功")
@node_log("node_item_name_recognition")
def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 主体识别 (node_item_name_recognition)
    """
    #TODO 1.添加节点到运行时列表
    add_running_task(state.get("task_id"),"node_item_name_recognition")
    #TODO 2.从状态中获取数据并进行校验
    chunks,file_title = step_1_validata_and_get_data(state)
    #TODO 3.调用模型识别 item_name
    item_name:str = step_2_call_llm_return_item_name(chunks,file_title)
    #TODO 4.向 chunk 中回填 item_name属性
    step_3_padding_item_name_to_chunks(chunks,item_name)
    #TODO 5.在 milvus 中创建存储 item_name 的 collection
    step_4_prepared_item_name_collection()
    #TODO 6.将数据写到milvus 的 collection中
    step_5_insert_item_name_data(item_name,file_title)
    #TODO 7.更新状态
    state["item_name"] = item_name
    state["chunks"] = chunks
    #TODO 8.添加节点到已完成列表
    add_done_task(state.get("task_id"),"node_item_name_recognition")
    return state

# ===================== 本地测试方法（直接运行调试，无需启动LangGraph） =====================
def test_node_item_name_recognition():
    """
    商品名称识别节点本地测试方法
    功能：模拟LangGraph流程输入，独立测试node_item_name_recognition节点全链路逻辑
    适用场景：本地开发、调试、单节点功能验证，无需启动整个LangGraph流程
    测试前准备：
        1. 确保项目环境变量配置完成（MILVUS_URL/ITEM_NAME_COLLECTION等）
        2. 确保大模型、Milvus、BGE-M3服务均可正常访问
        3. 确保prompt模板（item_name_recognition/product_recognition_system）已存在
    使用方法：
        直接运行该函数：if __name__ == "__main__": test_node_item_name_recognition()
    """
    logger.info("=== 开始执行商品名称识别节点本地测试 ===")
    try:
        # 1. 构造模拟的ImportGraphState状态（模拟上游节点产出数据）
        mock_state = ImportGraphState({
            "task_id": "test_task_123456",  # 测试任务ID
            "file_title": "华为Mate60 Pro手机使用说明书",  # 模拟文件标题
            "file_name": "华为Mate60Pro说明书.pdf",  # 模拟原始文件名（兜底用）
            "md_path":r"D:\dev\workspace\shopkeeper-brain-demo\output\hak180产品安全手册\hak180产品安全手册_new.md",
            # 模拟文本切片列表（上游切片节点产出，含title/content字段）
            "chunks": [
                {
                    "title": "产品简介",
                    "parent_title":"产品简介",
                    "content": "华为Mate60 Pro是华为公司2023年发布的旗舰智能手机，搭载麒麟9000S芯片，支持卫星通话功能，屏幕尺寸6.82英寸，分辨率2700×1224。"
                },
                {
                    "title": "拍照功能",
                    "parent_title": "拍照功能",
                    "content": "华为Mate60 Pro后置5000万像素超光变摄像头+1200万像素超广角摄像头+4800万像素长焦摄像头，支持5倍光学变焦，100倍数字变焦。"
                },
                {
                    "title": "电池参数",
                    "parent_title": "电池参数",
                    "content": "电池容量5000mAh，支持88W有线超级快充，50W无线超级快充，反向无线充电功能。"
                }
            ]
        })

        # 2. 调用商品名称识别核心节点
        result_state = node_item_name_recognition(mock_state)

        # 3. 打印测试结果（调试用）
        logger.info("=== 商品名称识别节点本地测试完成 ===")
        logger.info(f"测试任务ID：{result_state.get('task_id')}")
        logger.info(f"最终识别商品名称：{result_state.get('item_name')}")
        logger.info(f"切片数量：{len(result_state.get('chunks', []))}")
        logger.info(f"第一个切片商品名称：{result_state.get('chunks', [{}])[0].get('item_name')}")

        # 4. 验证 Milvus 存储（可选）
        milvus_client = get_milvus_client()
        collection_name = milvus_config.item_name_collection
        if milvus_client and collection_name:
            # 让刚插入的数据落盘，确保可查
            milvus_client.flush(collection_name=collection_name)

            # 检索测试结果
            item_name = result_state.get("item_name")
            safe_name = escape_milvus_string(item_name)
            res = milvus_client.query(
                collection_name=collection_name,
                filter=f'item_name=="{safe_name}"',
                output_fields=["file_title", "item_name"],
                consistency_level="Strong"
            )
            logger.info(f"Milvus中检索到的数据：{res}")

    except Exception as e:
        logger.error(f"商品名称识别节点本地测试失败，原因：{str(e)}", exc_info=True)


# 测试方法运行入口：直接执行该文件即可触发测试
if __name__ == "__main__":
    # 执行本地测试
    test_node_item_name_recognition()