"""
    @Author:MaYuhang
    @Time:2026/9/23
    @DESC:
"""
from pymilvus import DataType

from common.config.milvus_config import milvus_config
from common.logging.logger import logger, node_log, step_log
from processor.import_processor.state import ImportGraphState
from utils.clients.milvus_utils import get_milvus_client
from utils.task_utils import add_running_task, add_done_task

@step_log("step_1_validate_get_data")
def step_1_validate_get_data(state):
    """
    步骤 1: 校验并获取 embeddings_content
    从 state 中取 embeddings_content，若为空则记录日志并抛异常终止。
    """
    #从状态中获取 embeddings_content
    embeddings_content = state.get("embeddings_content")
    #判断是否存在
    if not embeddings_content:
        logger.error(f"embeddings_content没有值,业务无法继续,提前终止!")
        raise ValueError(f"embeddings_content没有值,业务无法继续,提前终止!")
    return embeddings_content

@step_log("step_2_prepared_item_name_collection")
def step_2_prepared_item_name_collection():
    """
    步骤 2: 准备 chunk 集合
    确保 Milvus 中用于存储 chunk 的集合存在：
    已存在则直接复用，不存在则创建 schema、索引并建集合。
    """
    #获取操作 milvus 的客户端
    milvus_client = get_milvus_client()
    #判断集合是否存在：存在直接用，不存在继续执行创建操作
    has_collection = milvus_client.has_collection(collection_name=milvus_config.chunks_collection)
    if has_collection:
        logger.info(f"{milvus_config.chunks_collection}已经存在,可以直接使用!")
        return
    logger.info(f"{milvus_config.chunks_collection}不存在,进行集合的创建!")

    #创建 schema：开启主键自增和动态字段
    schema = milvus_client.create_schema(
        auto_id=True,
        enable_dynamic_field=True,
    )
    # 添加字段
    schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="part", datatype=DataType.INT8)
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
    # 创建索引参数
    index_params = milvus_client.prepare_index_params()
    # 添加索引——稠密向量字段：HNSW 图索引，COSINE 相似度
    index_params.add_index(
        field_name="dense_vector",
        index_name="dense_vector_index",
        index_type="HNSW",
        params={
            "M": 64,  # 相邻最大的节点数量
            "efConstruction": 100  # 候选的节点数量
        },
        metric_type="COSINE"
    )
    # 添加索引——稀疏向量字段：倒排索引，IP 内积，DAAT_MAXSCORE 加速
    index_params.add_index(
        field_name="sparse_vector",
        index_name="sparse_vector_index",
        index_type="SPARSE_INVERTED_INDEX",
        params={"inverted_index_algo": "DAAT_MAXSCORE"},  # 跳过低分! 计算高分数据集!!
        metric_type="IP"
    )
    # 创建集合
    milvus_client.create_collection(
        collection_name=milvus_config.chunks_collection,
        schema=schema,
        index_params=index_params
    )
    logger.success(f"集合{milvus_config.chunks_collection}创建成功")

@step_log("step_3_insert_item_name_data")
def step_3_insert_item_name_data(embeddings_content):
    """
    步骤 3: 向 Milvus 插入 chunk 数据
    业务逻辑：同一份文档重新入库时，先删掉它的旧数据，再插入新数据，
    保证一个文件在库里只保留最新的一份，避免重复。
    """
    #获取 milvus 客户端
    milvus_client = get_milvus_client()
    # embeddings_content 是同一份文档切出来的所有 chunk，结构类似：
    # [
    #   {"file_title": "hak180产品安全手册", "title": "## 警告", "content": "...", ...},
    #   {"file_title": "hak180产品安全手册", "title": "## 电源线", "content": "...", ...},
    #   ...
    # ]
    # 同一批数据里 file_title 都一样，所以取第一条的即可代表整批。
    file_title = embeddings_content[0].get("file_title")

    # 按 file_title 删除 Milvus 中该文件的旧数据：
    # 如果这份文档之前已经入库过，先把旧 chunk 全部清掉，
    # 避免新旧数据混在一起，检索时出现重复或过时内容。
    milvus_client.delete(
        collection_name=milvus_config.chunks_collection,
        filter=f"file_title=='{file_title}'"
    )

    #把本次切好并已完成向量的 chunk 批量插入
    milvus_client.insert(
        collection_name=milvus_config.chunks_collection,
        data=embeddings_content
    )
    logger.success(f"向{milvus_config.chunks_collection}插入数据成功")

@node_log("node_import_milvus")
def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 导入向量库 (node_import_milvus)
    """
    # TODO 1. 添加节点到运行时列表
    add_running_task(state.get("task_id"),"node_import_milvus")
    # TODO 2. 从状态中获取数据并进行校验
    embeddings_content = step_1_validate_get_data(state)
    # TODO 3. 在milvus中创建集合
    step_2_prepared_item_name_collection()
    # TODO 4. 向milvus集合中插入数据
    step_3_insert_item_name_data(embeddings_content)
    # TODO 5. 添加节点到已完成列表
    add_done_task(state.get("task_id"),"node_import_milvus")
    return state

if __name__ == '__main__':
    # --- 单元测试 ---
    # 目的：验证 Milvus 导入节点的完整流程，包括连接、创建集合、清理旧数据和插入新数据。
    import os
    from dotenv import load_dotenv

    # 加载环境变量 (自动寻找项目根目录的 .env)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    load_dotenv(os.path.join(project_root, ".env"))

    # 构造测试数据
    dim = 1024
    test_state = {
        "task_id": "test_milvus_task",
        "item_name":"测试项目_Milvus",
        "file_title": "test.pdf",
        "embeddings_content": [
            {
                "content": "Milvus 测试文本 1",
                "title": "测试标题",
                "item_name": "测试项目_Milvus",  # 必须有 item_name，用于幂等清理
                "parent_title":"test.pdf",
                "part":1,
                "file_title": "test.pdf",
                "dense_vector": [0.1] * dim,  # 模拟 Dense Vector
                "sparse_vector": {1: 0.5, 10: 0.8}  # 模拟 Sparse Vector
            }
,
            {
                "content": "Milvus 测试文本 2",
                "title": "测试标题2",
                "item_name": "测试项目_Milvus",  # 必须有 item_name，用于幂等清理
                "parent_title": "test.pdf2",
                "part": 1,
                "file_title": "test.pdf",
                "dense_vector": [0.2] * dim,  # 模拟 Dense Vector
                "sparse_vector": {1: 0.5, 10: 0.8}  # 模拟 Sparse Vector
            }
        ]
    }

    print("正在执行 Milvus 导入节点测试...")
    try:
        # 执行节点函数
        result_state = node_import_milvus(test_state)
    except Exception as e:
        print(f"❌ 测试失败: {e}")