"""
    @Author:MaYuhang
    @Time:2026/10/12
    @DESC:
"""
import time
import sys
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from common.logging.logger import node_log, logger, step_log
from processor.query_processor.state import QueryGraphState
from utils.clients.mongo_history_utils import save_chat_message, get_recent_messages
from utils.lm.lm_utils import get_llm_client
from utils.load_prompt import load_prompt
from utils.task_utils import add_running_task, add_done_task

@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state):
    """
    步骤1：校验并获取原始查询（original_query）和会话 ID（session_id）。
    """
    #从状态中获取原始问题
    original_query = state.get("original_query")
    #从状态中获取会话id
    session_id = state.get("session_id")
    if (not original_query) or (not session_id):
        logger.error(f"session_id或者original_query为空,业务无法继续,提前终止!")
        raise ValueError(f"session_id或者original_query为空,业务无法继续,提前终止!")

    return original_query, session_id

@step_log("step_2_get_history_by_session_id")
def step_2_get_history_by_session_id(session_id):
    """
    步骤2：根据会话 ID 获取历史聊天记录。
        返回的记录结构说明：
            - _id:            记录的唯一标识
            - session_id:     所属会话 ID
            - role:           消息角色（如 user / assistant）
            - text:           消息文本内容
            - rewritten_query: 重写后的查询语句
            - item_names:     消息涉及的商品名称列表（用于过滤有效记录）
            - image_urls:     消息附带的图片 URL 列表
            - ts:             消息时间戳

        Returns:
            list[dict]: 过滤后的历史聊天记录列表，仅包含 item_names 非空的记录。
    """
    #根据 sesion_id 查询最近的聊天记录
    history_list = get_recent_messages(session_id)
    logger.debug(f"根据:{session_id}查询到的聊天记录:{len(history_list)}")

    #过滤出包含 item_names 的有效记录（即与商品相关的对话）
    historu_list = [item for item in history_list if item.get("item_names")]
    logger.debug(f"根据:{session_id}过滤后的有效聊天记录:{len(history_list)}")

    # 返回过滤后的历史记录
    return history_list

@step_log("step_3_call_llm_rewritten_and_extract_itemnames")
def step_3_call_llm_rewritten_and_extract_itemnames(history_list, original_query):
    """
    步骤3：调用大语言模型对原始查询进行重写，并提取关联的商品名称（item_names）。
        该函数接收历史聊天记录和当前用户的原始问题，通过构建包含历史上下文的提示词，
    调用 LLM 生成重写后的查询（rewritten_query）以及相关的商品名称列表（item_names）。

    面试题：如何保证大语言模型返回的是 JSON？
        1. 提示词中约定返回 JSON 格式。
        2. 提示词中提供 JSON 示例。
        3. 在创建模型时指定 json_mode，例如：
           model_kwargs["response_format"] = {"type": "json_object"}
        4. 在输出时使用 JsonOutputParser 解析器。

    单条记录示例::

        {
            "_id": "65f0c3a1e4b0a123456789ab",
            "session_id": "sess_abc123",
            "role": "user",
            "text": "有没有适合送女友的礼物",
            "rewritten_query": "适合送女友的礼物推荐",
            "item_names": ["香水", "口红"],
            "image_urls": [],
            "ts": 1697000000
        }

    Args:
        history_list (list[dict]): 过滤后的历史聊天记录（仅含 item_names 非空的记录）。
        original_query (str): 用户当前输入的原始问题。

    Returns:
        dict: 包含以下键：
            - rewritten_query (str): 重写后的查询；缺失时回退为 original_query。
            - item_names (list[str]): 关联商品名称列表；缺失时为空列表。
    """
    #获取大模型语言对象，开启 JSON 模式以确保输出 JSON 格式
    llm_model = get_llm_client(json_mode=True)
    #准备提示词参数：讲历史记录拼接成文本，供提示词使用
    history_text: str = None
    if history_list:
        # 定义一个新的列表，用于存储拼接的历史记录
        history_text_list: list[str] = []
        # 对 history_list 进行遍历，得到从 MongoDB 中查询出来的一个个文档（字典）
        for index, item in enumerate(history_list, start=1):
            # 判断当前遍历出来的这条历史记录角色是 user 还是 assistant
            if item.get("role") == "user":
                # 用户提问：包含原始问题、重写后的问题以及关联的 item_name
                history_text_list.append(
                    f"编号:{index},用户提问,原始问题:{item.get('text')},"
                    f"重写后的问题:{item.get('rewritten_query')}，"
                    f"关联item_name：{','.join(item.get('item_names'))}"
                )
            else:
                # 助手回答：包含重写问题、回答结果的前50个字符以及关联的 item_name
                history_text_list.append(
                    f"编号:{index},助手回答,重写问题:{item.get('rewritten_query')},"
                    f"回答结果:{item.get('text')[:50]},关联item_name:{','.join(item.get('item_names'))}"
                )
        # 将列表中的各行拼接成完整的字符串，以换行符分隔
        history_text = "\n".join(history_text_list)
    else:
        # 无历史记录时的提示文本
        history_text = "当前会话id没有对应的历史记录"
    # 拼接提示词：从配置中加载名为 "rewritten_query_and_itemnames" 的提示词模板，
    # 并传入当前查询和历史上下文
    prompt_text = load_prompt(
        "rewritten_query_and_itemnames",
        query=original_query,
        history_text=history_text
    )
    # 封装消息对象：将提示词包装为 HumanMessage
    message = HumanMessage(content=prompt_text)
    # 封装调用链：将 LLM 模型与 JSON 输出解析器组合成链
    chains = llm_model | JsonOutputParser()
    # 调用链并获取结果：传入消息列表，返回解析后的 JSON 字典
    json_dict = chains.invoke([message])
    # 对结果进行校验：确保必要字段存在，若缺失则填充默认值
    if "item_names" not in json_dict:
        json_dict["item_names"] = []
    if "rewritten_query" not in json_dict:
        json_dict["rewritten_query"] = original_query
    # 记录返回结果（此处使用 error 级别，可能是为了便于在日志中查看，实际可考虑改为 debug）
    logger.error(json_dict)
    # 返回包含重写查询和商品名称的字典
    return json_dict

@node_log("node_item_name_confirm")
def node_item_name_confirm(state:QueryGraphState):
    """
    节点功能：确认用户问题中的核心商品名称。
    """
    # TODO 1.添加节点到运行时列表(从 state 里取会话 ID;获取当前正在执行的函数名;从 state 里取是否流式标志)
    add_running_task(state.get("session_id"), sys._getframe().f_code.co_name,state.get("is_stream"))
    # TODO 2.从状态中获取数据并进行校验
    original_query,session_id = step_1_validate_and_get_data(state)
    # TODO 3.根据 session_id 获取当前会话的历史记录
    history_list: list[dict] = step_2_get_history_by_session_id(session_id)
    # TODO 4.根据历史记录，调用大语言模型进行意图识别以及问题重写
    llm_dict: dict[str,Any] = step_3_call_llm_rewritten_and_extract_itemnames(history_list,original_query)
    # TODO 5.对查询出来的主题进行向量化操作--先嵌入到 milvus 中 kb_item_names表，查询数据
    # TODO 6.根据向量数据库查询的结果 讲数据放到确认、可选列表中
    # TODO 7.更新状态
    # TODO 8.讲用户问题保存到 mongodb 数据库
    # TODO 9.添加节点到已完成列表
    add_done_task(state.get("session_id"), sys._getframe().f_code.co_name, state.get("is_stream"))
    return state

if __name__ == "__main__":
    mock_state = {
        "session_id": "test_session_001",
        "original_query": "HAK 180 烫金机怎么用？",
        "is_stream": False,
    }
    result_state = node_item_name_confirm(mock_state)
    print(result_state)