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

from common.config.milvus_config import milvus_config
from common.logging.logger import node_log, logger, step_log
from processor.query_processor.state import QueryGraphState
from utils.clients.milvus_utils import create_hybrid_search_requests, get_milvus_client, hybrid_search
from utils.clients.mongo_history_utils import save_chat_message, get_recent_messages
from utils.lm.embedding_utils import generate_embeddings
from utils.lm.lm_utils import get_llm_client
from utils.load_prompt import load_prompt
from utils.task_utils import add_running_task, add_done_task

# 主体名称确认阈值：高于该分数 → 直接确认 [0.75]
ITEM_NAME_CONFIRM_THRESHOLD = 0.70
# 主体名称候选阈值：介于两者之间 → 让用户选择
ITEM_NAME_CANDIDATE_THRESHOLD = 0.60
# 给用户选择时，最多展示几个候选
ITEM_NAME_OPTIONS_TOPK = 2
ITEM_NAME_CONFIRMED_TOPK = 1

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

@step_log("step_4_select_item_names_milvus")
def step_4_select_item_names_milvus(item_names) -> dict[str, list[dict]]:
    """
        对 LLM 识别出的主体名称进行向量化，并在 Milvus 的商品名称集合中进行混合检索。
    功能说明：
        1. 将 LLM 抽取出的主体名称列表批量生成 dense/sparse 向量；
        2. 对每个主体名称分别构造稠密 + 稀疏混合检索请求；
        3. 到 Milvus 的 item_name_collection 中检索相似商品名称；
        4. 返回“原始主体名称 -> 候选商品名称及相似度分数列表”的映射。
    参数:
        item_names: list[str]
            LLM 从用户问题中识别出的主体名称列表，例如：
            ["商品A", "商品B"]
    返回值:
        dict[str, list[dict]]
            每个原始主体名称对应一个候选结果列表，结构示例：
            {
                "商品A": [
                    {"item_name": "商品A", "score": 0.83},
                    {"item_name": "商品a", "score": 0.73},
                    {"item_name": "商品ab", "score": 0.56}
                ],
                "商品B": [
                    {"item_name": "商品B", "score": 0.83},
                    {"item_name": "商品ab", "score": 0.73},
                    {"item_name": "商品ababa", "score": 0.56}
                ]
            }
    注意:
        - score 来自 Milvus 返回的 distance 字段；
        - 当前 hybrid_search 使用 norm_score=True，因此 score 可理解为归一化后的融合相似度分数，
          通常分数越高表示越相似；
        - 后续 step_5 会基于 ITEM_NAME_CONFIRM_THRESHOLD 和 ITEM_NAME_CANDIDATE_THRESHOLD
          判断哪些候选是“确认主体”，哪些是“可选主体”。
    :param param:
    :return:
    """
    # 用于接收最终返回结果：原始主体名称 -> 候选商品名称及分数列表。
    milvus_search_result_dict: dict[str, list[dict]] = {}

    # 批量生成稠密和稀疏向量，dense/sparse 顺序与 item_names 一致。
    result = generate_embeddings(item_names)

    # 遍历每一个 LLM 识别出的主体名称。
    for index, llm_item_name in enumerate(item_names):
        # 根据当前下标获取该主体的稠密向量。
        dense_vector = result.get("dense")[index]

        # 根据当前下标获取该主体的稀疏向量。
        sparse_vector = result.get("sparse")[index]

        # 封装同时包含稠密和稀疏向量的混合检索请求。
        reqs_list = create_hybrid_search_requests(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector
        )

        # 在商品名称集合中执行混合检索，并只返回 item_name 字段。
        response = hybrid_search(
            client=get_milvus_client(),
            collection_name=milvus_config.item_name_collection,
            reqs=reqs_list,
            ranker_weights=(0.4, 0.6),
            norm_score=True,
            output_fields=["item_name"]
        )

        # 取当前主体对应的检索结果列表。
        real_response = response[0]

        # 如果当前主体无检索结果，则记录警告并跳出所有查询。
        if not real_response:
            logger.warning(
                f"根据item_name:{llm_item_name}进行向量数据库的相似度搜索,值为空,"
                f"向量数据库为空,跳出所有查询!"
            )
            break

        # 用于存放当前主体的候选商品名称及相似度分数。
        item_name_score_list: list[dict] = []

        # 将 Milvus 返回结构转换为 {"item_name": xxx, "score": xxx}。
        for item in real_response:
            # 注意：entity 默认值原逻辑为 []，缺失时再 .get 有风险，后续可改为 {}。
            item_name_score_list.append({
                "item_name": item.get("entity", []).get("item_name"),
                "score": item.get("distance")
            })

        # 保存当前原始主体名称对应的候选结果列表。
        milvus_search_result_dict[llm_item_name] = item_name_score_list

    # 返回所有主体名称对应的 Milvus 检索结果。
    return milvus_search_result_dict

@step_log("step_5_select_confirmed_and_option_item_names")
def step_5_select_confirmed_and_option_item_names(milvus_dict):
    """
    根据 Milvus 混合检索结果，将候选商品名称划分为“确认主体”和“可选主体”。

    功能：
        1. 遍历每个 LLM 原始主体名称对应的候选商品列表；
        2. 分数 >= ITEM_NAME_CONFIRM_THRESHOLD 的候选进入 confirmed_list；
        3. ITEM_NAME_CANDIDATE_THRESHOLD <= 分数 < ITEM_NAME_CONFIRM_THRESHOLD
           的候选进入 option_list；
        4. 每个主体优先取确认结果；无确认结果时再取可选结果。

    参数:
        milvus_dict: dict[str, list[dict]]
            step_4 返回的检索结果，例如：
            {
                "商品A": [
                    {"item_name": "商品A", "score": 0.83},
                    {"item_name": "商品a", "score": 0.73},
                    {"item_name": "商品ab", "score": 0.56}
                ],
                "商品B": [
                    {"item_name": "商品B", "score": 0.83},
                    {"item_name": "商品ab", "score": 0.73}
                ]
            }

    返回:
        dict[str, list[str]]
            {
                "confirmed": ["商品A", "商品B"],
                "option": ["商品a", "商品ab"]
            }

    注意:
        confirmed 表示可直接确认的主体；
        option 表示分数处于候选区间、需要用户进一步明确的主体；
        当前逻辑中，若某主体存在 high_list，则不会再处理该主体的 middle_list。
    """

    confirmed_list: list[str] = []  # 确认的 item_name 列表。
    option_list: list[str] = []  # 可选的 item_name 列表。

    # 遍历每个 LLM 原始主体名称及其 Milvus 候选结果列表。
    for llm_item_name, item_name_score_list in milvus_dict.items():
        # 提取分数达到确认阈值的候选商品名称。
        high_list = [
            item.get("item_name")
            for item in item_name_score_list
            if item.get("score") >= ITEM_NAME_CONFIRM_THRESHOLD
        ]
        # 提取分数处于候选区间的候选商品名称。
        middle_list = [
            item.get("item_name")
            for item in item_name_score_list
            if ITEM_NAME_CANDIDATE_THRESHOLD <= item.get("score", 0.0) < ITEM_NAME_CONFIRM_THRESHOLD
        ]

        # 有可信主体时，取前 ITEM_NAME_CONFIRMED_TOPK 个加入确认列表。
        if high_list:
            confirmed_list.extend(high_list[:ITEM_NAME_CONFIRMED_TOPK])
            logger.info(f"{llm_item_name}已经获取到可信的主体:{high_list[:ITEM_NAME_CONFIRMED_TOPK]}")
            continue

        # 无可信主体但有可选主体时，取前 ITEM_NAME_OPTIONS_TOPK 个加入可选列表。
        if middle_list:
            option_list.extend(middle_list[:ITEM_NAME_OPTIONS_TOPK])
            logger.info(f"{llm_item_name}没有可信的主体,但是有可选的主体:{middle_list[:ITEM_NAME_OPTIONS_TOPK]}")

    # 返回确认主体列表和可选主体列表。
    return {
        "confirmed": confirmed_list,
        "option": option_list
    }

@step_log("step_6_change_state_property")
def step_6_change_state_property(state, rewritten_query, confirmed_option_dict):
    """
    根据 step_5 的确认/可选结果更新图状态 state。

    功能：
        1. 若存在确认主体列表，则写入 state["item_names"] 和 state["rewritten_query"]；
        2. 若不存在确认主体但存在可选主体，则写入 state["answer"] 提示用户明确；
        3. 若确认主体和可选主体都不存在，则写入 state["answer"] 提示重新提问。

    参数:
        state: QueryGraphState
            当前查询图状态。
        rewritten_query: str
            LLM 改写后的问题。
        confirmed_option_dict: dict[str, list[str]]
            step_5 返回结果，例如：
            {
                "confirmed": ["商品A", "商品B"],
                "option": ["商品a", "商品ab"]
            }

    返回:
        None
            本函数直接原地修改 state，不返回新状态。

    注意:
        三种场景互斥：确认列表优先；无确认时看可选；两者都无则提示重新提问。
    """

    # 获取确认主体列表，缺失时默认为空列表。
    confirmed_list = confirmed_option_dict.get("confirmed", [])
    # 获取可选主体列表，缺失时默认为空列表。
    option_list = confirmed_option_dict.get("option", [])

    # 场景1：存在确认主体，更新 item_names 和 rewritten_query。
    if confirmed_list:
        state["item_names"] = confirmed_list  # expr= item_name in [sp_a,sp_b,sp_c]
        state["rewritten_query"] = rewritten_query
        logger.info(f"已经有明确的item_name列表,已经更新state item_name以及rewritten_query属性!")
        return
    # 场景2：无确认主体但有可选主体，更新 answer 提示用户明确。
    if option_list:
        state['answer'] = f"没有可信的item_name,但是有可选的:{option_list},请您明确!"
        logger.info(f"没有明确的item_name列表,但是有可选的!已经更新answer属性!")
        return
    # 场景3：确认主体和可选主体都不存在，更新 answer 提示重新提问。
    state['answer'] = f"没有可信,也没有可选的实体列表,请明确以后再次问题!!"


@step_log("step_7_save_user_chat_message")
def step_7_save_user_chat_message(state):
    """
    将当前用户问题及相关状态保存到 MongoDB 聊天记录。

    功能：
        保存 session_id、用户原始问题、改写后问题、确认主体列表及图片列表。

    参数:
        state: QueryGraphState
            当前查询图状态，需包含 session_id、original_query、
            rewritten_query、item_names 等字段。

    返回:
        None

    注意:
        text 使用 original_query，前端展示的是用户原始问题；
        item_names 缺失时默认为空列表；
        当前不保存图片，image_urls 固定为空列表。
    """

    # 保存用户聊天消息到 MongoDB。
    save_chat_message(
        session_id=state.get("session_id"),
        role="user",
        text=state.get("original_query"),  # 前端显示的就是这个字段
        rewritten_query=state.get("rewritten_query"),
        item_names=state.get("item_names", []),
        image_urls=[]
    )

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
    confirmed_option_dict = {}
    if llm_dict.get("item_names"):
        # TODO 5. 对查询出来的主体进行向量化操作--先嵌入，到milvus中kb_item_names表，查询数据
        milvus_dict: dict[str, list[dict]] = step_4_select_item_names_milvus(llm_dict.get("item_names"))
        # TODO 6. 根据向量数据库查询的结果  将数据放到确认、可选列表中--->{"confirmed":["商品A","商品B"],"option":[xx,xxx,xxx]}
        confirmed_option_dict: dict[str, list[str]] = step_5_select_confirmed_and_option_item_names(milvus_dict)
    # TODO 7.更新状态
    step_6_change_state_property(state, llm_dict.get("rewritten_query"), confirmed_option_dict)
    # TODO 8.讲用户问题保存到 mongodb 数据库
    step_7_save_user_chat_message(state)
    # TODO 9.添加节点到已完成列表
    add_done_task(state.get("session_id"), sys._getframe().f_code.co_name, state.get("is_stream"))
    return state

if __name__ == "__main__":
    mock_state = {
        "session_id": "test_session_001",
        "original_query": "苹果手机怎么用？",
        "is_stream": False,
    }
    result_state = node_item_name_confirm(mock_state)
    print(result_state)