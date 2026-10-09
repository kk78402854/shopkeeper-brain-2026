"""
    @Author:MaYuhang
    @Time:2026/9/23
    @DESC:
"""
import sys
from pathlib import Path

from common.logging.logger import logger, node_log
from processor.import_processor.state import ImportGraphState, create_default_state
from utils.task_utils import add_running_task, add_done_task


@node_log("node_entry")
def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 入口节点 (node_entry)
    为什么叫这个名字: 作为图的 Entry Point，负责接收外部输入并决定流程走向。
    """
    # 模拟简单的路由逻辑，防止报错 (仅 node_entry 需要)
    # TODO . 节点添加到运行时列表
    add_running_task(state.get("task_id"),"node_entry")
    #TODO . 从状态获取localfilepath
    local_file_path=state.get("local_file_path")
    # TODO . localfilepath判空
    if not local_file_path:
        logger.warning("从状态中没有获取到local_file_path，导入流程终止")
        return state
    #TODO。 判断类型 更新状态
    if local_file_path.endswith(".md"):
        state["is_md_read_enabled"] = True
        state["is_pdf_read_enabled"] = False

        state["md_path"] = local_file_path
        state["pdf_path"] = None
    elif local_file_path.endswith(".pdf"):
        state["is_pdf_read_enabled"] = True
        state["is_md_read_enabled"] = False

        state["pdf_path"] = local_file_path
        state["md_path"] = None
    else:
        logger.warning("不支持的文件类型")
        return state

    #TODO.  更新filetitle
    state["file_title"] =Path(local_file_path).stem
    # TODO 6. 将节点添加到已完成列表
    add_done_task(state.get("task_id"),"node_entry")
    return state

if __name__ == '__main__':
    # 单元测试：覆盖不支持类型、MD、PDF三种场景
    logger.info("===== 开始node_entry节点单元测试 =====")

    # 测试1: 不支持的TXT文件
    test_state1 = create_default_state(
        task_id="test_task_001",
        local_file_path="联想海豚用户手册.md"
    )
    node_entry(test_state1)

    # 测试2: MD文件
    test_state2 = create_default_state(
        task_id="test_task_002",
        local_file_path="小米用户手册.md"
    )
    node_entry(test_state2)

    # 测试3: PDF文件
    test_state3 = create_default_state(
        task_id="test_task_003",
    )
    node_entry(test_state3)

    logger.info("===== 结束node_entry节点单元测试 =====")