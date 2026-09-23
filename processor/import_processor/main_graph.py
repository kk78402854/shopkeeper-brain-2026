"""
    @Author:MaYuhang
    @Time:2026/9/23
    @DESC:
"""
import json
from langgraph.graph import StateGraph, END, START

from common.logging.logger import logger
from dotenv import load_dotenv
from processor.import_processor.state import ImportGraphState, create_default_state
from processor.import_processor.nodes.node_entry import node_entry
from processor.import_processor.nodes.node_pdf_to_md import node_pdf_to_md
from processor.import_processor.nodes.node_md_img import node_md_img
from processor.import_processor.nodes.node_document_split import node_document_split
from processor.import_processor.nodes.node_item_name_recognition import node_item_name_recognition
from processor.import_processor.nodes.node_bge_embedding import node_bge_embedding
from processor.import_processor.nodes.node_import_milvus import node_import_milvus

load_dotenv()

# ===================== 1. 初始化LangGraph状态图 =====================
workflow=StateGraph(ImportGraphState)

# ===================== 2. 注册所有业务节点 =====================

"""
                →not pdf or md：end
start→node_entry→pdf:node_pdf_to_md
                              ↓      
                →md: node_md_img→node_document_split→node_item_name_recognition→node_bge_embedding→node_import_milvus→end
1.node_entry：任务分发                
2.node_pdf_to_md：PDF结构化解析
3.node_md_img：多模态图片理解
4.node_document_split：智能文档切片
5.node_item_name_recognition：主体识别与表情提取
6.node_bge_embedding：混合向量化
7.node_import_milvus：数据持久化  导入 milvus
"""

workflow.add_node("node_entry",node_entry)
workflow.add_node("node_pdf_to_md",node_pdf_to_md)
workflow.add_node("node_md_img",node_md_img)
workflow.add_node("node_document_split",node_document_split)
workflow.add_node("node_item_name_recognition", node_item_name_recognition)
workflow.add_node("node_bge_embedding", node_bge_embedding)
workflow.add_node("node_import_milvus", node_import_milvus)

# ===================== 3. 设置工作流入口节点 =====================
workflow.set_entry_point("node_entry")

# ===================== 4. 定义条件路由函数（入口节点后的分支逻辑） =====================
#路由函数：判断pdf或md决定跳转
def route_after_entry(state:ImportGraphState):
    if state.get("is_md_read_enabled"): # 判断是否启用 Markdown 读取
        return "node_md_img"
    elif state.get("is_pdf_read_enabled"):# 否则判断是否启用 PDF 读取
        return "node_pdf_to_md"
    else:
        return END
#唯一的条件边
workflow.add_conditional_edges(
    "node_entry",
    route_after_entry,
    {
        "node_md_img":"node_md_img",
        "node_pdf_to_md":"node_pdf_to_md",
        END:END
    }#path_map 就是告诉 LangGraph"这条条件边可能通向哪几个节点"的清单——不给它，运行时能跑，但编译/画图时它只能看到一个断掉的 ?，ASCII 图就残缺了
)
# ===================== 5. 注册静态顺序边（分支合并后的统一流程） =====================
workflow.add_edge("node_pdf_to_md", "node_md_img")
workflow.add_edge("node_md_img", "node_document_split")
workflow.add_edge("node_document_split", "node_item_name_recognition")
workflow.add_edge("node_item_name_recognition", "node_bge_embedding")
workflow.add_edge("node_bge_embedding", "node_import_milvus")
workflow.add_edge("node_import_milvus", END)

# ===================== 6. 编译工作流为可执行对象 =====================
kb_import_app = workflow.compile()

#***************************************************************************************************

if __name__ == "__main__":
    logger.info("===== 开始测试 =====")

    initial_state = create_default_state(local_file_path="万用表RS-12的使用.pdf")
    final_state = None

    # 只输出更最终的状态值（字典形式），不包含节点名称、执行日志、元数据等额外信息
    # for event in kb_import_app.stream(initial_state):
    #     for key, value in event.items():
    #         logger.info(f"节点: {key}")
    #         final_state = value

    kb_import_app.invoke(initial_state)
    # 格式化输出最终状态
    logger.info(f"最终状态: {json.dumps(final_state, indent=4, ensure_ascii=False)}")

    logger.info("图结构:")
    # uv add grandalf
    kb_import_app.get_graph().print_ascii()

    logger.info("===== 测试结束 =====")