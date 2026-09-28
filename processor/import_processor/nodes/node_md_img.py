"""
    @Author:MaYuhang
    @Time:2026/9/23
    @DESC:
"""
import base64
import os
import re
import sys
from mimetypes import guess_type
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from common.config.lm_config import lm_config
from common.logging.logger import logger, node_log, step_log
from processor.import_processor.state import ImportGraphState
from utils.lm.lm_utils import get_llm_client
from utils.load_prompt import load_prompt
from utils.rate_limit_utils import apply_api_rate_limit
from utils.task_utils import add_running_task, add_done_task

#md_path-->md_path_obj-->md_content-->md_images_dir_obj
@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state)->tuple[str,Path,Path]:
    #从状态中获取md_path
    md_path=state.get("md_path")
    if not md_path:
        #如果状态中没有找到md_path，抛出异常
        logger.error("md_path变量为空!")
        raise ValueError("md_path变量为空!")

    #将md_path封装为一个Path对象
    md_path_obj=Path(md_path)
    #判断对应的md_path文件是否存在
    if not md_path_obj.is_file():
        #不存在则抛出异常
        logger.error(f"{md_path_obj}不存在或者不是一个文件！")
        raise FileNotFoundError(f"{md_path_obj}不存在或者不是一个文件！")

    #从md_path中读取内容->md_content
    md_content=md_path_obj.read_text(encoding="utf-8")
    #将md_content加入到状态（可选）
    state["md_content"]=md_content
    #获取存在md图片的路径

    md_images_dir_obj=md_path_obj.parent/"images"

    #返回
    return md_content,md_path_obj,md_images_dir_obj

#MinIO支持的图片格式合集（小写后缀，统一匹配标注）
IMAGE_EXTENSIONS={".jpg",".jpeg",".png",".gif",".bmp",".webp"}
@step_log("step_2_scan_images")
def is_supported_image(filename:str)->bool:
    """
    判断文件是否为 MinIO支持的图片格式（后缀不区分大小写）
    :param filename: 文件名（含后缀）
    :return: 支持返回True，否则False
    """
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS

#md_content,md_images_dir_obj-->list[(图片名,图片路径,(上文,下文))]
@step_log("step_2_scan_images")
def step_2_scan_images(md_content,md_images_dir_obj)->list[tuple[str, str, tuple[str, str]]]:
    #用于存储返回的数据列表
    image_info_list=[]
    #对图片目录下的所有图片进行遍历
    for image_file_obj in md_images_dir_obj.iterdir():
        #获取图片名
        image_name=image_file_obj.name
        #获取图片路径
        image_path=str(image_file_obj)
        #过滤非图片格式
        if not is_supported_image(image_name):
            logger.warning(f"跳过非图片文件:{image_name}")
            continue
        #使用正则表达式到md_content里找到对应的图片位置
        rep=re.compile(r"\!\[.*?\]\(.*?"+re.escape(image_name)+r".*?\)")
        search_match=rep.search(md_content)
        if not search_match:
            #为空，真没有匹配到
            logger.warning(f"{image_name}没有被md_content引用引用,跳过，直接下一次!!")
            continue
        #获取匹配内容的起始下标位置
        start=search_match.start()
        #获取匹配内容的结束下标位置
        end=search_match.end()
        #获取图片上文信息       起始位置向前截取100个字符
        pre_content=md_content[max(0,start-100):start]
        #获取图片下文位置       结束位置向后截取100个字符
        post_content=md_content[end:min(end+100,len(md_content))]
        logger.debug(f"{image_name}在md_content被引用,引用的位置:{start}:{end},截取的上文::{pre_content} , 下文:{post_content}")
        #封装元组并放到列表中
        image_info_list.append((image_name,image_path,(pre_content,post_content)))

    logger.info(f"所有图片的上下文信息已经识别完毕,数量为:{len(image_info_list)}")
    # 返回
    return image_info_list

@step_log("step_3_scan_images")
def step_3_image_summary(image_info_list,root_folder)->dict[str,str]:
    """

    获取图片对应的摘要
    :param image_info_list:      图片信息列表
    :param root_folder:          md文件名称
    :return:     dict[图片名,图片对应的摘要]
    """
    #定义语句接收返回数据的字典
    summary_image_dict:dict[str,str]={}
    #获取模型客户端对象
    vl_model=get_llm_client(lm_config.vl_model)

    #对图片信息列表进行遍历
    for image_name,image_path,image_content in image_info_list:
        #通过提示词工具类加载提示词
        prompt_text=load_prompt(name="image_summary",root_folder=root_folder,image_content=image_content)
        #封装HumanMessage
        # image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        # image_data = base64.b64encode(requests.get(image_url).content).decode("utf-8")
        image_path_obj=Path(image_path)
        image_data=base64.b64encode(image_path_obj.read_bytes()).decode("utf-8")

        message = HumanMessage(
            content=[
                #文本部分
                {"type": "text", "text": prompt_text},
                #图片部分
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{guess_type(image_name)[0]};base64,{image_data}"},
                },
            ]
        )
        #封装调用链
        #StrOutputParser 是类，StrOutputParser() 是实例；LCEL 管道 | 需要连接实例，所以写 StrOutputParser()
        chains=vl_model | StrOutputParser()
        #添加范围内限制(调用模型的时候，使用滑动窗口限速器对调用频率进行显示)
        apply_api_rate_limit()
        #调用获取结果
        image_summary=chains.invoke([message])
        #将图片名和摘要放到字典中
        summary_image_dict[image_name]=image_summary
        logger.debug(f"完成:{image_name}的视觉识别,对应的含义:{image_summary}")
    # 返回
    return summary_image_dict

@node_log("node_md_img")
def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 图片处理 (node_md_img)
    为什么叫这个名字: 处理 Markdown 中的图片资源 (Image)。
    """
    #TODO 1. 添加节点到运行时列表
    add_running_task(state.get("task_id"),"node_md_img")
    #TODO 2. 从状态中获取数据并进行校验       md_content,md_path_obj,md_images_dir_obj
    md_content,md_path_obj,md_images_dir_obj=step_1_validate_and_get_data(state)
    #判断解析后的md文件中是否存在图片,如果不存在,直接跳过当前节点
    if (not md_images_dir_obj) or len(list(md_images_dir_obj.iterdir())) == 0:
        logger.info(f"{md_path_obj}对应的md,没有图片内容,无需后续处理,直接跳出!!")
        return state

    #TODO 3. 获取图片信息和上下文            list[(图片名,图片路径,(上文,下文))]
    image_info_list:list[tuple[str, str, tuple[str, str]]]=step_2_scan_images(md_content,md_images_dir_obj)
    #TODO 4. 调用视觉模型生成图片摘要信息     dict{k:图片名,v:图片对应的摘要}
    summary_image_dict:dict[str,str]=step_3_image_summary(image_info_list,md_path_obj.stem)
    #TODO 5. 将图片上传到minio服务器         dict{k:图片名,v:图片在minio服务器上的网络地址}
    #TODO 6. 对md文件中图片的内容进行替换
    #TODO 7. 对新的md进行磁盘持久化存储
    #TODO 8. 更新状态  md_content  md_path
    #TODO 9. 添加节点到已完成列表
    add_done_task(state.get("task_id"),"node_md_img")
    return state

if __name__ == "__main__":
    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from utils.path_util import PROJECT_ROOT
    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output\hak180产品安全手册", "hak180产品安全手册.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟流程入参
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": ""
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")