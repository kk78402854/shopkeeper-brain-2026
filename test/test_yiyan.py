"""
    @Author:MaYuhang
    @Time:2026/10/10
    @DESC:
"""
import requests

# url = f"https://v1.hitokoto.cn/?c=b"
# http_response = requests.post(url)
# # 获取响应的数据
# # http_response.text  以字符串的形式获取响应数据   http_response.content  以字节的形式获取响应数据  http_response.json()以json形式获取服务器数据-->dict
# response_dict = http_response.json()
# print("一言:"+response_dict["hitokoto"])
# print("出处:"+response_dict["from"])
# if response_dict["from_who"]:
#     print("作者:"+response_dict["from_who"])


# """
#     @Author:MaYuhang
#     @Time:2026/10/10
#     @DESC: 一言 API 封装为 FastAPI 接口
# """
#
# import uvicorn
# import requests
# from fastapi import FastAPI
# from fastapi.responses import HTMLResponse
# from pathlib import Path
#
# app = FastAPI(title="Hitokoto API")
#
#
# @app.get("/hitokoto")
# def get_hitokoto(c: str = "b"):
#     url = f"https://v1.hitokoto.cn/?c={c}"
#     http_response = requests.post(url)
#     response_dict = http_response.json()
#     return {
#         "hitokoto": response_dict["hitokoto"],
#         "from": response_dict["from"],
#         "from_who": response_dict["from_who"],
#     }
#
#
# @app.get("/", response_class=HTMLResponse)
# def index():
#     """返回前端页面"""
#     html_path = Path(__file__).parent / "index.html"
#     return html_path.read_text(encoding="utf-8")
#
#
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)

"""
    @Author:MaYuhang
    @Time:2026/10/10
    @DESC: 一言 API 封装为 FastAPI 接口（Session 复用连接）
"""
import uvicorn
import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pathlib import Path

app = FastAPI(title="Hitokoto API")

# 复用 TCP 连接，避免每次请求都重新握手（这是慢的主因）
_session = requests.Session()
_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
})


@app.get("/hitokoto")
def get_hitokoto(c: str = "b"):
    url = f"https://v1.hitokoto.cn/?c={c}&encode=json"
    # 用 Session 发请求，连接会被复用，快很多
    http_response = _session.post(url, timeout=10)
    response_dict = http_response.json()
    return {
        "hitokoto": response_dict["hitokoto"],
        "from": response_dict["from"],
        "from_who": response_dict["from_who"],
    }


@app.get("/", response_class=HTMLResponse)
def index():
    """返回前端页面"""
    html_path = Path(__file__).parent / "index.html"
    return html_path.read_text(encoding="utf-8")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)