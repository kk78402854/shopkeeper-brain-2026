"""
    @Author:MaYuhang
    @Time:2026/9/26
    @DESC:
"""
import os
from pathlib import Path

local_file_path = r"D:\dev\workspace\shopkeeper-brain-0708\doc"
# bn = os.path.basename(local_file_path)
# print(bn)
local_file_path_obj = Path(local_file_path)
print(local_file_path_obj.name)
print(local_file_path_obj.stem)
print(local_file_path_obj.suffix)
print(local_file_path_obj.parent)
print(local_file_path_obj.absolute())

new_path = local_file_path_obj / "new_path"