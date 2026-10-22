import os
import time
import json

import generator

if __name__=="__main__":
    # 获取当前脚本的绝对路径
    current_file_path = os.path.abspath(__file__)
    # 获取当前脚本所在的目录
    current_dir = os.path.dirname(current_file_path)
    # 更改当前工作目录到脚本所在目录
    os.chdir(current_dir)

    case = generator.Chiplets("./", "case/", debug=True)
    case.generate()