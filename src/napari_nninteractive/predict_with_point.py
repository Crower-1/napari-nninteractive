import os
from pathlib import Path
import numpy as np
import torch
import json
from batchgenerators.utilities.file_and_folder_operations import join
import mrcfile

# 假设 nnInteractive 模块和相关工具已经正确安装
import nnInteractive
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class

def model_predict(image, point, checkpoint_path, session_cfg, prompt=True):
    """
    对单个点提示进行模型推断，生成对应的 mask。

    参数:
        image (ndarray): 3D 图像数据，shape 为 (Z, Y, X)。
        point (tuple): 提示点坐标，要求为 (z, y, x)。
        checkpoint_path (str or Path): 模型检查点路径。若该目录下存在
            inference_session_class.json，则根据其中的配置加载对应推断类，
            否则使用默认 "nnInteractiveInferenceSession"。
        session_cfg (dict): 会话配置字典，其中至少需要包含 "spacing" 和 "shape" 信息。
        prompt (bool): 提示类型，True 表示正向提示，False 表示负向提示。

    返回:
        target_mask (ndarray): 经过模型推断得到的 mask，形状与 session_cfg["shape"] 一致，
            数据类型为 uint8。
    """
    checkpoint_path = Path(checkpoint_path)
    # --- 加载推断会话类 ---
    inference_class_file = checkpoint_path.joinpath("inference_session_class.json")
    if inference_class_file.is_file():
        with open(inference_class_file, 'r') as f:
            data = json.load(f)
        # 若文件内容为字典则获取对应字段
        inference_class_name = data.get("inference_class", "nnInteractiveInferenceSession")
    else:
        inference_class_name = "nnInteractiveInferenceSession"

    # 动态查找推断类（路径根据 nnInteractive 模块中的 inference 子模块）
    inference_class = recursive_find_python_class(
        join(nnInteractive.__path__[0], "inference"),
        inference_class_name,
        "nnInteractive.inference",
    )

    # --- 设备选择 ---
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # --- 创建会话 ---
    session = inference_class(
        device=device,
        use_torch_compile=False,
        torch_n_threads=os.cpu_count(),
        verbose=False,
        do_autozoom=False,
    )
    session.initialize_from_trained_model_folder(
        str(checkpoint_path),
        0,
        "checkpoint_final.pth",
    )

    # --- 设置图像 ---
    # nninteractive 要求输入图像加上 batch 维度
    image_data = image[np.newaxis, ...]
    session.set_image(image_data, {"spacing": session_cfg["spacing"]})

    # --- 设置目标缓冲区 ---
    # 创建一个与会话配置中 shape 相同的空数组，用于存放模型输出的 mask
    target_buffer = np.zeros(session_cfg["shape"], dtype=np.uint8)
    session.set_target_buffer(target_buffer)

    # --- 添加点提示交互 ---
    # 将输入点转换为形状 (1, 3) 的 numpy 数组（模型内部可能要求批量格式）
    point_data = np.array(point).reshape(1, -1)
    # 调用会话的 add_point_interaction 方法，注意此处 auto_run 设为 False，后续调用 _predict 执行推断
    session.add_point_interaction(point_data, prompt)

    # --- 执行模型推断 ---
    session._predict()

    # 返回更新后的目标缓冲区，即为模型输出的 mask
    return session.target_buffer

# 示例：如何使用该函数
if __name__ == '__main__':
    # 假设已存在一个 3D MRC 图像文件，并使用 get_tomo 加载图像
    def get_tomo(path):
        with mrcfile.open(path) as mrc:
            data = mrc.data
        return data

    # 输入文件路径
    mrc_path = "/media/liushuo/data1/data/synapse_seg/pp4001/pp4001.mrc"
    checkpoint = "/home/liushuo/.cache/huggingface/hub/models--nnInteractive--nnInteractive/snapshots/67d7990b9ff7d68bee63e30ca9d3c575f2b2add0/nnInteractive_v1.0"  # 请替换为实际模型路径

    # 加载图像
    img = get_tomo(mrc_path)
    # 假设 session_cfg 来自图像层信息，这里构造一个示例配置
    session_cfg = {
        "spacing": [1.0, 1.0, 1.0],  # 根据实际情况设置
        "shape": img.shape,          # 要求 mask 与原图形状一致
    }

    # 示例点：(z, y, x) 格式（注意：真实场景中应从 .coords 文件中读取，并转换）
    example_point = (43, 626, 368)  # 示例：对应原始 (x,y,z) 转换为 (z,y,x)

    # 调用 model_predict 得到 mask
    mask = model_predict(img, example_point, checkpoint, session_cfg, prompt=True)
    print("模型推断得到的 mask shape:", mask.shape)
