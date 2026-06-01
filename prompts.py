from __future__ import annotations

from typing import Any, Dict, List


def build_stage1_messages(target_image_data_url: str) -> List[Dict[str, Any]]:
    system = (
        "你是一名严格的医学图像分诊助手。"
        "你的任务是判断给定图片是否属于由多个视觉子图/面板组成的复合图。"
        "这里的 composite 指：一张最终图片中至少包含两个彼此独立、可以分别描述的子图或面板。"
        "例如：网格拼图、左右对比图、示意图和内镜图并列、带 A/B/C 或 1/2/3 标记且语义上可分开的多面板图。"
        "箭头、局部插图标记、单一场景上的注释，不应判为 composite。"
        "如果若干画面虽然看起来是多帧，但它们之间具有强顺序依赖，必须合在一起按步骤阅读才能表达一个完整的手术或操作过程，"
        "例如分步骤手术 montage，这类图不应判为 composite，而应视为一张整体图。"
        "只返回 JSON，且只包含以下键：is_composite, estimated_subfigure_count, confidence, reason。"
    )
    user_content = [
        {"type": "text", "text": "请判断这张图片是否属于可拆分的复合多面板图。"},
        {"type": "image_url", "image_url": {"url": target_image_data_url}},
    ]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def build_stage2_messages(
    *,
    target_image_data_url: str,
    source_final_description: str,
) -> List[Dict[str, Any]]:
    system = (
        "你是一名精确的医学复合图拆分助手。"
        "你会收到："
        "1）一张待判断/待拆分的复合图候选图片；"
        "2）该整张图对应的完整整图描述。"
        "你的任务是："
        "A）先判断这张图是否真的应该被拆成多个相互独立的子图；"
        "B）如果应该拆分，识别图中所有有意义的子图/面板；"
        "C）对每个子图，返回其在整张图上的归一化整数边界框 [x0,y0,x1,y1]，坐标范围为 0-1000；"
        "D）基于整图描述，为每个子图写出对应的子图描述。"
        "规则如下："
        "1）如果图中的多个画面是强顺序性的操作/手术过程帧，必须合在一起阅读才能表达一个完整有效的医学过程，则不要拆分。"
        "2）如果按照规则 1 不应拆分，则返回空的 subfigures 列表。"
        "3）纯文字 caption、装饰性留白、箭头、覆盖在图上的标签，不要当作独立子图。"
        "4）优先保证语义上正确的子图划分，同时边界框要尽量贴紧该子图的真实视觉内容。除非只保留极小安全边距，否则不要把大块外部白边、宽的空白间隙或附近空白区域框进去。"
        "5）所有返回的子图框都必须完全落在整张图内部。"
        "6）每个子图描述都要以整图描述作为主要语义来源。如果整图描述没有逐一明确列出每个子图，也要结合整段描述推断该子图最可能对应的描述，而不是退化成泛泛的纯视觉短语。"
        "7）不要把同一段整图描述原样复制给每个子图。每个子图描述都要聚焦该子图本身，同时与整图语义保持一致。"
        "8）每个子图描述都应是完整、信息充分的一到两句话，而不是一个简短短语。除了子图特有信息，还应在必要时保留共通背景，例如解剖部位、疾病背景、成像方式或操作语境，使该描述单独拿出来也自洽、有信息量。"
        "只返回 JSON，且只包含键：subfigures。"
        "每个 subfigure 只包含：bbox_norm1000_xyxy, description。"
    )
    user_content = [
        {"type": "text", "text": f"这张整图的完整描述如下：{source_final_description}"},
        {"type": "text", "text": "下面是待判断/待拆分的图片："},
        {"type": "image_url", "image_url": {"url": target_image_data_url}},
    ]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
