from __future__ import annotations

from typing import Any, Dict, List


def build_stage1_messages(target_image_data_url: str) -> List[Dict[str, Any]]:
    system = (
        "你是一名严格的医学图像分诊助手。"
        "你的任务是对给定图片进行两步判断："
        "第一步：判断图片是否包含真实的内窥镜（消化内镜/胃肠镜/肠镜/胃镜等）图像。"
        "这里的“内窥镜图像”要求非常严格：必须是能够明确辨认出内窥镜光学视野的真实图像。"
        "内窥镜图像的典型特征：圆形或近圆形视野、黑色边角、粘膜组织纹理、腔道结构。"
        "只有在这些特征足够明确、能够较有把握地确认其为真实内窥镜图像时，才能判定 is_endoscopic=true。"
        "如果证据不足、图像模糊、视野过小、只有局部纹理但看不出明确内窥镜光学视野，或你无法稳定地区分它是不是内窥镜图像，则一律判定 is_endoscopic=false。"
        "宁可漏判，也不要把可疑图、边界图或非内窥镜图误判为内窥镜图。"
        "以下类型不属于内窥镜图像，应判定为 is_endoscopic=false："
        "手绘/电脑绘制的示意图、流程图、解剖示意图、统计图表、CT/MRI/X光/超声等非内窥镜影像学图像、"
        "纯文字/表格、病理切片显微镜图、照片（体表/术中开放手术）等。"
        "此外，器械照片、手术环境照片、咽喉或消化道相关但不具有明确内窥镜光学视野的普通临床照片，也都应判定为 is_endoscopic=false。"
        "注意：如果一张复合图中同时包含内窥镜子图和非内窥镜子图（如CT+内镜并列），仍判定 is_endoscopic=true，"
        "但前提是其中至少有一个子图能够被明确识别为真实内窥镜图像；如果只是疑似，不算。"
        "第二步：如果 is_endoscopic=true，再判断图片是否属于由多个视觉子图/面板组成的复合图。"
        "这里的 composite 指：一张最终图片中至少包含两个彼此独立、可以分别描述的子图或面板。"
        "例如：网格拼图、左右对比图、带 A/B/C 或 1/2/3 标记且语义上可分开的多面板图。"
        "箭头、局部插图标记、单一场景上的注释，不应判为 composite。"
        "如果若干画面虽然看起来是多帧，但它们之间具有强顺序依赖，必须合在一起按步骤阅读才能表达一个完整的手术或操作过程，"
        "例如分步骤手术 montage，这类图不应判为 composite，而应视为一张整体图。"
        "重要背景：这些图片从 PDF 医学报告中提取得到。"
        "因此图片周围或内部可能夹杂着 PDF 排版残留的文字、标题、图注、页眉页脚等非图像内容，"
        "这些文字区域不应被视为独立子图，请忽略它们。"
        "如果 is_endoscopic=false，则 is_composite 必须为 false，estimated_subfigure_count 必须为 0。"
        "只返回 JSON，且只包含以下键：is_endoscopic, is_composite, estimated_subfigure_count, confidence, reason。"
    )
    user_content = [
        {"type": "text", "text": "请判断这张图片：1）是否包含内窥镜图像；2）如果是，是否属于可拆分的复合多面板图。"},
        {"type": "image_url", "image_url": {"url": target_image_data_url}},
    ]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def build_stage2_vlm_messages(
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
        "C）对每个子图，返回其在整张图上的归一化整数边界框 [x0,y0,x1,y1]；"
        "D）基于整图描述，为每个子图写出对应的子图描述。"
        "坐标系定义：图片左上角为原点 (0,0)，x 轴向右、y 轴向下。"
        "将图片的实际宽度映射为 0-1000、实际高度映射为 0-1000。"
        "返回的 bbox_norm1000_xyxy 格式为 [x0, y0, x1, y1]，其中 (x0,y0) 是子图左上角，(x1,y1) 是子图右下角。"
        "【严格约束】所有坐标值必须是 0 到 1000 之间的整数（含 0 和 1000），严禁超出此范围。x0 必须 < x1，y0 必须 < y1。"
        "重要背景：这些图片来自内窥镜（消化内镜/胃肠镜等）领域的 PDF 医学报告。"
        "【只框选内窥镜图像】复合图中可能混杂多种类型的子图，你必须只框选真实的内窥镜图像面板。"
        "以下类型的面板不是内窥镜图像，必须跳过、不要框选："
        "- 光谱分析图、波形图、曲线图、统计图表"
        "- 手绘/电脑绘制的解剖示意图、流程图、光路图"
        "- CT/MRI/X光/超声等非内窥镜影像"
        "- 实物照片（如注射器、药瓶、手术器械的体外照片）"
        "- 病理切片显微镜图"
        "- 纯文字区域、表格、图注、页眉页脚等 PDF 排版残留"
        "内窥镜图像的典型特征：圆形或近圆形视野、黑色边角、粘膜组织纹理、腔道结构、内窥镜特有的光学成像外观。"
        "如果一个面板的内容不符合上述内窥镜特征，即使它有编号标签，也不要框选。"
        "规则如下："
        "1）如果图中的多个画面是强顺序性的操作/手术过程帧，必须合在一起阅读才能表达一个完整有效的医学过程，则不要拆分。"
        "2）如果按照规则 1 不应拆分，或者图中只有 0~1 个内窥镜面板（其余都是非内窥镜内容），则返回空的 subfigures 列表。"
        "3）纯文字 caption、装饰性留白、箭头、覆盖在图上的标签，不要当作独立子图。"
        "4）【关键】每个子图的边界框必须根据该子图的实际视觉边缘独立定位，不要假设子图是均匀网格排列。"
        "复合图中的子图经常大小不一、行列不齐、有不等宽的间隙。你必须逐个观察每个面板的真实像素边界来确定坐标。"
        "bbox 的四条边应尽量贴到图像内容本身，在不截断有效像素的前提下，把外围白边、分隔带和空白间隙压到最小。"
        "如果面板下方、上方或侧边带有 caption、编号说明或其他排版文字，边界框必须在这些文字之外收住，只保留面板图像本体，不要把说明文字裁进去。"
        "5）所有返回的子图框都必须完全落在整张图内部。"
        "6）每个子图描述都要以整图描述作为主要语义来源，但必须与该子图自身可见内容相互印证。"
        "如果整图描述没有逐一明确列出每个子图，也只能在图像证据与整图语义一致时做谨慎对应；证据不足时不要臆测。"
        "7）不要把同一段整图描述原样复制给每个子图。每个子图描述都要聚焦该子图本身，同时与整图语义保持一致。"
        "8）description 最好尽量按“部位 - 概览 - 细节”的信息顺序来组织，使结构更清楚，但不要求机械地使用固定标签、冒号或完全一致的句式。"
        "其中：部位只写能够从图像或整图描述中可靠确定的解剖部位/检查部位；如果无法可靠判断，就直接省略这部分。"
        "概览用于概括该子图最主要的可见发现、操作场景或整体表现；如果只能确认是普通内镜视野而无更明确结论，也要如实写，不要夸大。"
        "细节用于补充局部形态、颜色、表面结构、器械、病变边界、出血或分泌物等确实可辨认的信息；如果没有足够把握，就直接省略这些不确定细节"
        "9）禁止编造。凡是无法从该子图视觉内容或整图描述中可靠支持的信息，都不要写入 description。"
        "不要猜测病理结论、分期、具体诊断名称、精确部位或操作步骤，除非图像与整图描述都提供了足够证据。"
        "10）description 应信息清楚、自然、可读，优先写成 1 到 2 句；有把握的信息写进去，没把握的信息省略即可。不要写成长段，不要输出项目符号，也不要添加 description 之外的额外字段。"
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


def build_stage2_codex_prompt(
    *,
    sample_id: str,
    image_path: str,
    work_dir: str,
    splits_dir: str,
    projection_image_path: str,
    result_json_path: str,
    source_final_description: str,
    stage1_reason: str,
    estimated_subfigure_count: int,
) -> str:
    return (
        "你是负责医学复合图拆分的 agent。请用 Python/PIL 和文件系统完成子图定位、裁剪、复核，"
        "并把最终 JSON 写入 result_json_path。\n\n"
        "目标：从原图中拆出所有真实内窥镜图像面板。只要面板可以独立裁剪并描述，即使它们属于观察顺序、操作过程或病例流程，也应该拆分。"
        "跳过示意图、病理/影像学图、统计图、器械体外照片、纯文字、caption、页眉页脚等非内镜图像。有效内镜面板少于 2 个时返回 subfigures=[]。\n\n"
        "执行规范：\n"
        "- 先实际保存 crop 和 projection，再写 result_json_path；不要只返回计划路径。每个最终 crop 写入到 "
        f"{splits_dir}/{sample_id}__NN.jpg，NN 从 01 连续编号；不要保存中间 crop。\n"
        "- bbox_source_xyxy 使用原图像素坐标，crop 必须直接来自该 bbox，不缩放、不拼接、不额外加边。\n"
        "- bbox/crop 要贴合内镜面板图像本体：不漏掉有效图像边缘，不混入可避免的 caption、说明文字、分隔空白或相邻面板。\n"
        "- 多个面板竖向或横向紧邻时，白色间隔、分隔线、外部编号、箭头和右侧/下方说明文字都是面板外内容，不能框进去。"
        "每个 bbox 必须只覆盖一个独立内镜面板，不能跨越两个面板之间的空白分隔带，也不能为了包含文字说明而扩大到面板外。\n"
        "- 以真实图像矩形边界、黑色三角角区或内镜画面边缘作为裁剪边界。若中文标注/箭头是印在内镜图像内部且无法避开，可以保留；"
        "但如果文字位于面板外白底区域、caption 区或相邻说明区，必须排除。\n"
        "- 不能把同一个内镜面板从中间截断或只裁一部分。检查 crop 的四条边：如果边界外紧邻区域仍属于同一面板的图像内容、黑色角区或连续黏膜画面，必须外扩到完整面板边界；"
        "不要为了避开面板内部文字或箭头而截掉真实图像内容。\n"
        "- 用 Python/PIL 将最终 bbox 画到 projection_image_path，并重新打开 projection 和每个 crop 检查；不准就修正 bbox、覆盖 crop、重画 projection。无法确认准确的子图不要输出。\n"
        "- description 必须用中文写，主要从 source_final_description 中提取和对应；如果原图中有与该子图直接相关的 caption、编号或标注文字，可以适当结合。"
        "描述最好按“部位 - 概览 - 细节”的信息顺序组织，但不要机械使用固定标签或完全一致句式。证据不足的信息直接省略，切记不要瞎编；每个子图 1-2 句即可。\n\n"
        "result_json_path 的 JSON 要求：\n"
        "- 顶层包含 sample_id, image_path, projection_image_path, subfigures。\n"
        "- 每个 subfigure 包含 subfigure_index, bbox_source_xyxy, split_image_path, description。\n"
        "- split_image_path 和 projection_image_path 必须指向实际保存并复核过的最终文件；没有有效拆分时 projection_image_path 可为空字符串。\n"
        "- 不输出 schema 之外的字段。\n\n"
        f"sample_id: {sample_id}\n"
        f"image_path: {image_path}\n"
        f"work_dir: {work_dir}\n"
        f"splits_dir: {splits_dir}\n"
        f"projection_image_path: {projection_image_path}\n"
        f"result_json_path: {result_json_path}\n"
        f"stage1_reason: {stage1_reason}\n"
        f"estimated_subfigure_count: {estimated_subfigure_count}\n"
        f"source_final_description: {source_final_description}\n"
    )


def build_stage2_messages(
    *,
    target_image_data_url: str,
    source_final_description: str,
) -> List[Dict[str, Any]]:
    return build_stage2_vlm_messages(
        target_image_data_url=target_image_data_url,
        source_final_description=source_final_description,
    )
