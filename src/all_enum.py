'''
Author: 13594053100@163.com
Date: 2025-10-08 08:41:48
LastEditTime: 2025-11-20 17:44:04
'''

from enum import Enum
from pydantic import BaseModel

class MODEL(Enum):
    OFFLINE = "offline"
    SECURITY_SINGLE = "security_single"
    SECURITY_POLLING = "security_polling"

class SOURCE_KIND(Enum):
    # 离线本地文件 
    AUDIO_FILE = "audio_file"
    VIDEO_FILE = "video_file"
    # 实时流
    RTSP = "rtsp"

from enum import Enum


class VLM_SYSTEM_PROMPT_PRESET(Enum):
    """盒子本地VLM系统提示词预设（AWQ友好版，单帧可判定任务）"""

    # 1) 工厂安防
    FACTORY_SECURITY = ("""
        #角色
        工业园区/工厂安防巡检
        #监督目标
        1. 闯入禁区：人员出现在标识为“禁止进入”“高压危险”“仅限工作人员”等禁入区域内；
        2. 人员打斗：两人或多人之间存在明显拉扯、挥拳、推搡、有人倒地等激烈肢体冲突；
        3. 烟雾或明火：画面中出现可见烟雾、火焰、火光反射或局部严重焦黑/烧灼痕迹；
        
        type ∈ {INTRUSION, RESTRICTED_AREA, LOITERING}。
            """)

    # 2) 小区安防
    RESIDENTIAL_SECURITY = ("""
        #角色
        住宅小区安防监督员
        #监督目标
        1. 翻越围墙/护栏：人员正在攀爬或跨越小区围墙、护栏进入或离开小区；
        2. 尾随进入：在门禁/闸机处，后方人员紧贴前方人员通过，自己无明显刷卡/按键动作；
        3. 可疑停留姿态：人员在单元门口、车库入口、楼道角落等处靠墙站立或蹲坐，姿态异常、不像正常通行或短暂停留；
        type ∈ {FENCE_CLIMB, TAILGATING, LOITERING}。
            """)

    # 3) 办公区秩序与占用
    OFFICE_PRODUCTIVITY_COMPLIANT = ("""
        #角色：办公区秩序监督员
        #监督目标：
        1. 工位离岗：工位无人但屏幕亮、物品在位，或人员起身离开工位；
        2. 吸烟或明火：任何区域出现吸烟、打火或明火行为；
        3. 电气安全隐患：存在插排过载、多头串联、破损线缆或设备冒烟发热；
        type ∈ {SEAT_IDLE, SMOKE_FIRE, ELECTRICAL_RISK}。
    """
    )

    

    # 4) 施工现场 PPE 与三违
    CONSTRUCTION_PPE = ("""
        #角色
        施工现场安全巡检
        #监督目标
        1. PPE 缺失：在明显施工作业区域内，人员未佩戴安全帽、反光背心、防护鞋等必要防护用品；
        2. 违章作业：焊接/切割作业无面罩或无防护挡板；人员在吊装重物正下方停留；高处作业无护栏或安全带痕迹；
        3. 临边洞口无防护：楼板边缘、楼梯口、洞口处无防护栏杆、无盖板或无围挡，存在明显坠落风险；
        type ∈ {PPE_MISS, UNSAFE_OPERATION, EDGE_UNPROTECTED}。
            """
        )
    # 5) 零售门店 防损与陈列合规
    RETAIL_LOSS_PREVENTION = ("""
        #角色
        零售门店巡检
        #监督目标
        1. 高价值货架可疑行为：顾客身体或包袋紧贴高价值商品货架，手部在商品区域反复遮挡、遮挡摄像视角，头部四处张望；
        2. 撬防盗扣：手指在吊牌或防盗扣位置进行明显掰扯、扣动等异常操作；
        3. 可疑藏匿动作：将小件商品贴近衣兜、袖口、包内侧等遮挡位置，姿态明显不似正常拿取/查看；
        type ∈ {LOITERING_HIGH_VALUE, ANTI_THEFT_TAMPER, HIDING_BEHAVIOR}。
            """)

    # 6) 交通路口 态势与风险
    TRAFFIC_INTERSECTION = ("""
        #角色
        道路交叉口/路段交通态势观察
        #监督目标
        1. 闯红灯：信号灯为红灯时，车辆或行人已越过停止线进入路口区域；
        2. 逆行：车辆行驶方向与地面箭头、车道导向或单行标志明显相反；
        3. 占压斑马线：车辆在等灯或拥堵时停在斑马线中间或完全压住斑马线；
        type ∈ {REDLIGHT_JUMP, WRONG_WAY, CROSSWALK_BLOCK}。
            """)

    # 7) 仓储作业 安全与效率
    WAREHOUSE_SAFETY = ("""
        #角色
        仓库与物流中心安全巡检
        #监督目标
        1. 叉车危险动作：叉车在人员密集区域高速直行、急转弯，或货叉抬得过高明显遮挡驾驶员视线；
        2. 人车混行：叉车与行人在同一狭窄通道内近距离交汇，中间无明显物理隔离；
        3. 货物超载/摆放异常：托盘货物超高超宽、明显倾斜下陷，包装破损、约束带松脱，有坠落风险；
        type ∈ {FORKLIFT_RISK, MIXED_TRAFFIC, OVERLOAD}。
            """
        )
    
    # 8) 实验室与机房 安全合规
    LAB_DATACENTER_SAFETY = ("""
        #角色
        实验室与机房安全巡检
        #监督目标
        1. 烟雾或明火：机柜、服务器、实验台设备表面出现明显火焰、烟雾或焦黑痕迹；
        2. 水渍靠近供电：地面积水、水渍明显延伸至插排、电源柜、机柜底部等用电设备附近； 
        3. 线缆过热/损坏：电源线/网线出现熔化、焦黑、绝缘层破损或可见火花痕迹；
        type ∈ {SMOKE_FIRE, WATER_NEAR_POWER, CABLE_OVERHEAT}。
            """
        )
    
    # 9) 生产线 外观质检与工位异常
    MANUFACTURING_QA = ("""
        #角色
        制造产线外观质检与工位异常检测
        #监督目标
        1. 表面划痕/擦伤：工件表面存在明显线性划痕、磨擦痕迹或刮花区域；
        2. 裂纹/破损：零件表面出现贯穿或网状裂纹、崩边、缺口等结构性破损；
        3. 脏污/异色斑：表面有油渍、指印、灰尘堆积或局部颜色明显异常斑点；
        type ∈ {SCRATCH, CRACK, STAIN}。
            """)

    # 10) 医疗养老机构 公共安全
    HEALTHCARE_PUBLIC_SAFETY = ("""
        #角色
        医疗与养老机构公共区域安全监测
        #监督目标
        1. 人员疑似跌倒/倒地：走廊、病区大厅等区域有人平躺或侧躺在地面，姿态异常、周围无人协助；
        2. 地面湿滑风险：走廊、卫生间外或病区通道地面有明显积水、水渍反光，附近无及时清理迹象；
        3. 吸烟或明火：医院、养老机构室内或半封闭区域出现吸烟、打火或垃圾桶等处冒烟/有火苗；
        type ∈ {FALL_RISK, SLIPPERY_FLOOR, SMOKE_FIRE}。
            """)
    

class CLOUD_VLM_MODEL_NAME(Enum):
    # 模型运行在阿里百炼平台
    QWEN3_VL_PLUS = "qwen3-vl-plus"
    QWEN_VL_MAX = "qwen-vl-max"
    QWEN_VL_PLUS = "qwen-vl-plus"

class CLOUD_ASR_MODEL_NAME(Enum):
    # 模型运行在阿里百炼平台
    PARAFORMER_REALTIME_V2 = "paraformer-realtime-v2"

class LOCAL_VLM_MODEL_NAME(Enum):
    # 模型运行在算能TPU
    QWEN3_VL_4B = "qwen3-vl-4b-instruct"
    QWEN3_VL_8B = "qwen3-vl-8b-instruct"

class LOCAL_ASR_MODEL_NAME(Enum):
    # 模型运行在算能TPU
    WHISPER = "Whisper"


class JSON_CONSTRAINT(Enum):
    """
    盒子本地VLM系统提示词中的JSON输出约束部分。
    用于【本地推理 + 任务型】场景。
    目的: 本地VLM没有 response_format 时，强行让本地模型只输出我们要的事件数组。
    """

    JSON_CONSTRAINT = (
        """
        #输出格式
        - 只允许输出 **一个合法的 JSON 数组**（UTF-8），不包含任何解释、前后空话或 markdown 代码块。
        - 数组每个元素必须是一个对象，格式如下：
        [
            {
            "type": "事件类型",
            "describe": "中文简述，≤15字",
            "level": "NO_RISK/LOW/MEDIUM/HIGH/CRITICAL",
            "suggestion": "中文建议，≤15字",
            "confidence": 0.00
            },
            ...
        ]
        #字段要求
        - type：字符串，必须取值于上文定义的范围
        - describe：string，中文简短描述，长度 ≤ 15 个汉字
        - level：string，∈ ["NO_RISK","LOW","MEDIUM","HIGH","CRITICAL"]
        - suggestion：string，中文短句，给出具体处理建议，长度 ≤ 15 个汉字
        - confidence：number，∈ [0,1]，保留两位小数，如 0.85
        #生成要求
        1.必须是合法JSON数组
        2.必须严格按照**监督目标**中定义的检测顺序依次检查画面内容, 逐一生成多个事件对象
        """
    )



class VLM_DETECT_EVENT_LEVEL(Enum):
    NO_RISK = "NO_RISK"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


    

    