"""
config.py —— 目标检测与避障参数常量

所有可调参数集中在此，方便按赛道环境调整。
"""

# ====================================================================
# 目标检测模型类别
# ====================================================================
#
# 注意: 类别数量由模型输出 shape 自动推断 (K - 4 = num_classes)
#       模型输出 (1, 8400, K) → K=4(bbox) + num_classes
#       当前模型 K=9 → 5 个类别
#       更换模型后请同步更新此列表，使 CLASS_NAMES 长度 = num_classes
# ====================================================================

CLASS_CAR   = 0
CLASS_GOLD  = 1
CLASS_HUMAN = 2
CLASS_SIGN  = 3
CLASS_STOP  = 4


CLASS_NAMES = ['car', 'gold', 'human', 'sign', 'stop']

# 需要避障的类别及其处理方式
OBSTACLE_CLASSES = {
    CLASS_HUMAN: "pedestrian",   # 行人 → 停车等待 / 绕行
    CLASS_CAR:   "vehicle",      # 车辆 → 偏移绕行
}

# 路牌类别（需要 OCR 识别）
SIGN_CLASSES = {CLASS_SIGN}

# ====================================================================
# 检测参数
# ====================================================================

DETECT_CONFIDENCE = 0.5          # 目标检测置信度阈值
NMS_THRESHOLD = 0.45             # NMS 阈值
DETECT_EVERY_N_FRAMES = 1        # 每 N 帧检测一次（降低 NPU 负载）
DETECT_IMG_SIZE = (320, 320)     # 检测模型输入尺寸（原 640×640 → 320×320，推理提速约4倍）

# 检测模型 NPU 核心分配（避免与循迹模型冲突）
# 循迹模型默认用 core 0 → 检测用 core 1,2
DETECT_NPU_CORE_OFFSET = 1

# ====================================================================
# 多目标追踪参数
# ====================================================================

TRACK_IOU_THRESHOLD = 0.5        # IoU 匹配阈值
TRACK_MAX_AGE = 30               # 最大丢失帧数（超过则删除轨迹）
TRACK_MIN_HITS = 2               # 最小命中帧数（确认有效轨迹）
TRACK_VELOCITY_WINDOW = 5        # 计算速度的滑动窗口帧数

# 行人 "移动" 判定
PEDESTRIAN_MOVE_THRESHOLD = 3   # 像素/帧，越过此阈值视为正在移动

# 障碍物接近阈值（以下两个条件同时满足才触发避障）
# 1. bbox 底部 y2 / frame_height ≥ Y2_RATIO（物体进入画面下半部分）
# 2. bbox 面积 / 画面面积 ≥ AREA_RATIO（物体达到足够大小）
# 值越大要求越近/越大才触发避障
PEDESTRIAN_PROXIMITY_Y2_RATIO = 0.4     # 行人 y2 门控
PEDESTRIAN_PROXIMITY_AREA_RATIO = 0.0035 # 行人面积占比门控（v2 恢复启用，与车辆一致，过滤小噪声框）
VEHICLE_PROXIMITY_Y2_RATIO = 0.4        # 车辆 y2 门控
VEHICLE_PROXIMITY_AREA_RATIO = 0.005  # 车辆面积占比门控（2026-08-16 恢复：过滤 <0.5% 噪声框）

# ====================================================================
# 避障决策参数
# ====================================================================

# 行人
PEDESTRIAN_WAIT_TIMEOUT = 5.0    # 等待行人离开的超时（秒）【暂未使用，v2 无超时强制绕行】
PEDESTRIAN_WAIT_FRAMES = 150     # 等待行人离开的超时（帧，~5s@30fps）【暂未使用】
PEDESTRIAN_BYPASS_OFFSET = 0  # 行人绕行偏置幅度（沿用旧值 0.1；符号由绕行方向动态决定：左绕负/右绕正）
PEDESTRIAN_MOVE_DIR_THRESHOLD = 5  # 行人运动方向判定阈值 (px/frame)【v2.1】：|side×vx|>5 才算让行/侵入；
                                  # side=行人相对车道中线的侧向（+1右侧/-1左侧），vx=追踪器5帧滑动平均

PEDESTRIAN_BYPASS_SPEED = 1600          # 行人绕行偏置时的速度（比默认速度慢，可调试）

# 行人 v2（2026-08-16 车道边界感知）：占道判定用"边界链 + 行人中心 x"，
# 采样点从 y2 下移 PEDESTRIAN_IN_LANE_SAMPLE_DOWN_PX——行人站在车道里会在
# 分割图中形成空洞、边界链在行人行绕洞内凹，下移后取到干净的真实边界。
# 掩码法（bbox 底部 25% 行带内车道掩码占比 ≥ PEDESTRIAN_IN_LANE_RATIO）仅作
# 缺链时的兜底。
PEDESTRIAN_IN_LANE_RATIO = 0.3          # 占道判定兜底阈值（掩码法）
PEDESTRIAN_IN_LANE_SAMPLE_DOWN_PX = 10  # 边界链采样点从 y2 下移的像素数

# 车辆
# 车辆绕行固定偏置（2026-08-17 起按"循迹手性 × 绕行方向"四象限分别可配，
# 板测某方向绕行效果差时只调对应格，不影响其他组合）[-1,1]
# 符号约定：左绕为负、右绕为正；幅值一律取正数，符号由绕行方向决定
VEHICLE_BYPASS_STEER_LH_LEFT = 0.5     # 左手循迹 · 左绕
VEHICLE_BYPASS_STEER_LH_RIGHT = 0.3    # 左手循迹 · 右绕
VEHICLE_BYPASS_STEER_RH_LEFT = 0.4     # 右手循迹 · 左绕
VEHICLE_BYPASS_STEER_RH_RIGHT = 0.1    # 右手循迹 · 右绕
VEHICLE_SLOW_SPEED = 1200           # 绕行车辆时的减速速度（绕行/去抖全程；恢复后立即回 DEFAULT_SPEED）

# 车辆绕行退出去抖（2026-08-16）：只有"本帧完全检测不到车辆"才计缺席帧，
# 连续缺席 VEHICLE_RETURN_MISS_FRAMES 帧后直接恢复正常循迹（防单帧丢检抖动）；
# 车辆重新出现即清零。去抖期间保持绕行（锁定方向 + 固定偏置 + VEHICLE_SLOW_SPEED）。
# 2026-08-16 二次修改（用户方案）：删除回中阶段——不再设置与绕行相反的
# 回中偏置（VEHICLE_RETURN_BIAS_RATIO/DEAD_ZONE/TIMEOUT 已删除）；去抖达标后
# 绕行偏置瞬时归零、速度立即恢复 DEFAULT_SPEED，由正常循迹的 P 反馈按偏差
# 比例自然回中（闭环回中，越靠近中线力度越小，避免开环回中偏置的过冲/蛇形）。
VEHICLE_RETURN_MISS_FRAMES = 25   # 连续缺席帧数阈值（约 0.3~0.5s @10~15fps）

# 车辆绕行方向判断（2026-08-16 v5，用户方案）：躲车期间每帧连续判断占道方向，
# 迟滞滤波 + 并排锁死，替代原"触发时一次判断锁死"（远处判决不可靠，锁错错到底）。
VEHICLE_JUDGE_Y2_RATIO = 0.5    # 轻量距离门控：y2 低于此值只减速、不判方向不给偏置
                                # （远处判决不可靠；达到后才开始连续判断+迟滞）
VEHICLE_DIR_FLIP_FRAMES = 5     # 迟滞：相反判决需连续 N 帧才翻转绕行方向（防判决抖动）
VEHICLE_DIR_LOCK_Y2_RATIO = 0.8 # 并排锁死线：y2 达到后方向锁死直到车辆离开
                                # （并排时 bbox 被画面边缘截断、中心估计退化，防误翻转）

# 行人 v2 迟滞/去抖（2026-08-16 用户决定：复用车辆参数、不用并排锁死）
PEDESTRIAN_RETURN_MISS_FRAMES = 10  # 消失去抖：连续 N 帧无新鲜检测 → 恢复正常
PEDESTRIAN_DIR_FLIP_FRAMES = VEHICLE_DIR_FLIP_FRAMES        # 方向迟滞：相反判决连续 N 帧才翻转；
                                                             # 停车→绕行同样用此帧数（连续 N 帧 must_stop=False 才转）

# 车道占用判断
LANE_OVERLAP_THRESHOLD = 0.15    # 检测框与车道重叠超过此比例视为占用

# ====================================================================
# 路牌处理参数
# ====================================================================

SIGN_CROP_MARGIN = 10            # 裁剪路牌区域时向外扩展的像素
SIGN_OCR_TIMEOUT = 15.0          # OCR 超时（秒）
SIGN_LLM_TIMEOUT = 15.0          # LLM 调用超时（秒）
SIGN_TEMP_DIR = "obstacle_detection/temp"  # 临时文件目录

# 路牌触发条件（避免远处小图 OCR 识别不全）
SIGN_OCR_BBOX_AREA = 0.035        # 启动 OCR 的最小路牌面积占比（较小→先启动OCR）
SIGN_STOP_BBOX_AREA = 0.035       # 触发停车的最小路牌面积占比（较大→后停车）
SIGN_MIN_TRACK_HITS = 2          # 触发前最少追踪命中帧数，确保稳定检测
SIGN_MAX_BBOX_AREA = 0.1        # 路牌检测框面积上限（超过画面20%视为场地外干扰，丢弃）
PEDESTRIAN_MAX_BBOX_AREA = 0.03  # 行人检测框面积上限（超过画面8%视为场地外干扰，丢弃）
SIGN_COOLDOWN_SECONDS = 8.0     # 路牌决策执行后的冷却时间（秒），冷却期内忽略所有路牌
FORK_RIGHT_HAND_SECONDS = 5.0   # 右岔路决策后切换右手循迹的持续时间（秒），板上按支路长度实测调整
FORK_RIGHT_BIAS = 0.6           # 【已废弃】右岔路偏置值（旧逻辑：加到 track_offset 上偏右循迹）
                                # 2026-08-14 起右岔路改为"切换右手循迹 N 秒"（FORK_RIGHT_HAND_SECONDS），
                                # 该常量仅保留供 navigation.py 旧代码 import，不再被 full_pipeline 使用

# stop 牌处理：程序启动 STOP_GATE_SECONDS 秒后才启用；之后识别到 stop
# （新鲜检测 + hits ≥ STOP_MIN_HITS）→ 继续行驶 STOP_DELAY_SECONDS 秒
# （首次识别起算，期间消失不重置计时）→ 执行 STOP_ACTION 指定动作。
# 两种模式：
#   "stop"      → 永久停车（速度 0、舵机保持，不再恢复）；
#                 优先级：路牌 > 行人 > 车辆 > stop > 正常。
#   "blind_box" → 盲盒任务（2026-08-21 用户方案）：舵机向右打固定角 + 低速
#                 行驶，锁存后最高优先级（覆盖行人/车辆/路牌强制停车），
#                 保持到 r 重置/程序结束。
STOP_GATE_SECONDS = 10000.0        # 程序启动后多少秒才启用 stop 处理
STOP_DELAY_SECONDS = 3.0        # 首次识别到 stop 后继续行驶的秒数，到点执行 STOP_ACTION
STOP_MIN_HITS = 2               # stop 触发 hits 门槛（确保稳定检测）
STOP_ACTION = "blind_box"       # "stop"=原永久停车 | "blind_box"=盲盒任务

# ── 盲盒任务（STOP_ACTION="blind_box" 时生效）──
BLIND_BOX_STEER_US = 150        # 右打固定角：相对舵机中位 1500µs 的偏移，正=右转（输出=1500+此值）
BLIND_BOX_SPEED = 800           # 低速行驶速度（正常循迹基速 2000，避让低速 1000）

# 路牌 Toggle 模式
# True:  第一次路牌正常 OCR+LLM，第二次路牌直接执行相反决策（不停车、不调OCR+LLM），
#        第三次及之后全部忽略，只正常循迹。
# False: 每次路牌都正常 OCR+LLM 流程（原始行为）。
SIGN_TOGGLE_MODE = True

# 路牌三级接近参数（检测到路牌 → 减速接近 → 停车）
SIGN_SLOW_MIN_HITS = 2           # 减速接近前最少追踪命中帧数（刚稳定检测就减速）
SIGN_SLOW_MIN_AREA = 0.015       # 减速接近的最小面积占比（≈3m外开始减速）
SIGN_APPROACH_SPEED = 800        # 减速接近时的速度（比 SLOW_SPEED 更低，平顺过渡到停车）

# 路牌 LLM 决策类型
class SignDecision:
    STRAIGHT   = "straight"       # 直行
    FORK_RIGHT = "fork_right"     # 走右岔路

# LLM 返回格式模板
LLM_DECISION_PROMPT = """你是一个自动驾驶小车的决策系统。根据以下路牌的OCR识别内容，判断路牌指示小车应该向左走、向右走还是直行。

路牌OCR识别内容：
{ocr_text}

请判断路牌指示的方向：
- left: 向左走
- right: 向右走
- straight: 直行

请严格按照以下JSON格式回复，不要包含其他文字：
{{"action": "<left|right|straight>", "confidence": <0.0-1.0>, "reason": "<简短的判断理由>"}}"""

# ====================================================================
# 控制参数
# ====================================================================

DEFAULT_SPEED = 2200                # 默认前进速度
SLOW_SPEED = 1800                   # 减速速度
STOP_SPEED = -250                   # 停车速度
MAX_STEER_OFFSET = 0.7           # 避障时最大额外转向偏移

# 串口默认值
DEFAULT_SERIAL_PORT = "/dev/ttyUSB0"
DEFAULT_BAUDRATE = 115200

# ====================================================================
# 可视化
# ====================================================================

DRAW_DETECT_BOXES = True         # 是否在画面上绘制检测框
DRAW_TRACK_IDS = False            # 是否显示追踪 ID
DRAW_LANE_OVERLAP = True         # 是否高亮与车道重叠的区域
BOX_COLORS = {
    "pedestrian": (0, 0, 255),    # 红色
    "vehicle":    (255, 0, 0),    # 蓝色
    "sign":       (0, 255, 255),  # 黄色
    "unknown":    (128, 128, 128),# 灰色
}

# ====================================================================
# 工具函数
# ====================================================================

def clamp(value, low, high):
    """数值限幅"""
    return max(low, min(high, value))
