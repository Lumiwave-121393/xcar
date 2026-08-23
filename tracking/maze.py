"""
maze.py —— 迷宫法边线提取（八邻域行走器）

参考：第16届智能车智能视觉组-上海交通大学AuTop战队开源算法讲解(二)边线提取

原理:
  赛道的边界线是一条连续曲线（良好的二值化是前提），因此不需要每帧都从
  图像中央向两侧扫描，而只需在图像下侧扫描一次得到边界线起始点，然后
  沿着边界线"一直走"即可。

  想象在起始点处有一个面朝图像上方的小人，黑色像素为墙，白色像素为路，
  小人保持左（右）手扶墙向前走（时刻脚踩白色、手扶黑色），直到走到图像
  边缘或步数上限。其走过的路径即为赛道边线。

  小人移动的三种情形（以左手为例，右手镜像）:
    1. 前进方向被墙挡住（前方像素为黑）
       → 小人右转（保持左手扶墙；右手扶墙则左转），原地重新判断
    2. 前进方向没有被挡住，且遇到了墙角（前方为白，且左前方为白）
       → 小人左转并斜向前进一格，绕过墙角（右手则判断右前方、右转）
    3. 前进方向没有被挡住，且当前不是墙角（前方为白，左前方为黑）
       → 小人向前走，方向不变

  针对赛道左边线使用"左手"巡线，赛道右边线使用"右手"巡线。

适配本项目:
  - 文章的"懒二值化"不需要：分割模型直接输出 seg_map → lane_mask。
  - 越界像素视为墙（黑）。
  - 增加 (位置, 朝向) 状态去重，防止在病态掩码（噪声洞）上无限打转。
"""

import numpy as np

# 8 方向（图像坐标，y 向下），按顺时针排列:
# N, NE, E, SE, S, SW, W, NW
_DIRS = [(0, -1), (1, -1), (1, 0), (1, 1),
         (0, 1), (-1, 1), (-1, 0), (-1, -1)]


def _is_road(mask, x, y):
    """越界视为墙（黑），否则返回该像素是否为路（白）"""
    h, w = mask.shape
    if x < 0 or y < 0 or x >= w or y >= h:
        return False
    return mask[y, x] > 0


def trace_boundary(mask, hand, start, max_steps=3000,
                   wrap_descent_px=20):
    """
    从起始点出发，沿边界按迷宫法行走，返回边界链。

    参数:
        mask:      (H, W) 二值图，255=路（白），0=墙（黑）
        hand:      "left"  左手扶墙（赛道左边线，墙在左侧）
                   "right" 右手扶墙（赛道右边线，墙在右侧）
        start:     (x, y) 起始点（白像素，且扶墙侧为黑）
        max_steps: 步数上限，防病态掩码上无限行走
        wrap_descent_px: 整块掩码防包绕：低于已达最高点超过此值（y 下行）
                   → 判为越过顶部回包，停止并裁剪到水平台末尾
                   （None/0 = 不启用）

    停止条件（按优先级）:
      1. 走到图像上边缘（y <= 0）
      2. 状态重复（成环）或步数上限
      3. 整块掩码防包绕：边线越过掩码顶部后持续下行（超过
         wrap_descent_px），停止并裁剪到最后一个"最高水平台"末尾，
         只保留上行段 + 顶部横向延伸段。

    2026-08-16 修改:
      - 原"连续 stall_steps 步未刷新最高点即停止"已退役——水平延伸
        （y 不刷新）在正常车道上是合法的横向路面/分割噪声台阶，
        横向段应完整保留而非截断；防包绕只认"下行"信号。
      - 裁剪点从"最后一个最高点"改为"最高水平台末尾"：保留横向段，
        只切掉下行段（下行段本应由对侧边界链负责）。

    返回:
        chain: [(x, y), ...] 按行走顺序的边界点列（含起点）

    注意:
      不再在图像左右边缘强制停止——掩码贴到画面边缘时，边线沿图像边缘
      继续上行（越界视为墙的规则保证行走器不会越界）。
    """
    x, y = start
    d = 0  # 初始朝向：图像上方 N
    chain = [(x, y)]
    seen = {(x, y, d)}

    min_y = y    # 已达最高点（最小 y，图像 y 向上减小）
    top_end = 0  # 链中"最高水平台"的最后一个下标（包绕裁剪点）

    for _ in range(max_steps):
        dx, dy = _DIRS[d]
        fx, fy = x + dx, y + dy

        if not _is_road(mask, fx, fy):
            # 情形1：前方被墙挡住 → 转向（左手右转 / 右手左转），原地重新判断
            d = (d + 1) % 8 if hand == "left" else (d - 1) % 8
        else:
            # 扶墙侧前方（左手: 左前 = 逆时针再进一步；右手: 右前 = 顺时针再进一步）
            sd = (d - 1) % 8 if hand == "left" else (d + 1) % 8
            sx, sy = fx + _DIRS[sd][0], fy + _DIRS[sd][1]

            if _is_road(mask, sx, sy):
                # 情形2：遇到墙角，墙拐走了 → 转向并斜向前进一格（绕过墙角）
                d = sd
                x, y = sx, sy
            else:
                # 情形3：直行
                x, y = fx, fy

        state = (x, y, d)
        if state in seen:
            # 状态重复 → 成环（病态掩码/死路），终止
            break
        seen.add(state)
        chain.append((x, y))

        # 走到图像上边缘 → 结束（文章：走到图像边缘）
        if y <= 0:
            break

        # 整块掩码防包绕：只认"越过最高点下行"为包绕信号。
        # 水平延伸（y == min_y）是合法横向路面/噪声台阶 → 完整保留；
        # 下行超过阈值 → 停止并裁剪到水平台末尾（切掉包绕下行段）。
        if y < min_y:
            min_y = y
            top_end = len(chain) - 1
        elif y == min_y:
            top_end = len(chain) - 1   # 水平台延伸 → 更新台尾
        elif wrap_descent_px and (y - min_y) > wrap_descent_px:
            chain = chain[:top_end + 1]
            break

    return chain


def find_start_points(mask, center_x=None, scan_rows=24, skip_bottom_px=0):
    """
    在图像底部向上扫描找左右边线起始点。

    从最下行向上逐行扫描，取第一个有路（白）的行:
      左起点 = 该行最左的白像素（其左侧必为墙）
      右起点 = 该行最右的白像素（其右侧必为墙）

    参数:
        mask:      (H, W) 二值图
        center_x:  保留参数（文章从图像中央向两侧扫描；本实现取最左/最右
                   白像素，对单路区域等价，对底部岔路也更稳）
        scan_rows: 自底部向上扫描的行数（默认 24）。2026-08-16 起调用方
                   可传更大值把搜索范围上扩（如 h//2 = 搜到画面中部）：
                   底部没有路时继续向上找，直到此范围内第一个有路的行。
        skip_bottom_px: 跳过画面最底边 N 行再从下往上扫（默认 0=不跳过）。
                   用于避开底边近场分割噪声/车身阴影；N 过大会直接丢线。

    返回:
        (left_start, right_start): (x, y) 或 None
    """
    h, w = mask.shape
    y0 = max(0, h - scan_rows)
    y_start = h - 1 - max(0, int(skip_bottom_px))   # 跳过最底部 N 行后的起始行

    left = right = None
    for y in range(y_start, y0 - 1, -1):
        xs = np.nonzero(mask[y])[0]
        if xs.size == 0:
            continue
        left = (int(xs[0]), y)
        right = (int(xs[-1]), y)
        break

    return left, right
