/**
 * encoder_share.h —— cpu0 / cpu1 共享编码器数据
 * ===============================================
 *
 * cpu0: PID 中断中写入左右轮编码器数据
 *       UART 接收中断中写入 g_target_speed
 * cpu1: 主循环中只读这些变量，组装帧后通过 UART3 TX 发回上位机
 *
 * 注意:
 *   - cpu1 需要 #include "encoder_share.h" 即可访问所有变量
 *   - 单字长(int16/float)读写是 TriCore 硬件级别的原子操作，无需自旋锁
 *   - cpu0 必须在 cpu1 启动前调用 uart_init(UART_3, ...) 初始化硬件
 */

#ifndef ENCODER_SHARE_H
#define ENCODER_SHARE_H

#include "zf_common_headfile.h"

/** 右轮：最近一个 PID 周期(10ms)的编码器脉冲增量，有符号，cpu0 写入 */
extern int16  g_encoder_count_right;

/** 右轮：实测速度 = (float)g_encoder_count_right，cpu0 写入 */
extern float  g_actual_speed_right;

/** 左轮：最近一个 PID 周期(10ms)的编码器脉冲增量，有符号，cpu0 写入 */
extern int16  g_encoder_count_left;

/** 左轮：实测速度 = (float)g_encoder_count_left，cpu0 写入 */
extern float  g_actual_speed_left;

/** 上位机下发的目标速度 [-1000, 1000]，cpu0 写入 */
extern int16  g_target_speed;

#endif
