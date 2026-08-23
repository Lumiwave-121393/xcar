/**
 * cpu1_main.c —— 编码器数据定时上报
 * =====================================
 *
 * 运行在 TC264 的 CPU1 核心上，负责将编码器数据通过 UART3 TX 发回上位机。
 *
 * 工作原理:
 *   - cpu0 在 PID 中断(10ms)中更新左右轮编码器数据
 *   - cpu0 在 UART 协议解析中更新 g_target_speed
 *   - cpu1 主循环每 50ms 读取这些共享变量，组装帧 → uart_write_buffer()
 *
 * 帧格式（上行，MCU → 上位机）:
 *   AA 55 04 0A [R_enc int16 BE] [R_spd int16 BE] [L_enc int16 BE] [L_spd int16 BE] [tgt int16 BE] [XOR]
 *
 * 依赖:
 *   - encoder_share.h 提供 extern 变量声明
 *   - cpu0 必须在 core0_main 的 hardware_init() 中调用 uart_init(UART_3, ...) 初始化硬件
 *   - uart_write_buffer() 是逐飞库的标准 UART 发送 API
 */

#include "zf_common_headfile.h"
#include "encoder_share.h"
#pragma section all "cpu1_dsram"

// ====================================================================
// 1. 协议常量（与 cpu0_main.c 严格一致）
// ====================================================================

#define FRAME_HEADER1        0xAA
#define FRAME_HEADER2        0x55
#define CMD_ENCODER_REPORT   0x04
#define REPORT_INTERVAL_MS   50           // 上报间隔 50ms = 20Hz

// ====================================================================
// 2. 帧发送
// ====================================================================

/**
 * 通过 UART3 TX 发送一帧二进制数据
 *
 * 帧结构: AA 55 [cmd 1B] [len 1B] [data NB] [XOR 1B]
 *
 * @param cmd   命令字节
 * @param data  载荷数据指针
 * @param len   载荷长度（字节数）
 */
static void uart_send_frame(uint8 cmd, uint8 *data, uint8 len)
{
    uint8 buf[16];
    uint8 checksum = 0;

    buf[0] = FRAME_HEADER1;          // 0xAA
    buf[1] = FRAME_HEADER2;          // 0x55
    buf[2] = cmd;
    buf[3] = len;

    checksum ^= cmd;
    checksum ^= len;

    for (uint8 i = 0; i < len; i++) {
        buf[4 + i] = data[i];
        checksum ^= data[i];
    }
    buf[4 + len] = checksum;

    uart_write_buffer(UART_3, buf, 5 + len);
}

// ====================================================================
// 3. 编码器上报
// ====================================================================

/**
 * 读取共享变量，组装编码器帧并发送
 *
 * 共享变量由 cpu0 维护:
 *   g_encoder_count_right — PID 中断(10ms)中由 encoder_get_count(ENCODER_CH) 更新
 *   g_actual_speed_right  — 同上，=(float)g_encoder_count_right
 *   g_encoder_count_left  — PID 中断(10ms)中由 encoder_get_count(ENCODER_CH_LEFT) 更新
 *   g_actual_speed_left   — 同上，=(float)g_encoder_count_left
 *   g_target_speed        — UART 接收中断中由上位机控制帧更新
 *
 * 注意: 单字长读取(int16/float)在 TriCore 上是原子操作，无需自旋锁
 */
static void send_encoder_report(void)
{
    /* 原子读取共享变量 */
    int16 enc_r   = g_encoder_count_right;
    int16 spd_r   = (int16)g_actual_speed_right;   // float → int16 截断
    int16 enc_l   = g_encoder_count_left;
    int16 spd_l   = (int16)g_actual_speed_left;    // float → int16 截断
    int16 tgt_spd = g_target_speed;

    /* 组装 10 字节载荷（5×int16 BE） */
    uint8 data[10];
    data[0] = (uint8)((enc_r   >> 8) & 0xFF);
    data[1] = (uint8)( enc_r          & 0xFF);
    data[2] = (uint8)((spd_r   >> 8) & 0xFF);
    data[3] = (uint8)( spd_r          & 0xFF);
    data[4] = (uint8)((enc_l   >> 8) & 0xFF);
    data[5] = (uint8)( enc_l          & 0xFF);
    data[6] = (uint8)((spd_l   >> 8) & 0xFF);
    data[7] = (uint8)( spd_l          & 0xFF);
    data[8] = (uint8)((tgt_spd >> 8) & 0xFF);
    data[9] = (uint8)( tgt_spd        & 0xFF);

    uart_send_frame(CMD_ENCODER_REPORT, data, 10);
}

// ====================================================================
// 4. 主函数
// ====================================================================

void core1_main(void)
{
    disable_Watchdog();                     // 关闭看门狗
    interrupt_global_enable(0);             // 关全局中断（本核心不需要中断）

    /*
     * 等待 cpu0 完成外设初始化（最关键的是 uart_init(UART_3, ...)）。
     * cpu_wait_event_ready() 阻塞直到所有核心都到达该调用点。
     */
    cpu_wait_event_ready();

    /*
     * 再给 100ms 余量，确保 UART 硬件完全就绪、电平稳定。
     * 对于 115200 bps 链路，这段时间足够发送约 1KB 数据，初始化绰绰有余。
     */
    system_delay_ms(100);

    while (TRUE)
    {
        send_encoder_report();
        system_delay_ms(REPORT_INTERVAL_MS);   // 50ms = 20Hz
    }
}
#pragma section all restore
