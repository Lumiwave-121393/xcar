/**
 * cpu0_main.c —— 香橙派 ↔ TC264 串口通信 + 舵机 + 双电机差速 + 编码器 PID
 * ======================================================================
 *
 * 功能概述:
 *   通过 UART0 接收香橙派发来的二进制控制帧，解析后:
 *   1. 舵机 —— 50Hz PWM (P33.9) 控制前轮转向
 *   2. 双电机 —— 两个 DRV8701E 驱动后轮，差速辅助转向，编码器 PID 闭环
 *
 * 通信协议（与 car_control.py 对应，8 字节定长帧）:
 *   AA 55 [cmd 1B] [len 1B] [舵机 uint16 BE] [电机 int16 BE] [XOR 校验]
 *
 * 安全保护:
 *   - 心跳超时 1 秒 → 自动停车
 *   - 控制超时 500ms → 舵机回中 + 速度归零
 *   - 帧校验失败 → 丢弃，不影响当前状态
 *
 * 硬件连线（基于逐飞例程默认引脚）:
 *   TC264                 外设
 *   ─────────────────────────────────────────
 *   P14_0 (UART0_TX)  →  CH340 RXD
 *   P14_1 (UART0_RX)  ←  CH340 TXD
 *   P33_9 (PWM)       →  舵机信号线（50Hz）
 *   P02_4 (DIR_R1)    →  DRV8701E#1 DIR (右电机)
 *   P02_5 (PWM_R1)    →  DRV8701E#1 PWM (右电机, 17kHz)
 *   P21_2 (DIR_R2)    →  DRV8701E#2 DIR (左电机)
 *   P21_3 (PWM_R2)    →  DRV8701E#2 PWM (左电机, 17kHz)
 *   P33_7 (右编码器脉冲) ←  右电机编码器脉冲输出
 *   P33_6 (右编码器方向) ←  右电机编码器方向输出
  *   P10_3 (左编码器脉冲) ←  左电机编码器脉冲输出
  *   P10_1 (左编码器方向) ←  左电机编码器方向输出
 *   GND               ──  所有设备共地
 *
 * 集成方式:
 *   复制此文件到逐飞 TC264 工程的 user/ 目录，替换原有 cpu0_main.c
 *   同时确保 user/ 下的 isr_config.h 正确配置了中断优先级
 */

#include "zf_common_headfile.h"
#include "isr_config.h"
#include "zf_device_tft180.h"
#include "encoder_share.h"
#pragma section all "cpu0_dsram"

// ====================================================================
// 1. 引脚与硬件配置（匹配逐飞例程默认引脚）
// ====================================================================

/* ---- 舵机（50Hz 标准舵机）---- */
#define SERVO_PWM_CH        ATOM1_CH1_P33_9      // 舵机 PWM 通道

/* ---- 右电机 (DRV8701E #1) ---- */
#define MOTOR_R_DIR_PIN     P21_4                // 右电机方向
#define MOTOR_R_PWM_CH      ATOM0_CH3_P21_5      // 右电机速度 PWM

/* ---- 左电机 (DRV8701E #2) ---- */
#define MOTOR_L_DIR_PIN     P21_2                // 左电机方向
#define MOTOR_L_PWM_CH      ATOM0_CH1_P21_3      // 左电机速度 PWM

/* ---- 右轮编码器（方向/脉冲模式，TIM2）---- */
#define ENCODER_CH          TIM2_ENCODER
#define ENCODER_PULSE_PIN   TIM2_ENCODER_CH1_P33_7  // 右轮 脉冲信号
#define ENCODER_DIR_PIN     TIM2_ENCODER_CH2_P33_6  // 右轮 方向信号
/* ---- 左轮编码器（方向/脉冲模式，TIM5）---- */
#define ENCODER_CH_LEFT     TIM5_ENCODER
#define ENCODER_PULSE_PIN_LEFT  TIM5_ENCODER_CH1_P10_3  // 左轮 脉冲信号
#define ENCODER_DIR_PIN_LEFT    TIM5_ENCODER_CH2_P10_1  // 左轮 方向信号

// ====================================================================
// 2. 运行参数
// ====================================================================

#define SERVO_FREQ          300                  // 舵机频率 (Hz)
#define SERVO_PERIOD_US     (1000000u / SERVO_FREQ)  // 舵机PWM周期 (微秒)
#define SERVO_STEP_STRAIGHT_US   30               // 直道附近每10ms最大变化步长 (us)
#define SERVO_STEP_CURVE_US      80               // 弯道每10ms最大变化步长 (us)
#define SERVO_CURVE_THRESHOLD_US 40               // 偏离中值超过该值视为弯道
#define MOTOR_FREQ          17000                // 电机 PWM 频率 (Hz)
#define PWM_DUTY_MAX        10000                // 逐飞库 PWM 占空比最大值

#define PID_PERIOD_MS       10                   // PID 控制周期 (ms)
#define PID_PIT_CH          CCU60_CH0            // PID 定时器通道

#define UART_RX_FIFO_SIZE   64                   // 串口接收 FIFO 大小

// ====================================================================
// 3. 协议常量（与 car_control.py 严格一致）
// ====================================================================

#define FRAME_HEADER1       0xAA
#define FRAME_HEADER2       0x55
#define CMD_CONTROL         0x01
#define CMD_HEARTBEAT       0x02

#define SERVO_MIN_US        500
#define SERVO_MID_US        1500
#define SERVO_MAX_US        2500
#define STEERING_INPUT_MAX_US   80   // 上位机实际舵机打角范围，相对中值最大 ±80us

#define STEERING_DIFF_MAX_RATIO  0.15f   // 满舵时的最大差速比例，先给 30%，实际可调

#define MOTOR_CMD_MAX       3000                 // 上位机允许下发的电机指令上限
#define MOTOR_SPEED_MAX     1000                 // PID 输出/电机 PWM 满量程速度量纲

// ====================================================================
// 4. 安全保护参数
// ====================================================================

#define HEARTBEAT_TIMEOUT_MS    1000             // 心跳超时 → 停车
#define CONTROL_TIMEOUT_MS      500              // 控制指令超时 → 停车

// ====================================================================
// 5. 数据结构
// ====================================================================

/** UART 接收状态机 */
typedef enum {
    RX_WAIT_HEADER1,
    RX_WAIT_HEADER2,
    RX_WAIT_CMD,
    RX_WAIT_LEN,
    RX_WAIT_DATA,
    RX_WAIT_CHECKSUM
} rx_state_t;

/** PID 控制器 */
typedef struct {
    float kp, ki, kd;
    float integral;
    float prev_error;
    float integral_limit;
    float output_limit;
    float error_buf[5];   /* 误差滤波环形缓冲区（最近 5 次） */
    uint8_t buf_idx;       /* 环形缓冲区写入位置 (0~4) */
    uint8_t filled;        /* 缓冲区是否已填满（≥5 次采样） */
} pid_t;
/**
 * SPEED_DIVISOR —— 速度目标值除数
 *
 * 作用：
 *   Python 下发的速度值经过此除数缩放后再送入 PID。
 *   不改变 Python 侧的整型精度，所有 1~50 的档位在除后都是不同的值。
 *
 * 调整方法：
 *   SPEED_DIVISOR = 10  →  speed=1 → PID 目标=0.1，车极慢
 *   SPEED_DIVISOR = 5   →  speed=1 → PID 目标=0.2，慢速蠕动
 *   SPEED_DIVISOR = 1   →  speed=1 → PID 目标=1.0，恢复原始行为
 */
#define SPEED_DIVISOR       10
// ====================================================================
// 7. 全局状态变量
// ====================================================================

/* ---- 控制目标（由香橙派下发）---- */
int16  g_target_speed  = 0;               // 上位机下发的目标速度 [-3000, 3000]
static uint16 g_target_servo  = SERVO_MID_US;    // 目标舵机脉宽 [500, 2500] us

/* ---- 安全计时 ---- */
static uint32 g_last_heartbeat_ms = 0;
static uint32 g_last_control_ms  = 0;

/* ---- 编码器反馈 ---- */
int16  g_encoder_count_right   = 0;             // 本周期编码器增量
float  g_actual_speed_right    = 0.0f;          // 实测速度
int16  g_encoder_count_left  = 0;             // 左轮：本周期编码器增量
float  g_actual_speed_left   = 0.0f;          // 左轮：实测速度

/* ---- PID ---- */
static pid_t  g_speed_pid;             // 右轮 PID
static pid_t  g_speed_pid_left;        // 左轮 PID

/* ---- 舵机当前实际位置（斜坡限幅用）---- */
static uint16 g_current_servo_us = SERVO_MID_US;
static int16 g_pid_out_disp = 0 ;          // 右轮 PID 输出（TFT 调试显示）
static int16 g_pid_out_disp_left = 0 ;     // 左轮 PID 输出（TFT 调试显示）

/* ---- UART 接收上下文 ---- */
static rx_state_t g_rx_state  = RX_WAIT_HEADER1;
static uint8      g_rx_cmd    = 0;
static uint8      g_rx_len    = 0;
static uint8      g_rx_data[16];
static uint8      g_rx_idx    = 0;
static uint8      g_rx_checksum;
static uint8      g_rx_calc;

/* ---- FIFO ---- */
static uint8       uart_fifo_buf[UART_RX_FIFO_SIZE];
static fifo_struct uart_fifo;
// ====================================================================
// 8. PID 控制器
// ====================================================================

static void pid_init(pid_t *pid, float kp, float ki, float kd,
                     float i_limit, float o_limit)
{
    pid->kp = kp;
    pid->ki = ki;
    pid->kd = kd;
    pid->integral = 0.0f;
    pid->prev_error = 0.0f;
    pid->integral_limit = i_limit;
    pid->output_limit = o_limit;
    pid->buf_idx = 0;
    pid->filled = 0;

    for (int i = 0; i < 5; i++)
        pid->error_buf[i] = 0.0f;
}

static float pid_update(pid_t *pid, float target, float actual)
{
    float error = target - actual;

    /* --------------------------------------------------
     * 去极值平均滤波（滑动窗口）
     * 每次循环将当前 error 写入环形缓冲区，
     * 缓冲区填满后，每轮去掉 5 个值中的最大最小值，
     * 剩余 3 个取平均作为本次的 error。
     * -------------------------------------------------- */
   /* pid->error_buf[pid->buf_idx] = error;
    pid->buf_idx++;

    if (pid->buf_idx >= 5) {
        pid->buf_idx = 0;
        pid->filled = 1;
    }

    if (pid->filled) {
        float min_val = pid->error_buf[0];
        float max_val = pid->error_buf[0];
        float sum     = pid->error_buf[0];

        for (int i = 1; i < 5; i++) {
            float v = pid->error_buf[i];
            sum += v;
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
        }

        error = (sum - min_val - max_val) / 3.0f;
    }*/

    /* 积分抗饱和 */
    pid->integral += error;
    if (pid->integral > pid->integral_limit)
        pid->integral = pid->integral_limit;
    else if (pid->integral < -pid->integral_limit)
        pid->integral = -pid->integral_limit;

    /* 微分 */
    float derivative = error - pid->prev_error;
    pid->prev_error = error;

    /* PID 输出 */
    float output = pid->kp * error
                 + pid->ki * pid->integral
                 + pid->kd * derivative;

    /* 输出限幅 */
    if (output > pid->output_limit)
        output = pid->output_limit;
    else if (output < -pid->output_limit)
        output = -pid->output_limit;

    return output;
}

// ====================================================================
// 9. 底层驱动控制
// ====================================================================

/**
 * 设置舵机角度
 *
 * 换算: duty = (pwm_us / 周期_us) * PWM_DUTY_MAX
 * 周期 = 1000000 / SERVO_FREQ
 */
static void servo_set(uint16 pwm_us)
{
    /* 反转舵机方向：以中位对称翻转 */
    if (pwm_us >= SERVO_MID_US)
        pwm_us = SERVO_MID_US - (pwm_us - SERVO_MID_US);
    else
        pwm_us = SERVO_MID_US + (SERVO_MID_US - pwm_us);

    if (pwm_us < SERVO_MIN_US) pwm_us = SERVO_MIN_US;
    if (pwm_us > SERVO_MAX_US) pwm_us = SERVO_MAX_US;

    uint32 duty = (uint32)pwm_us * PWM_DUTY_MAX / SERVO_PERIOD_US;
    if (duty > PWM_DUTY_MAX) duty = PWM_DUTY_MAX;

    pwm_set_duty(SERVO_PWM_CH, duty);
}

/**
 * 设置单个电机速度（DRV8701E 方式）
 *
 * @param dir_pin   方向 GPIO
 * @param pwm_ch    PWM 通道
 * @param speed     速度 [-MOTOR_SPEED_MAX, MOTOR_SPEED_MAX]
 *                  正值=前进, 负值=后退
 */
static void motor_single_set(gpio_pin_enum dir_pin,
                             pwm_channel_enum pwm_ch,
                             int16 speed)
{
    uint16 duty;

    if (speed >= 0) {
        gpio_set_level(dir_pin, GPIO_HIGH);
        duty = (uint16)((uint32)speed * PWM_DUTY_MAX / MOTOR_SPEED_MAX);
    } else {
        gpio_set_level(dir_pin, GPIO_LOW);
        duty = (uint16)((uint32)(-speed) * PWM_DUTY_MAX / MOTOR_SPEED_MAX);
    }

    if (duty > PWM_DUTY_MAX) duty = PWM_DUTY_MAX;
    pwm_set_duty(pwm_ch, duty);
}

/**
 * 紧急停车
 */
static void emergency_stop(void)
{
    servo_set(SERVO_MID_US);
    g_current_servo_us = SERVO_MID_US;
    motor_single_set(MOTOR_R_DIR_PIN, MOTOR_R_PWM_CH, 0);
    motor_single_set(MOTOR_L_DIR_PIN, MOTOR_L_PWM_CH, 0);
    g_target_speed = 0;
    g_actual_speed_right = 0.0f;
    g_actual_speed_left = 0.0f;
    g_speed_pid.integral = 0.0f;
    g_speed_pid.prev_error = 0.0f;
    g_speed_pid_left.integral = 0.0f;
    g_speed_pid_left.prev_error = 0.0f;
}

// ====================================================================
// 10. UART 协议解析
// ====================================================================

static void protocol_parse_byte(uint8 byte)
{
    switch (g_rx_state) {

    case RX_WAIT_HEADER1:
        if (byte == FRAME_HEADER1) {
            g_rx_calc = 0;
            g_rx_state = RX_WAIT_HEADER2;
        }
        break;

    case RX_WAIT_HEADER2:
        if (byte == FRAME_HEADER2) {
            g_rx_state = RX_WAIT_CMD;
        } else if (byte != FRAME_HEADER1) {
            g_rx_state = RX_WAIT_HEADER1;
        }
        break;

    case RX_WAIT_CMD:
        g_rx_cmd = byte;
        g_rx_calc ^= byte;
        g_rx_state = RX_WAIT_LEN;
        break;

    case RX_WAIT_LEN:
        g_rx_len = byte;
        g_rx_calc ^= byte;
        g_rx_idx = 0;
        if (g_rx_len == 0) {
            g_rx_state = RX_WAIT_CHECKSUM;
        } else if (g_rx_len <= sizeof(g_rx_data)) {
            g_rx_state = RX_WAIT_DATA;
        } else {
            g_rx_state = RX_WAIT_HEADER1;
        }
        break;

    case RX_WAIT_DATA:
        g_rx_data[g_rx_idx] = byte;
        g_rx_calc ^= byte;
        g_rx_idx++;
        if (g_rx_idx >= g_rx_len) {
            g_rx_state = RX_WAIT_CHECKSUM;
        }
        break;

    case RX_WAIT_CHECKSUM:
        g_rx_checksum = byte;
        if (g_rx_calc == g_rx_checksum) {
            /* 校验通过 */
            uint32 now = system_getval_ms();

            switch (g_rx_cmd) {
            case CMD_CONTROL:
                if (g_rx_len == 4) {
                    uint16 servo = ((uint16)g_rx_data[0] << 8) | g_rx_data[1];
                    int16  motor = (int16)(((uint16)g_rx_data[2] << 8) | g_rx_data[3]);

                    if (servo < SERVO_MIN_US) servo = SERVO_MIN_US;
                    if (servo > SERVO_MAX_US) servo = SERVO_MAX_US;
                    if (motor > MOTOR_CMD_MAX)  motor = MOTOR_CMD_MAX;
                    if (motor < -MOTOR_CMD_MAX) motor = -MOTOR_CMD_MAX;

                    g_target_servo = servo;
                    g_target_speed = motor;

                    /* 舵机不再立即响应，由 PID 中断中的斜坡限幅平滑过渡 */
                    g_last_control_ms = now;
                }
                break;

            case CMD_HEARTBEAT:
                break;

            default:
                break;
            }

            g_last_heartbeat_ms = now;
        }
        g_rx_state = RX_WAIT_HEADER1;
        break;
    }
}

// ====================================================================
// 11. 中断回调函数（由 isr.c 中的 ISR 调用）
// ====================================================================

/**
 * UART0 接收中断回调
 *
 * 由 isr.c 的 uart0_rx_isr 调用。
 * 从 UART 读取一个字节写入 FIFO，主循环再从 FIFO 取出解析。
 */
void car_uart_rx_callback(void)
{
    uint8 byte;
    if (uart_query_byte(UART_3, &byte)) {
        fifo_write_buffer(&uart_fifo, &byte, 1);
    }
}

/**
 * PID 定时中断回调（每 10ms）
 *
 * 由 isr.c 的 cc60_pit_ch0_isr 调用。
 * 读取编码器 → 安全检查 → PID 计算 → 双电机差速输出。
 */
void car_pid_timer_callback(void)
{
    pit_clear_flag(PID_PIT_CH);

    /* ---- 1. 读取编码器 ---- */
    g_encoder_count_right = encoder_get_count(ENCODER_CH);
    encoder_clear_count(ENCODER_CH);
    /* ---- 1b. 读取左轮编码器 ---- */
    g_encoder_count_left = -encoder_get_count(ENCODER_CH_LEFT);   // 编码器方向反接，取反修正
    encoder_clear_count(ENCODER_CH_LEFT);

    /*
     * 编码器增量 → 速度映射
     *
     * 方向/脉冲模式下，TC264 硬件根据 DIR 引脚电平自动判断方向:
     *   - DIR=HIGH → 计数器递增（正转）
     *   - DIR=LOW  → 计数器递减（反转）
     * 因此 encoder_get_count() 返回的是有符号值。
     *
     * g_actual_speed_right 单位: 编码器脉冲数 / PID_PERIOD_MS 毫秒
     * PID 系数需根据实际轮径、编码器线数、减速比标定。
     */
    g_actual_speed_right = (float)g_encoder_count_right;
    g_actual_speed_left = (float)g_encoder_count_left;

    /* ---- 2. 安全检查 ---- */
    uint32 now = system_getval_ms();

    if ((now - g_last_heartbeat_ms) > HEARTBEAT_TIMEOUT_MS) {
        emergency_stop();
        return;
    }

    if ((now - g_last_control_ms) > CONTROL_TIMEOUT_MS) {
        //控制指令超时: 速度平滑归零，舵机回中
        if (g_target_speed > 0) {
            g_target_speed -= 50;
            if (g_target_speed < 0) g_target_speed = 0;
        } else if (g_target_speed < 0) {
            g_target_speed += 50;
            if (g_target_speed > 0) g_target_speed = 0;
        }
    }
    if (g_target_speed == 0) {
        g_speed_pid.integral = 0.0f;
        g_speed_pid_left.integral = 0.0f;
    }

    /* ---- 3. 双电机独立 PID 计算 ---- */
    /* 目标速度经除数缩放后再送入 PID，以达到调速目的 */
    float pid_target = (float)g_target_speed / SPEED_DIVISOR;

    /* 根据舵机打角计算左右轮差速 */
    /* 正值=右转，右轮减速、左轮加速；负值=左转，反之 */
    float steer_diff = ((float)g_target_servo - (float)SERVO_MID_US)
                     / (float)(STEERING_INPUT_MAX_US);
    float diff_ratio = steer_diff * STEERING_DIFF_MAX_RATIO;

    if (diff_ratio >  STEERING_DIFF_MAX_RATIO)
        diff_ratio =  STEERING_DIFF_MAX_RATIO;
    if (diff_ratio < -STEERING_DIFF_MAX_RATIO)
        diff_ratio = -STEERING_DIFF_MAX_RATIO;

    float pid_target_r = pid_target * (1.0f - diff_ratio);
    float pid_target_l = pid_target * (1.0f + diff_ratio);

    /* 右轮 PID */
    float pid_output_r = pid_update(&g_speed_pid,
                                    pid_target_r,
                                    g_actual_speed_right);
    g_pid_out_disp = (int16)pid_output_r;

    /* 左轮 PID */
    float pid_output_l = pid_update(&g_speed_pid_left,
                                    pid_target_l,
                                    g_actual_speed_left);
    g_pid_out_disp_left = (int16)pid_output_l;

    /* ---- 4. 双电机独立输出 ---- */
    motor_single_set(MOTOR_R_DIR_PIN, MOTOR_R_PWM_CH, (int16)pid_output_r);//pid_output_r
    motor_single_set(MOTOR_L_DIR_PIN, MOTOR_L_PWM_CH, (int16)pid_output_l);//pid_output_l

    /* ---- 5. 舵机斜坡限幅 ---- */
    /* 每 10ms 向目标角度逼近一步，避免舵机瞬间跳变 */
    int16 steer_error = (int16)g_target_servo - (int16)SERVO_MID_US;

    uint16 servo_step;
    if (steer_error > SERVO_CURVE_THRESHOLD_US ||
        steer_error < -SERVO_CURVE_THRESHOLD_US) {
        servo_step = SERVO_STEP_CURVE_US;
    } else {
        servo_step = SERVO_STEP_STRAIGHT_US;
    }

    if (g_current_servo_us < g_target_servo) {
        g_current_servo_us += servo_step;
        if (g_current_servo_us > g_target_servo)
            g_current_servo_us = g_target_servo;
    } else if (g_current_servo_us > g_target_servo) {
        g_current_servo_us -= servo_step;
        if (g_current_servo_us < g_target_servo)
            g_current_servo_us = g_target_servo;
    }
    servo_set(g_current_servo_us);
}

// ====================================================================
// 12. 外设初始化
// ====================================================================

static void hardware_init(void)
{
    clock_init();
    debug_init();       // 初始化 debug UART + printf

    printf("========================================\n");
    printf("  TC264 智能车控制固件\n");
    printf("========================================\n");

    /* ---- FIFO ---- */
    fifo_init(&uart_fifo, FIFO_DATA_8BIT, uart_fifo_buf, UART_RX_FIFO_SIZE);

    /* ---- UART3 (P15_6/P15_7, 115200, 接收中断) ---- */
    uart_init(UART_3, 115200, UART3_TX_P15_6, UART3_RX_P15_7);
    uart_rx_interrupt(UART_3, 1);
    printf("  UART:   115200 8N1 (P15_6/P15_7)\n");

    /* ---- 舵机 PWM (P33_9, 50Hz) ---- */
    pwm_init(SERVO_PWM_CH, SERVO_FREQ, 0);
    printf("  舵机:    %dHz (P33_9)\n", SERVO_FREQ);

    /* ---- 右电机 (DRV8701E, 17kHz) ---- */
    gpio_init(MOTOR_R_DIR_PIN, GPO, GPIO_LOW, GPO_PUSH_PULL);
    pwm_init(MOTOR_R_PWM_CH, MOTOR_FREQ, 0);

    /* ---- 左电机 (DRV8701E, 17kHz) ---- */
    gpio_init(MOTOR_L_DIR_PIN, GPO, GPIO_LOW, GPO_PUSH_PULL);
    pwm_init(MOTOR_L_PWM_CH, MOTOR_FREQ, 0);
    printf("  电机:    %dHz R(P02_4/P02_5) L(P21_2/P21_3)\n", MOTOR_FREQ);

    /* ---- 右轮编码器（方向/脉冲模式，TIM2）---- */
    encoder_dir_init(ENCODER_CH, ENCODER_PULSE_PIN, ENCODER_DIR_PIN);
    printf("  右编码器: 方向/脉冲模式 TIM2 (P33_7/P33_6)\n");
    /* ---- 左轮编码器（方向/脉冲模式，TIM5）---- */
    encoder_dir_init(ENCODER_CH_LEFT, ENCODER_PULSE_PIN_LEFT, ENCODER_DIR_PIN_LEFT);
    printf("  左编码器: 方向/脉冲模式 TIM5 (P10_3/P10_1) ");

    /* ---- PID ---- */
    /* 右轮 PID */
    pid_init(&g_speed_pid,
             2.6f,                     // kp3
             0.007f,                     // Ki0.15
             0.0f,                     // Kd1
             400.0f,                   // 积分限幅
             (float)MOTOR_SPEED_MAX);  // 输出限幅
    /* 左轮 PID（与右轮相同参数） */
    pid_init(&g_speed_pid_left,
             2.6f,                     // Kp
             0.007f,                     // Ki
             0.0f,                     // Kd
             400.0f,                   // 积分限幅
             (float)MOTOR_SPEED_MAX);  // 输出限幅
    printf(" PID: Kp=6.0 Ki=0.3 Kd=2.0 (双轮独立闭环)\n");

    /* ---- PID 定时中断 (10ms) ---- */
    pit_ms_init(PID_PIT_CH, PID_PERIOD_MS);
    printf("  控制周期: %dms\n", PID_PERIOD_MS);
    /* ---- TFT 1.8寸初始化 ---- */
        tft180_init();
        tft180_set_dir(TFT180_PORTAIT);
        tft180_set_font(TFT180_8X16_FONT);
        tft180_set_color(RGB565_BLUE, RGB565_WHITE);
        tft180_clear();
        tft180_show_string(0, 0,  "ENC_L:");
        tft180_show_string(0, 16, "SPD_L:");
        tft180_show_string(0, 32, "ENC_R:");
        tft180_show_string(0, 48, "SPD_R:");
        tft180_show_string(0, 64, "TGT:");
        tft180_show_string(0, 80, "OUT_R:");
        tft180_show_string(0, 96, "OUT_L:");
        printf("  显示:   TFT 1.8寸\n");
    printf("========================================\n");
    printf("  等待香橙派连接...\n");
}

// ====================================================================
// 13. 主函数
// ====================================================================

int core0_main(void)
{
    hardware_init();
    emergency_stop();
    cpu_wait_event_ready();

    while (TRUE) {
        /* 从 FIFO 读取串口数据，逐字节解析 */
        uint32 fifo_count = fifo_used(&uart_fifo);
        if (fifo_count > 0) {
            uint8 buf[32];
            uint32 read_count = fifo_count;
            if (read_count > sizeof(buf)) read_count = sizeof(buf);

            fifo_read_buffer(&uart_fifo, buf, &read_count, FIFO_READ_AND_CLEAN);

            for (uint32 i = 0; i < read_count; i++) {
                protocol_parse_byte(buf[i]);
            }
        }
        /* TFT显示更新（每50ms刷新一次） */
                {
                    static uint32 last_tft_ms = 0;
                    uint32 now_tft = system_getval_ms();
                    if (now_tft - last_tft_ms >= 50) {
                        last_tft_ms = now_tft;
                        tft180_show_int(48, 0, g_encoder_count_left, 6);
                        tft180_show_float(48, 16, g_actual_speed_left, 6, 2);
                        tft180_show_int(48, 32, g_encoder_count_right, 6);
                        tft180_show_float(48, 48, g_actual_speed_right, 6, 2);
                        tft180_show_int(48, 64, g_target_speed, 6);
                        tft180_show_int(48, 80, g_pid_out_disp, 6);
                        tft180_show_int(48, 96, g_pid_out_disp_left, 6);
                    }
                }
        system_delay_ms(1);
    }
}

#pragma section all restore
