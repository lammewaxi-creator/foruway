"""
config.py - 配置中心
职责：存储所有静态常量，是整个系统的"真理来源"
任何模块需要参数都从这里读，杜绝硬编码
"""

# ===== 系统物理参数 (基于宇峰项目文档) =====
# 1. 设备数量
LIFT_COUNT = 6       # 提升机数量
RGV_COUNT = 15       # 四向车数量

# 2. 运动学参数 (m/s, m/s^2)
# 穿梭车
RGV_V_MAX = 1.5      # 四向车最大速度
RGV_ACC = 2.0        # 四向车加速度

# 提升机
LIFT_V_EMPTY = 0.75  # 提升机空载速度 (45m/min)
LIFT_V_LOADED = 0.6  # 提升机负载速度 (36m/min)
LIFT_ACC = 1.5       # 提升机加速度

# 3. 提升机物理坐标映射
# 格式: {'Lift01': (Row, Col), ...}
# Row: 行号 (X方向), Col: 列号 (Y方向)
LIFT_LOCATIONS = {
    'Lift01': (1, 3),   # 第1排, 第3列
    'Lift02': (1, 8),   # 第1排, 第8列
    'Lift03': (1, 13),  # 第1排, 第13列
    'Lift04': (1, 18),  # 第1排, 第18列
    'Lift05': (1, 23),  # 第1排, 第23列
    'Lift06': (1, 28),  # 第1排, 第28列
    'Lift07': (1, 33),  # 第1排, 第33列 (扩展)
    'Lift08': (1, 38),  # 第1排, 第38列 (扩展)
    'Lift09': (1, 43),  # 第1排, 第43列 (扩展)
    'Lift10': (1, 48)   # 第1排, 第48列 (扩展)
}
IO_PORTS = LIFT_LOCATIONS.copy()

# 4. 其它时间参数
T_EXIT = 5.0         # 穿梭车驶出电梯并释放电梯的解耦时间 (秒)
T_LOAD_UNLOAD = 3.0  # 模拟取放货时间 (秒)

# 5. 仓库几何尺寸 (米)
LAYER_HEIGHT = 0.8   # 层高
CELL_LENGTH = 0.4    # 储位长度 (X方向)
CELL_WIDTH = 0.5     # 储位宽度 (Y方向)
# 假设布局: 44行 x 31列
ROWS = 44
COLS = 31

# ===== NSGA-II算法参数 =====
POP_SIZE = 100       # 种群大小
N_GEN = 100          # 迭代次数
P_CROSSOVER = 0.8    # 交叉概率
P_MUTATION = 0.1     # 变异概率

# ===== 数据路径配置 =====
DATA_PATH_R1 = 'R1_medium_arrival_rate.csv'
DATA_PATH_R2 = 'R2_high_arrival_rate.csv'
DATA_PATH_R3 = 'R3_full_load.csv'
RESULT_DIR = 'results'

# ===== 设备编码解析规则 =====
# 2开头的4位数字：提升机位置编码 (2 + Lift编号 + 层号相关)
# 3开头的4位数字：传送带设备编码（需过滤）

def parse_lift_device_code(code_str):
    """
    解析2开头的提升机设备编码为坐标
    
    编码规则分析：
    - 2018-2023: Lift02 相关位置
    - 2030-2031: Lift03 相关位置
    
    简化为：使用Lift02作为默认映射（因为大部分出库任务使用Lift02）
    """
    if not isinstance(code_str, str):
        return None
    
    code_str = code_str.strip()
    
    # 检查是否是2开头的4位数字
    if len(code_str) == 4 and code_str.startswith('2') and code_str[1:].isdigit():
        # 根据第二位判断提升机
        second_digit = int(code_str[1])
        
        # 映射到对应的提升机
        if second_digit == 0:
            lift_id = 'Lift02'  # 2018, 2019, 2020, 2022, 2023
        elif second_digit == 3:
            lift_id = 'Lift03'  # 2030, 2031
        else:
            lift_id = 'Lift02'  # 默认
        
        # 获取提升机位置
        if lift_id in LIFT_LOCATIONS:
            row, col = LIFT_LOCATIONS[lift_id]
            # 返回第1层的坐标（地面层）
            return f"{col:02d}-{row:03d}-01"
    
    return code_str

def is_convery_device(code_str):
    """检查是否是传送带设备（3开头）"""
    if not isinstance(code_str, str):
        return False
    code_str = code_str.strip()
    return len(code_str) == 4 and code_str.startswith('3') and code_str[1:].isdigit()
