"""
动态提升机位置配置模块
根据LIFT_COUNT自动均匀分布在仓库范围内
"""

def calculate_lift_locations(lift_count, cols, row=1):
    """
    根据提升机数量动态计算均匀分布的位置
    
    参数:
        lift_count: 提升机数量
        cols: 仓库列数
        row: 提升机所在行（默认第1行-地面层）
    
    返回:
        dict: {'Lift01': (row, col), ...}
    """
    if lift_count <= 0:
        return {}
    
    locations = {}
    
    # 均匀分布算法：将列数分成 lift_count+1 段，取中间点
    # 例如：31列，4台Lift -> 位置在 6, 12, 19, 25 (大致)
    for i in range(lift_count):
        # 计算列位置 (1-indexed)
        # 间隔 = cols / (lift_count + 1)
        # 位置 = 间隔 * (i + 1)
        col = int((i + 1) * cols / (lift_count + 1))
        
        # 确保不超出范围且至少为1
        col = max(1, min(col, cols))
        
        lift_id = f'Lift{i+1:02d}'
        locations[lift_id] = (row, col)
    
    return locations


def get_lift_locations(lift_count=None, cols=None):
    """
    获取提升机位置配置
    
    如果lift_count和cols为None，则使用config.py中的默认值
    """
    if lift_count is None or cols is None:
        # 导入默认配置
        from src.config import LIFT_COUNT, COLS, LIFT_LOCATIONS
        if lift_count is None:
            lift_count = LIFT_COUNT
        if cols is None:
            cols = COLS
        
        # 如果请求的数量与默认配置一致，返回默认配置
        if lift_count == LIFT_COUNT:
            # 过滤出有效的提升机位置
            return {k: v for k, v in LIFT_LOCATIONS.items() 
                   if int(k[4:6]) <= lift_count and v[1] <= cols}
    
    # 动态计算
    return calculate_lift_locations(lift_count, cols)


def print_lift_distribution(lift_count, cols):
    """打印提升机分布情况"""
    locations = calculate_lift_locations(lift_count, cols)
    
    print(f"\n提升机分布配置 (数量={lift_count}, 仓库列数={cols})")
    print("-" * 50)
    
    # 可视化显示
    grid = ['_'] * (cols + 1)  # 0-index忽略，从1开始
    
    for lift_id, (row, col) in locations.items():
        grid[col] = lift_id[4]  # 取数字部分
        print(f"{lift_id}: 第{row}排, 第{col}列")
    
    print("\n位置可视化 (L=提升机位置):")
    print("Col: ", end="")
    for i in range(1, min(cols + 1, 32)):  # 只显示前31列
        print(f"{i%10}", end="")
    print()
    print("     ", end="")
    for i in range(1, min(cols + 1, 32)):
        if grid[i] == '_':
            print("·", end="")
        else:
            print("L", end="")
    print()
    
    # 计算间隔
    cols_list = sorted([v[1] for v in locations.values()])
    if len(cols_list) > 1:
        intervals = [cols_list[i+1] - cols_list[i] for i in range(len(cols_list)-1)]
        print(f"\n列间隔: {intervals}")
        print(f"平均间隔: {sum(intervals)/len(intervals):.1f}列")
    
    return locations


# 测试不同配置
if __name__ == "__main__":
    print("=" * 60)
    print("动态提升机位置配置测试")
    print("=" * 60)
    
    test_cases = [
        (4, 31),
        (6, 31),
        (8, 48),
        (10, 50),
    ]
    
    for lift_count, cols in test_cases:
        print_lift_distribution(lift_count, cols)
        print("\n" + "=" * 60)
