# ====================== 基础约束参数 =======================
SOC_MIN = 20.0          # 最小SOC（%）
SOC_MAX = 100.0         # 最大SOC（%）
SOC_STEP = 2.0          # SOC离散步长（%）
T_STEP = 5.0            # 时间离散步长（分钟）
TOTAL_TIME_STEPS = 400  # 总时间步
LOCAL_CONSTRAINT_RATIO = 0.8  # z值约束系数
CACHE_SIZE = 10000      # 缓存大小

# ====================== TOU电价最大时间限制（用户TOU3最后一个时间步是400） =======================
MAX_TOU_TIME_STEP = 400
MAX_TOU_MINUTE = MAX_TOU_TIME_STEP * T_STEP  # 400×5=2000分钟

# ====================== 空驶模型（用户设定） =======================
SOC_CONSUME_PER_MIN = 0.2  # 空驶耗电量（%/分钟）
COST_PER_MIN = 1.0         # 空驶成本（元/分钟）
STATION_NUM = 2            # 充电站数量（用户设定2个）
PILE_PER_STATION = {"slow": 1, "fast": 1}  # 每站桩数（1快1慢）

# ====================== 列生成参数 =======================
EPS = 1e-6             # 收敛精度（检验数≥-EPS即收敛）
MAX_ITER = 50          # 最大迭代次数
VERBOSE = True         # 打印日志开关