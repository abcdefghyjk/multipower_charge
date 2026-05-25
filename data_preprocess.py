import pandas as pd
import numpy as np
from config import *


def read_vehicle_trips(vehicle_excel_path):
    """读取车辆运营计划（每个sheet对应1辆车）
    Excel列假设（需与用户文件匹配，否则需调整列名）：
    车次序号, 车次开始时间(分钟), 车次结束时间(分钟), 车次耗电量(%), 耗电波动(%),
    起点到充电站1用时(分钟), 终点到充电站1用时(分钟), 起点到充电站2用时(分钟), 终点到充电站2用时(分钟)
    """
    vehicle_trips = {}
    excel_file = pd.ExcelFile(vehicle_excel_path)

    for sheet_name in excel_file.sheet_names:
        try:
            df = pd.read_excel(vehicle_excel_path, sheet_name=sheet_name)
            # 提取车辆ID（假设sheet名为“车辆1”“车辆2”...）
            vehicle_id = int(sheet_name.replace("车辆", ""))
            trips = []

            for _, row in df.iterrows():
                # 读取单车次信息
                trip = {
                    "trip_idx": int(row["车次序号"]),
                    "t_start": float(row["车次开始时间"]),
                    "t_end": float(row["车次结束时间"]),
                    "discharge_mean": float(row["车次耗电量"]),
                    "discharge_delta": float(row["耗电波动"]),
                    # 空驶时间（终点到充电站：充电前；起点到充电站：预留，暂用终点）
                    "drive_time_to_s1": float(row["到充电站1用时"]),
                    "drive_time_to_s2": float(row["到充电站2用时"]),
                    # 充电后到下一站起点的空驶时间（假设与“起点到充电站”相同）
                    "drive_time_from_s1": float(row["到充电站1用时"]),
                    "drive_time_from_s2": float(row["到充电站2用时"])
                }
                trips.append(trip)

            vehicle_trips[vehicle_id] = trips
            if VERBOSE:
                print(f"✅ 读取车辆{vehicle_id}：{len(trips)}个车次")

        except KeyError as e:
            raise ValueError(f"❌ Excel列名不匹配，缺失列：{e}\n请确保列名与代码注释一致")
        except Exception as e:
            print(f"⚠️ 读取sheet {sheet_name} 失败：{e}")

    if not vehicle_trips:
        raise ValueError("❌ 未读取到任何车辆数据，请检查Excel文件路径和格式")
    return vehicle_trips



def read_wear_data(wear_excel_path):
    """读取电池损耗表（slow/fast两个sheet）"""
    try:
        wear_dict = {}
        excel_file = pd.ExcelFile(wear_excel_path)
        for mode in ["slow", "fast"]:
            if mode not in excel_file.sheet_names:
                raise ValueError(f"❌ 损耗表缺失sheet：{mode}")
            df = pd.read_excel(wear_excel_path, sheet_name=mode)
            wear_dict[mode] = df.iloc[:, 1:].to_numpy()  # 第1列是SOC，后续是损耗
        return wear_dict
    except Exception as e:
        raise ValueError(f"❌ 读取电池损耗表失败：{e}")


def read_tou_price(tou_excel_path):
    """
    读取分时电价 TOU3.xlsx（修正版）
    格式：3列 → 开始时间步 | 结束时间步 | 电价(元/度)
    时间步范围：1~400（对应0~399数组索引）
    """
    try:
        df = pd.read_excel(tou_excel_path)
        tou_array = np.zeros(TOTAL_TIME_STEPS)  # TOTAL_TIME_STEPS=400，索引0~399

        for _, row in df.iterrows():
            # 关键修正：TOU3表中的数值是时间步编号，不是分钟！
            start_step = int(row["开始时间"])
            end_step = int(row["结束时间"])
            price = float(row["电价"])

            # 转换为Python数组的0索引
            t_start = start_step - 1
            t_end = end_step - 1

            # 边界保护
            t_start = max(0, min(t_start, TOTAL_TIME_STEPS - 1))
            t_end = max(0, min(t_end, TOTAL_TIME_STEPS - 1))

            # 赋值电价（包含两端）
            tou_array[t_start:t_end + 1] = price

        # 验证电价是否全部赋值
        if np.any(tou_array == 0):
            zero_steps = np.where(tou_array == 0)[0]
            print(f"⚠️ 警告：以下时间步电价为0：{zero_steps}")

        return tou_array
    except Exception as e:
        raise ValueError(f"❌ 读取分时电价失败：{e}")

def read_nonlinear_data(nonlinear_excel_path):
    """读取非线性充电时间表（slow/fast两个sheet）"""
    try:
        nonlinear_dict = {}
        excel_file = pd.ExcelFile(nonlinear_excel_path)
        for mode in ["slow", "fast"]:
            if mode not in excel_file.sheet_names:
                raise ValueError(f"❌ 充电表缺失sheet：{mode}")
            df = pd.read_excel(nonlinear_excel_path, sheet_name=mode)
            nonlinear_dict[mode] = df.iloc[:, 1:].to_numpy()  # 第1列是SOC，后续是时间
        return nonlinear_dict
    except Exception as e:
        raise ValueError(f"❌ 读取非线性充电表失败：{e}")


def compute_drive_params(vehicle_trips):
    """计算空驶参数（基于Excel中的用时）"""
    drive_data = {}  # key: (v, k, s, mode)，mode=0(去)/1(回)；或(v,k,'direct')
    for vehicle_id, trips in vehicle_trips.items():
        for k, trip in enumerate(trips):
            # 1. 不充电：直连空驶(设为第一次空驶的时间）
            direct_time = trip["drive_time_to_s1"]
            drive_data[(vehicle_id, k, "direct")] = {
                "time": direct_time,
                "soc_consume": direct_time * SOC_CONSUME_PER_MIN,
                "cost": direct_time * COST_PER_MIN
            }

            # 2. 充电：往返充电站空驶（s=1,2）
            for s in [1, 2]:
                # 去充电站（终点→充电站s）
                drive_time_to = trip[f"drive_time_to_s{s}"]
                # 回下一站起点（充电站s→下一站起点）
                drive_time_from = trip[f"drive_time_from_s{s}"]
                total_time = drive_time_to + drive_time_from

                drive_data[(vehicle_id, k, s, "to")] = {
                    "time": drive_time_to,
                    "soc_consume": drive_time_to * SOC_CONSUME_PER_MIN,
                    "cost": drive_time_to * COST_PER_MIN
                }
                drive_data[(vehicle_id, k, s, "from")] = {
                    "time": drive_time_from,
                    "soc_consume": drive_time_from * SOC_CONSUME_PER_MIN,
                    "cost": drive_time_from * COST_PER_MIN
                }
                drive_data[(vehicle_id, k, s, "total")] = {
                    "time": total_time,
                    "soc_consume": total_time * SOC_CONSUME_PER_MIN,
                    "cost": total_time * COST_PER_MIN
                }
    return drive_data


def precompute_elec_cumsum(tou_array):
    """预计算电费前缀和（完全匹配用户规则：每时间步电费=单位电价×5分钟）
    :param tou_array: 分时电价数组（单位：元/度）
    :return: elec_cumsum: 前缀和数组，elec_cumsum[t] = 前t步总电费
    """
    # 关键修正：每步电费 = 单位电价 × 5分钟（用户明确规则，无需转小时）
    elec_cost_per_step = tou_array * T_STEP  # T_STEP=5，直接相乘
    # 计算前缀和（索引0=0，索引1=第1步电费，索引t=前t步总电费）
    elec_cumsum = np.cumsum(elec_cost_per_step)
    elec_cumsum = np.insert(elec_cumsum, 0, 0)  # 前缀和数组开头补0，方便计算[t_start, t_end)的和
    return elec_cumsum
def get_uniform_distribution(mean, delta, step):
    """生成均匀分布的放电值（全局公共函数，对齐单车辆DP）"""
    min_val = max(mean - delta, 0)
    max_val = mean + delta
    values = np.arange(min_val, max_val + step, step)
    return {round(val, 1): 1.0 for val in values}