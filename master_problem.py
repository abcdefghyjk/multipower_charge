import gurobipy as gp
from gurobipy import GRB
import numpy as np
from config import *


def init_initial_columns(vehicle_trips, drive_data, tou_array, wear_dict, nonlinear_dict):
    """
    生成满足桩容量约束的初始列（对齐单车辆DP期望计算逻辑）
    强制规则（仅用于初始列快速生成可行解）：
    - 充电站：车辆1、3 → 充电站1；车辆2 → 充电站2
    - 充电模式：车辆1、2 → 慢充；车辆3 → 快充
    成本计算：完全对齐单车辆DP的期望逻辑，引入车次演进的生存概率缩放
    """
    from subproblem_dp import init_subproblem_cache, cached_wear_cost, cached_charge_time
    from data_preprocess import get_uniform_distribution  # 导入公共函数

    init_subproblem_cache(wear_dict, nonlinear_dict)
    initial_columns = {v: [] for v in vehicle_trips.keys()}
    vehicle_ids = sorted(vehicle_trips.keys())

    # ==============================================
    # 强制配置表（仅初始列使用）
    # ==============================================
    VEHICLE_FORCED_STATION = {
        1: 1,  # 车辆1 → 充电站1
        2: 2,  # 车辆2 → 充电站2
        3: 1  # 车辆3 → 充电站1
    }
    VEHICLE_FORCED_MODE = {
        1: 0,  # 车辆1 → 慢充
        2: 0,  # 车辆2 → 慢充
        3: 1  # 车辆3 → 快充
    }

    # ---------------------- 策略1：非末班车充到90%，末班车充到100%（优先） ----------------------
    if VERBOSE:
        print("\n🚀 生成初始策略1：非末班车充到90%，末班车强制充到100%")
        print("📌 强制规则：")
        print("   充电站：车辆1/3→站1，车辆2→站2")
        print("   充电模式：车辆1/2→慢充，车辆3→快充")
        print("📌 成本计算：采用随机期望逻辑（对齐单车辆DP）")

    pile_usage_strategy1 = np.zeros((STATION_NUM, 2, TOTAL_TIME_STEPS), dtype=int)
    strategy1_schemes = {}

    for v in vehicle_ids:
        trips = vehicle_trips[v]
        trip_num = len(trips)
        current_soc_mean = SOC_MAX  # 平均SOC（用于决策）
        total_expected_cost = 0.0
        current_survival_prob = 1.0  # 初始化状态存活（到达）概率
        a_smt = np.zeros((STATION_NUM, 2, TOTAL_TIME_STEPS))
        scheme_detail = []
        feasible = True

        # 获取当前车辆的强制配置
        forced_station = VEHICLE_FORCED_STATION.get(v, 1)
        forced_mode = VEHICLE_FORCED_MODE.get(v, 0)
        station_name = f"充电站{forced_station}"
        mode_name = "慢充" if forced_mode == 0 else "快充"

        try:
            for k in range(trip_num):
                trip = trips[k]
                # ==============================================
                # 1. 计算本次放电的期望损耗（对齐单车辆DP + z值约束过滤）
                # ==============================================
                discharge_mean = trip["discharge_mean"]
                discharge_delta = trip["discharge_delta"]
                discharge_vals = get_uniform_distribution(discharge_mean, discharge_delta, SOC_STEP)
                discharge_list = list(discharge_vals.keys())
                single_prob = 1.0 / len(discharge_list) if discharge_list else 0.0
                d_to_z = {d: (d - discharge_mean) / discharge_delta if discharge_delta != 0 else 0.0 for d in
                          discharge_list}

                # 过滤满足z值约束的放电值
                valid_discharge = []
                for d in discharge_list:
                    z = d_to_z[d]
                    if abs(z) <= LOCAL_CONSTRAINT_RATIO * (k + 1) + EPS:
                        valid_discharge.append(d)

                valid_prob = len(valid_discharge) * single_prob
                expected_discharge_wear = 0.0
                for d in valid_discharge:
                    soc_after = max(current_soc_mean - d, SOC_MIN)
                    soc_after = round(soc_after, 1)
                    wear = cached_wear_cost(0, current_soc_mean, soc_after)
                    # 关键修改：放电损耗期望 = 损耗 * 分量概率 * 当前车次生存概率
                    expected_discharge_wear += wear * single_prob * current_survival_prob

                total_expected_cost += expected_discharge_wear

                # 暂存当前阶段前序生存概率，用于随后的充电状态推进
                old_survival_prob = current_survival_prob
                current_survival_prob *= valid_prob  # 推进生存概率到当前车次结束后

                # 记录放电
                scheme_detail.append({
                    "stage": k,
                    "type": "discharge",
                    "discharge_energy": discharge_mean,
                    "valid_discharge_count": len(valid_discharge),
                    "total_discharge_count": len(discharge_list),
                    "soc_before": current_soc_mean,
                    "soc_after": round(current_soc_mean - discharge_mean, 1),
                    "wear_cost": round(expected_discharge_wear, 4)
                })

                # 更新平均SOC
                current_soc_mean = round(current_soc_mean - discharge_mean, 1)

                # ==============================================
                # 2. 必须充电决策模拟
                # ==============================================
                charge_success = False
                # 充电目标降级：90% → 80% → 70% → 60%
                for target_soc in [90.0, 80.0, 70.0, 60.0] if k < trip_num - 1 else [100.0]:
                    if charge_success:
                        break

                    s_real = forced_station
                    s_idx = s_real - 1
                    m = forced_mode

                    drive_time_to = trip[f"drive_time_to_s{s_real}"]
                    drive_time_from = trip[f"drive_time_from_s{s_real}"] if k < trip_num - 1 else 0
                    drive_cost_to = drive_time_to * COST_PER_MIN
                    drive_cost_from = drive_time_from * COST_PER_MIN if k < trip_num - 1 else 0
                    total_drive_cost = drive_cost_to + drive_cost_from

                    soc_consume_to = drive_time_to * SOC_CONSUME_PER_MIN
                    soc_at_station_mean = current_soc_mean - soc_consume_to
                    if soc_at_station_mean < SOC_MIN - EPS:
                        continue  # 真实跌破20%，直接跳过
                    soc_at_station_mean = round(soc_at_station_mean, 1)

                    charge_amount = target_soc - soc_at_station_mean
                    if charge_amount < 0:
                        charge_amount = 0.0
                    charge_amount = round(charge_amount, 1)

                    if charge_amount == 0:
                        if k < trip_num - 1:
                            direct_key = (v, k, "direct")
                            if direct_key in drive_data:
                                direct = drive_data[direct_key]
                                soc_after_drive_mean = max(current_soc_mean - direct["soc_consume"], SOC_MIN)
                                soc_after_drive_mean = round(soc_after_drive_mean, 1)

                                # 关键修改：不充电决策下的空驶固定成本乘以存活概率
                                total_expected_cost += direct["cost"] * current_survival_prob

                                scheme_detail.append({
                                    "stage": k,
                                    "type": "no_charge",
                                    "drive_time": direct["time"],
                                    "drive_cost": direct["cost"],
                                    "soc_before": current_soc_mean,
                                    "soc_after": soc_after_drive_mean
                                })
                                current_soc_mean = soc_after_drive_mean
                            charge_success = True
                            break
                        else:
                            continue

                    # 计算充电时间
                    charge_time_steps = cached_charge_time(m, soc_at_station_mean, target_soc)
                    charge_time_minutes = charge_time_steps * T_STEP
                    charge_steps = int(charge_time_steps)

                    if charge_steps <= 0:
                        continue

                    # 时间窗口校验
                    if k == trip_num - 1:
                        available_charge_time = MAX_TOU_MINUTE - trip["t_end"] - drive_time_to
                        t_arr_minute = trip["t_end"] + drive_time_to
                        t_dep_deadline_minute = MAX_TOU_MINUTE
                    else:
                        gap_time = trips[k + 1]["t_start"] - trip["t_end"]
                        total_required_time = drive_time_to + charge_time_minutes + drive_time_from
                        if total_required_time > gap_time + EPS:
                            continue
                        available_charge_time = gap_time - drive_time_to - drive_time_from
                        t_arr_minute = trip["t_end"] + drive_time_to
                        t_dep_deadline_minute = trips[k + 1]["t_start"] - drive_time_from

                    if charge_time_minutes > available_charge_time + EPS:
                        continue

                    # 转换为时间步
                    t_arr_step = int(round(t_arr_minute / T_STEP))
                    t_dep_deadline_step = int(round(t_dep_deadline_minute / T_STEP))
                    t_min_start = max(0, t_arr_step)
                    t_max_start = t_dep_deadline_step - charge_steps

                    if t_min_start > t_max_start:
                        continue

                    # 寻找第一个连续空闲的充电时段
                    t_start = None
                    for t_candidate in range(t_min_start, t_max_start + 1):
                        t_end_candidate = t_candidate + charge_steps
                        if t_end_candidate > TOTAL_TIME_STEPS:
                            break
                        if np.sum(pile_usage_strategy1[s_idx, m, t_candidate:t_end_candidate]) == 0:
                            t_start = t_candidate
                            break

                    if t_start is None:
                        continue

                    t_end = t_start + charge_steps
                    pile_usage_strategy1[s_idx, m, t_start:t_end] = 1
                    a_smt[s_idx, m, t_start:t_end] = 1

                    # 计算充电的期望成本（对齐单车辆DP）
                    elec_cost = np.sum(tou_array[t_start:t_end]) * T_STEP
                    expected_charge_wear = 0.0
                    for d in valid_discharge:
                        soc_after_discharge = max(current_soc_mean + discharge_mean - d, SOC_MIN)
                        soc_at_station = max(soc_after_discharge - soc_consume_to, SOC_MIN)
                        soc_at_station = round(soc_at_station, 1)
                        wear = cached_wear_cost(m, soc_at_station, target_soc)
                        # 关键修改：充电引发的损耗使用当前车次放电前的 old_survival_prob 联合计算
                        expected_charge_wear += wear * single_prob * old_survival_prob

                    # 关键修改：固定空驶费和分时电费必须乘以当前存活概率，再加上充电引发的损耗期望
                    total_expected_cost += (total_drive_cost + elec_cost) * current_survival_prob + expected_charge_wear

                    # 空驶去下一站
                    if k < trip_num - 1:
                        soc_consume_from = drive_time_from * SOC_CONSUME_PER_MIN
                        soc_after_drive_mean = max(target_soc - soc_consume_from, SOC_MIN)
                        soc_after_drive_mean = round(soc_after_drive_mean, 1)
                        current_soc_mean = soc_after_drive_mean
                    else:
                        current_soc_mean = target_soc

                    charge_success = True

                if not charge_success:
                    raise ValueError(f"车辆{v}第{k}车次后在{station_name}无可用{mode_name}桩")

            if abs(current_soc_mean - 100.0) > EPS:
                raise ValueError(f"车辆{v}末班车未充满电，当前平均SOC={current_soc_mean}%")

            strategy1_schemes[v] = {
                "c_omega": round(total_expected_cost, 2),
                "a_smt": a_smt,
                "scheme_detail": scheme_detail,
                "final_soc": 100.0,
                "vehicle_id": v
            }

        except Exception as e:
            feasible = False
            print(f"\n❌ 车辆{v}策略1生成失败：{e}\n")
            break

    if all(v in strategy1_schemes for v in vehicle_ids):
        for v in vehicle_ids:
            initial_columns[v].append(strategy1_schemes[v])
        if VERBOSE:
            print("🎉 策略1全部车辆生成成功，已加入初始列")
    else:
        print("⚠️ 策略1部分车辆生成失败，跳过该策略")

    # ---------------------- 策略2：所有车次充到100%（备用） ----------------------
    if VERBOSE:
        print("\n🚀 生成初始策略2：所有车次充到100%，末班车强制充到100%")

    pile_usage_strategy2 = np.zeros((STATION_NUM, 2, TOTAL_TIME_STEPS), dtype=int)
    strategy2_schemes = {}

    for v in vehicle_ids:
        trips = vehicle_trips[v]
        trip_num = len(trips)
        current_soc_mean = SOC_MAX
        total_expected_cost = 0.0
        current_survival_prob = 1.0  # 初始化策略2状态存活率
        a_smt = np.zeros((STATION_NUM, 2, TOTAL_TIME_STEPS))
        scheme_detail = []
        feasible = True

        forced_station = VEHICLE_FORCED_STATION.get(v, 1)
        forced_mode = VEHICLE_FORCED_MODE.get(v, 0)
        station_name = f"充电站{forced_station}"
        mode_name = "慢充" if forced_mode == 0 else "快充"

        try:
            for k in range(trip_num):
                trip = trips[k]
                discharge_mean = trip["discharge_mean"]
                discharge_delta = trip["discharge_delta"]
                discharge_vals = get_uniform_distribution(discharge_mean, discharge_delta, SOC_STEP)
                discharge_list = list(discharge_vals.keys())
                single_prob = 1.0 / len(discharge_list) if discharge_list else 0.0
                d_to_z = {d: (d - discharge_mean) / discharge_delta if discharge_delta != 0 else 0.0 for d in
                          discharge_list}

                valid_discharge = []
                for d in discharge_list:
                    z = d_to_z[d]
                    if abs(z) <= LOCAL_CONSTRAINT_RATIO * (k + 1) + EPS:
                        valid_discharge.append(d)

                valid_prob = len(valid_discharge) * single_prob
                expected_discharge_wear = 0.0
                for d in valid_discharge:
                    soc_after = max(current_soc_mean - d, SOC_MIN)
                    soc_after = round(soc_after, 1)
                    wear = cached_wear_cost(0, current_soc_mean, soc_after)
                    # 关键修改
                    expected_discharge_wear += wear * single_prob * current_survival_prob

                total_expected_cost += expected_discharge_wear
                old_survival_prob = current_survival_prob
                current_survival_prob *= valid_prob

                scheme_detail.append({
                    "stage": k,
                    "type": "discharge",
                    "discharge_energy": discharge_mean,
                    "soc_before": current_soc_mean,
                    "soc_after": round(current_soc_mean - discharge_mean, 1),
                    "wear_cost": round(expected_discharge_wear, 4)
                })

                current_soc_mean = round(current_soc_mean - discharge_mean, 1)

                charge_success = False
                for target_soc in [100.0, 95.0, 90.0, 85.0]:
                    if charge_success:
                        break

                    s_real = forced_station
                    s_idx = s_real - 1
                    m = forced_mode

                    drive_time_to = trip[f"drive_time_to_s{s_real}"]
                    drive_time_from = trip[f"drive_time_from_s{s_real}"] if k < trip_num - 1 else 0
                    drive_cost_to = drive_time_to * COST_PER_MIN
                    drive_cost_from = drive_time_from * COST_PER_MIN if k < trip_num - 1 else 0
                    total_drive_cost = drive_cost_to + drive_cost_from

                    soc_consume_to = drive_time_to * SOC_CONSUME_PER_MIN
                    soc_at_station_mean = current_soc_mean - soc_consume_to
                    if soc_at_station_mean < SOC_MIN - EPS:
                        continue  # 真实跌破20%，直接跳过
                    soc_at_station_mean = round(soc_at_station_mean, 1)

                    charge_amount = target_soc - soc_at_station_mean
                    if charge_amount < 0:
                        charge_amount = 0.0
                    charge_amount = round(charge_amount, 1)

                    if charge_amount == 0:
                        if k < trip_num - 1:
                            direct_key = (v, k, "direct")
                            if direct_key in drive_data:
                                direct = drive_data[direct_key]
                                soc_after_drive_mean = max(current_soc_mean - direct["soc_consume"], SOC_MIN)
                                soc_after_drive_mean = round(soc_after_drive_mean, 1)

                                # 关键修改
                                total_expected_cost += direct["cost"] * current_survival_prob

                                scheme_detail.append({
                                    "stage": k,
                                    "type": "no_charge",
                                    "drive_time": direct["time"],
                                    "drive_cost": direct["cost"],
                                    "soc_before": current_soc_mean,
                                    "soc_after": soc_after_drive_mean
                                })
                                current_soc_mean = soc_after_drive_mean
                            charge_success = True
                            break
                        else:
                            continue

                    charge_time_steps = cached_charge_time(m, soc_at_station_mean, target_soc)
                    charge_time_minutes = charge_time_steps * T_STEP
                    charge_steps = int(charge_time_steps)

                    if charge_steps <= 0:
                        continue

                    if k == trip_num - 1:
                        available_charge_time = MAX_TOU_MINUTE - trip["t_end"] - drive_time_to
                        t_arr_minute = trip["t_end"] + drive_time_to
                        t_dep_deadline_minute = MAX_TOU_MINUTE
                    else:
                        gap_time = trips[k + 1]["t_start"] - trip["t_end"]
                        total_required_time = drive_time_to + charge_time_minutes + drive_time_from
                        if total_required_time > gap_time + EPS:
                            continue
                        available_charge_time = gap_time - drive_time_to - drive_time_from
                        t_arr_minute = trip["t_end"] + drive_time_to
                        t_dep_deadline_minute = trips[k + 1]["t_start"] - drive_time_from

                    if charge_time_minutes > available_charge_time + EPS:
                        continue

                    t_arr_step = int(round(t_arr_minute / T_STEP))
                    t_dep_deadline_step = int(round(t_dep_deadline_minute / T_STEP))
                    t_min_start = max(0, t_arr_step)
                    t_max_start = t_dep_deadline_step - charge_steps

                    if t_min_start > t_max_start:
                        continue

                    t_start = None
                    for t_candidate in range(t_min_start, t_max_start + 1):
                        t_end_candidate = t_candidate + charge_steps
                        if t_end_candidate > TOTAL_TIME_STEPS:
                            break
                        if np.sum(pile_usage_strategy2[s_idx, m, t_candidate:t_end_candidate]) == 0:
                            t_start = t_candidate
                            break

                    if t_start is None:
                        continue

                    t_end = t_start + charge_steps
                    pile_usage_strategy2[s_idx, m, t_start:t_end] = 1
                    a_smt[s_idx, m, t_start:t_end] = 1

                    elec_cost = np.sum(tou_array[t_start:t_end]) * T_STEP
                    expected_charge_wear = 0.0
                    for d in valid_discharge:
                        soc_after_discharge = max(current_soc_mean + discharge_mean - d, SOC_MIN)
                        soc_at_station = max(soc_after_discharge - soc_consume_to, SOC_MIN)
                        soc_at_station = round(soc_at_station, 1)
                        wear = cached_wear_cost(m, soc_at_station, target_soc)
                        # 关键修改
                        expected_charge_wear += wear * single_prob * old_survival_prob

                    # 关键修改
                    total_expected_cost += (total_drive_cost + elec_cost) * current_survival_prob + expected_charge_wear

                    if k < trip_num - 1:
                        soc_consume_from = drive_time_from * SOC_CONSUME_PER_MIN
                        soc_after_drive_mean = max(target_soc - soc_consume_from, SOC_MIN)
                        soc_after_drive_mean = round(soc_after_drive_mean, 1)
                        current_soc_mean = soc_after_drive_mean
                    else:
                        current_soc_mean = target_soc

                    charge_success = True

                if not charge_success:
                    raise ValueError(f"车辆{v}第{k}车次后在{station_name}无可用{mode_name}桩")

            if abs(current_soc_mean - 100.0) > EPS:
                raise ValueError(f"车辆{v}末班车未充满电")

            strategy2_schemes[v] = {
                "c_omega": round(total_expected_cost, 2),
                "a_smt": a_smt,
                "scheme_detail": scheme_detail,
                "final_soc": 100.0,
                "vehicle_id": v
            }

        except Exception as e:
            feasible = False
            print(f"\n❌ 车辆{v}策略2生成失败：{e}\n")
            break

    if all(v in strategy2_schemes for v in vehicle_ids):
        for v in vehicle_ids:
            initial_columns[v].append(strategy2_schemes[v])
        if VERBOSE:
            print("🎉 策略2全部车辆生成成功，已加入初始列")
    else:
        print("⚠️ 策略2部分车辆生成失败，跳过该策略")

    # 检查初始列是否足够
    for v in vehicle_ids:
        if len(initial_columns[v]) == 0:
            raise ValueError(f"❌ 车辆{v}无任何可行初始方案")

    if VERBOSE:
        print(f"\n✅ 初始列生成完成，每辆车有 {len(initial_columns[vehicle_ids[0]])} 个初始方案")
        print(f"📌 所有初始列均采用期望成本计算（对齐单车辆DP）")

    return initial_columns


def build_rmp(initial_columns):
    """构建RMP模型"""
    model = gp.Model("Bus_Charging_RMP")
    model.setParam("OutputFlag", 0 if not VERBOSE else 1)
    model.setParam("Threads", 4)

    # 变量：x[v][i] = 车辆v选用第i个方案
    x = {}
    for v in initial_columns.keys():
        x[v] = []
        for i, scheme in enumerate(initial_columns[v]):
            var = model.addVar(
                lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                name=f"x_{v}_{i}", obj=scheme["c_omega"]
            )
            x[v].append(var)

    # 约束1：车辆方案唯一性
    vehicle_constr = {}
    for v in initial_columns.keys():
        expr = gp.quicksum(x[v]) == 1.0
        constr = model.addConstr(expr, name=f"vehicle_{v}")
        vehicle_constr[v] = constr

    # 约束2：桩容量约束（每站每模式桩≤1）
    pile_constr = {}
    for s in range(STATION_NUM):
        for m in range(2):
            for t in range(TOTAL_TIME_STEPS):
                expr = gp.LinExpr()
                for v in initial_columns.keys():
                    for i, scheme in enumerate(initial_columns[v]):
                        if scheme["a_smt"][s, m, t] > 0:
                            expr += scheme["a_smt"][s, m, t] * x[v][i]
                constr = model.addConstr(expr <= 1.0, name=f"pile_{s}_{m}_{t}")
                pile_constr[(s, m, t)] = constr

    return model, x, vehicle_constr, pile_constr


def solve_rmp(model, vehicle_constr, pile_constr):
    """求解RMP，返回对偶变量"""
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise ValueError(f"❌ RMP求解失败，状态码：{model.Status}")

    # 提取对偶变量
    pi = {v: vehicle_constr[v].Pi for v in vehicle_constr.keys()}
    sigma = np.zeros((STATION_NUM, 2, TOTAL_TIME_STEPS))
    for (s, m, t), constr in pile_constr.items():
        sigma[s, m, t] = constr.Pi

    return model.ObjVal, pi, sigma