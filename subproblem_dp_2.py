import numpy as np
from functools import lru_cache
from config import *
from data_preprocess import precompute_elec_cumsum

# 模块内部私有全局变量
_cached_wear_dict = None
_cached_nonlinear_dict = None
_elec_cumsum = None


@lru_cache(maxsize=CACHE_SIZE)
def cached_wear_cost(mode, start_soc, end_soc):
    if _cached_wear_dict is None:
        raise ValueError("❌ 损耗表未初始化，请先调用 init_subproblem_cache()")
    mode_str = "slow" if mode == 0 else "fast"
    start_idx = int(round((start_soc - SOC_MIN) / SOC_STEP))
    end_idx = int(round((end_soc - SOC_MIN) / SOC_STEP))
    start_idx = max(0, min(start_idx, _cached_wear_dict[mode_str].shape - 1))
    end_idx = max(0, min(end_idx, _cached_wear_dict[mode_str].shape - 1))
    return _cached_wear_dict[mode_str][start_idx, end_idx]


@lru_cache(maxsize=CACHE_SIZE)
def cached_charge_time(mode, start_soc, end_soc):
    if _cached_nonlinear_dict is None:
        raise ValueError("❌ 充电表未初始化，请先调用 init_subproblem_cache()")
    mode_str = "slow" if mode == 0 else "fast"
    start_idx = int(round((start_soc - SOC_MIN) / SOC_STEP))
    end_idx = int(round((end_soc - SOC_MIN) / SOC_STEP))
    start_idx = max(0, min(start_idx, _cached_nonlinear_dict[mode_str].shape - 1))
    end_idx = max(0, min(end_idx, _cached_nonlinear_dict[mode_str].shape - 1))
    return _cached_nonlinear_dict[mode_str][start_idx, end_idx]


def init_subproblem_cache(wear_dict, nonlinear_dict):
    global _cached_wear_dict, _cached_nonlinear_dict
    _cached_wear_dict = wear_dict
    _cached_nonlinear_dict = nonlinear_dict
    cached_wear_cost.cache_clear()
    cached_charge_time.cache_clear()
    if VERBOSE:
        print("✅ 子问题缓存初始化完成")


def precompute_comprehensive_cumsum(tou_array, sigma_smt):
    comprehensive_cumsum = np.zeros((STATION_NUM, 2, TOTAL_TIME_STEPS + 1))
    for s in range(STATION_NUM):
        for m in range(2):
            w = np.zeros(TOTAL_TIME_STEPS)
            for t in range(TOTAL_TIME_STEPS):
                elec_cost = tou_array[t] * T_STEP
                sigma_penalty = -sigma_smt[s, m, t]
                w[t] = elec_cost + sigma_penalty
            comprehensive_cumsum[s, m] = np.cumsum(np.insert(w, 0, 0))
    return comprehensive_cumsum


def get_uniform_distribution(mean, delta, step):
    min_val = max(mean - delta, 0)
    max_val = mean + delta
    values = np.arange(min_val, max_val + step, step)
    return {round(val, 1): 1.0 for val in values}


def solve_subproblem(vehicle_id, trips, drive_data, tou_array, sigma_smt, pi_v):
    if _cached_wear_dict is None or _cached_nonlinear_dict is None:
        raise ValueError("❌ 子问题缓存未初始化，请先调用 init_subproblem_cache()")

    trip_num = len(trips)
    elec_cumsum = precompute_elec_cumsum(tou_array)
    comprehensive_cumsum = precompute_comprehensive_cumsum(tou_array, sigma_smt)

    discharge_base = []
    for k in range(trip_num):
        trip = trips[k]
        mean = trip["discharge_mean"]
        delta = trip["discharge_delta"]
        discharge_vals = get_uniform_distribution(mean, delta, SOC_STEP)
        discharge_list = list(discharge_vals.keys())
        single_prob = 1.0 / len(discharge_list) if discharge_list else 0.0
        d_to_z = {d: (d - mean) / delta if delta != 0 else 0.0 for d in discharge_list}
        discharge_base.append((mean, delta, discharge_list, single_prob, d_to_z))

    dp = [{} for _ in range(trip_num)]

    first_mean, first_delta, first_d_list, first_single_prob, first_d_to_z = discharge_base
    first_symbolic_soc = round(SOC_MAX - first_mean, 2)
    first_path_tags = {}

    for d in first_d_list:
        z = first_d_to_z[d]
        cum_abs_z = abs(z)
        if cum_abs_z > LOCAL_CONSTRAINT_RATIO * 1 + EPS:
            continue
        soc_after = max(SOC_MAX - d, SOC_MIN)
        soc_after = round(soc_after, 2)
        wear_cost = cached_wear_cost(0, SOC_MAX, soc_after)
        path_cost = wear_cost * first_single_prob
        first_path_tags[(round(z, 2),)] = {
            'prob': first_single_prob, 'cum_abs_z': cum_abs_z, 'soc': soc_after, 'cost': path_cost
        }

    if not first_path_tags:
        return {"is_feasible": False, "min_rc": np.inf, "best_scheme": None}

    total_first_cost = sum(tag['cost'] for tag in first_path_tags.values())
    dp[first_symbolic_soc] = {
        'total_cost': total_first_cost, 'total_obj': total_first_cost,
        'path_tags': first_path_tags, 'a_smt': np.zeros((STATION_NUM, 2, TOTAL_TIME_STEPS)),
        'charge_amount': None, 'charge_mode': None, 'station': None, 'prev_soc': None,
        'exp_wear': total_first_cost, 'exp_elec': 0.0, 'exp_drive': 0.0,
        'unscaled_elec': 0.0, 'unscaled_drive': 0.0, 't_start': 0, 't_end': 0
    }

    for k in range(trip_num - 1):
        current_dp = dp[k]
        next_dp = {}
        next_mean, next_delta, next_d_list, next_single_prob, next_d_to_z = discharge_base[k + 1]
        trip = trips[k]
        gap_time = trips[k + 1]["t_start"] - trip["t_end"]

        for sym_soc_prev, state_prev in current_dp.items():
            prev_path_tags = state_prev['path_tags']
            prev_total_cost = state_prev['total_cost']
            prev_total_obj = state_prev['total_obj']
            a_smt_prev = state_prev['a_smt'].copy()

            path_info = {}
            for prev_z_seq, prev_tag in prev_path_tags.items():
                prev_cum_abs_z = prev_tag['cum_abs_z']
                prev_soc = prev_tag['soc']
                feasible_next_d = [
                    d for d in next_d_list
                    if prev_cum_abs_z + abs(next_d_to_z[d]) <= LOCAL_CONSTRAINT_RATIO * (k + 2) + EPS
                ]
                if feasible_next_d:
                    path_info[prev_z_seq] = {
                        'max_feasible_d': max(feasible_next_d), 'soc_prev': prev_soc,
                        'cum_abs_z': prev_cum_abs_z, 'feasible_next_d': feasible_next_d, 'prob': prev_tag['prob']
                    }

            if not path_info:
                continue

            # ====== 1. 不充电分支 ======
            direct_key = (vehicle_id, k, "direct")
            if direct_key in drive_data:
                direct = drive_data[direct_key]
                direct_time = direct["time"]
                direct_cost = direct["cost"]
                direct_soc_consume = direct["soc_consume"]

                if direct_time <= gap_time + EPS:
                    is_valid_decision = True
                    for info in path_info.values():
                        if info['soc_prev'] - direct_soc_consume - info['max_feasible_d'] < SOC_MIN - EPS:
                            is_valid_decision = False
                            break

                    if is_valid_decision:
                        next_path_tags = {}
                        total_next_cost = 0.0
                        total_next_obj = 0.0
                        exp_stage_drive = 0.0
                        exp_stage_wear = 0.0

                        for prev_z_seq, info in path_info.items():
                            prev_prob = info['prob']
                            prev_soc = info['soc_prev']
                            prev_cum_abs_z = info['cum_abs_z']

                            soc_after_direct = max(prev_soc - direct_soc_consume, SOC_MIN)
                            no_charge_cost = direct_cost * prev_prob
                            total_next_cost += no_charge_cost
                            total_next_obj += no_charge_cost
                            exp_stage_drive += no_charge_cost

                            for next_d in info['feasible_next_d']:
                                next_z = next_d_to_z[next_d]
                                soc_after_discharge = max(soc_after_direct - next_d, SOC_MIN)

                                joint_prob = prev_prob * next_single_prob
                                discharge_wear = cached_wear_cost(0, round(soc_after_direct, 1),
                                                                  round(soc_after_discharge, 2))
                                discharge_cost = discharge_wear * joint_prob
                                total_next_cost += discharge_cost
                                total_next_obj += discharge_cost
                                exp_stage_wear += discharge_cost
                                next_path_tags[prev_z_seq + (round(next_z, 2),)] = {
                                    'prob': joint_prob, 'cum_abs_z': prev_cum_abs_z + abs(next_z),
                                    'soc': round(soc_after_discharge, 2), 'cost': discharge_cost, 'obj': discharge_cost
                                }

                        if next_path_tags:
                            new_symbolic_soc = round(sym_soc_prev - direct_soc_consume - next_mean, 2)
                            new_total_cost = prev_total_cost + total_next_cost
                            new_total_obj = prev_total_obj + total_next_obj
                            if new_symbolic_soc not in next_dp or new_total_obj < next_dp[new_symbolic_soc][
                                'total_obj']:
                                next_dp[new_symbolic_soc] = {
                                    'total_cost': new_total_cost, 'total_obj': new_total_obj,
                                    'path_tags': next_path_tags, 'a_smt': a_smt_prev.copy(),
                                    'charge_amount': 0.0, 'charge_mode': "不充电", 'station': None,
                                    'prev_soc': sym_soc_prev,
                                    't_start': 0, 't_end': 0, 'unscaled_elec': 0.0, 'unscaled_drive': direct_cost,
                                    'exp_elec': 0.0, 'exp_drive': exp_stage_drive, 'exp_wear': exp_stage_wear
                                }

            # ====== 2. 充电分支 ======
            for s_real in range(1, STATION_NUM + 1):
                s_idx = s_real - 1
                for m in [0, 1]:
                    drive_to_key = (vehicle_id, k, s_real, "to")
                    drive_from_key = (vehicle_id, k, s_real, "from")

                    if drive_to_key not in drive_data or drive_from_key not in drive_data:
                        continue

                    drive_to = drive_data[drive_to_key]
                    drive_from = drive_data[drive_from_key]

                    total_drive_time = drive_to["time"] + drive_from["time"]
                    total_drive_cost = drive_to["cost"] + drive_from["cost"]

                    if total_drive_time > gap_time + EPS:
                        continue

                    # 去充电站路上不能跌破SOC_MIN
                    if any(info['soc_prev'] - drive_to["soc_consume"] < SOC_MIN - EPS for info in path_info.values()):
                        continue

                    valid_paths = list(path_info.items())

                    # 计算所需的充电量下界和上界
                    min_reqs = []
                    max_reqs = []
                    for seq, info in valid_paths:
                        # 保证放完电后还有SOC_MIN
                        req = info['max_feasible_d'] + drive_to["soc_consume"] + drive_from["soc_consume"] - info[
                            'soc_prev'] + SOC_MIN
                        min_reqs.append(max(0.0, req))
                        # 保证充电时不溢出SOC_MAX
                        max_reqs.append(SOC_MAX - (info['soc_prev'] - drive_to["soc_consume"]))

                    min_charge_raw = max(min_reqs)
                    max_charge_raw = min(max_reqs)

                    # 向上取整和向下取整
                    min_charge = np.ceil(min_charge_raw / SOC_STEP) * SOC_STEP
                    max_charge = np.floor(max_charge_raw / SOC_STEP) * SOC_STEP

                    if min_charge > max_charge + EPS:
                        continue

                    charge_amounts = np.arange(min_charge, max_charge + EPS, SOC_STEP)

                    for charge_soc in charge_amounts:
                        if charge_soc > max_charge + EPS:
                            continue

                        charge_steps_list = []
                        for seq, info in valid_paths:
                            soc_start_chg = round(info['soc_prev'] - drive_to["soc_consume"], 1)
                            soc_end_chg = min(soc_start_chg + charge_soc, SOC_MAX)
                            charge_steps_list.append(int(cached_charge_time(m, soc_start_chg, soc_end_chg)))

                        max_charge_steps = max(charge_steps_list) if charge_steps_list else 0

                        if max_charge_steps == 0:
                            continue

                        t_min_start = max(0, int(round((trip["t_end"] + drive_to["time"]) / T_STEP)))
                        t_max_start = int(
                            round((trips[k + 1]["t_start"] - drive_from["time"]) / T_STEP)) - max_charge_steps

                        if t_min_start > t_max_start:
                            continue

                        # 寻找最优充电时间窗口
                        window_costs = comprehensive_cumsum[s_idx, m,
                                       t_min_start + max_charge_steps: t_max_start + max_charge_steps + 1] - comprehensive_cumsum[
                                                                                                             s_idx, m,
                                                                                                             t_min_start: t_max_start + 1]

                        if len(window_costs) == 0:
                            continue

                        min_cost_idx = np.argmin(window_costs)
                        t_start = t_min_start + min_cost_idx
                        t_end = t_start + max_charge_steps
                        min_elec = elec_cumsum[t_end] - elec_cumsum[t_start]

                        next_path_tags = {}
                        total_next_cost = 0.0
                        total_next_obj = 0.0
                        exp_stage_drive = 0.0
                        exp_stage_elec = 0.0
                        exp_stage_wear = 0.0

                        for prev_z_seq, info in valid_paths:
                            prev_prob = info['prob']
                            prev_soc = info['soc_prev']
                            prev_cum_abs_z = info['cum_abs_z']

                            soc_at_station = max(prev_soc - drive_to["soc_consume"], SOC_MIN)
                            soc_after_charge = min(soc_at_station + charge_soc, SOC_MAX)
                            soc_after_drive = max(soc_after_charge - drive_from["soc_consume"], SOC_MIN)

                            charge_wear = cached_wear_cost(m, round(soc_at_station, 1), round(soc_after_charge, 1))
                            charge_cost = (total_drive_cost + charge_wear + min_elec) * prev_prob

                            total_next_cost += charge_cost
                            total_next_obj += (total_drive_cost + charge_wear + window_costs[min_cost_idx]) * prev_prob

                            exp_stage_drive += total_drive_cost * prev_prob
                            exp_stage_elec += min_elec * prev_prob
                            exp_stage_wear += charge_wear * prev_prob

                            for next_d in info['feasible_next_d']:
                                next_z = next_d_to_z[next_d]
                                soc_after_discharge = max(soc_after_drive - next_d, SOC_MIN)

                                joint_prob = prev_prob * next_single_prob
                                discharge_wear = cached_wear_cost(0, round(soc_after_drive, 1),
                                                                  round(soc_after_discharge, 2))

                                discharge_cost = discharge_wear * joint_prob
                                total_next_cost += discharge_cost
                                total_next_obj += discharge_cost
                                exp_stage_wear += discharge_cost

                                next_path_tags[prev_z_seq + (round(next_z, 2),)] = {
                                    'prob': joint_prob, 'cum_abs_z': prev_cum_abs_z + abs(next_z),
                                    'soc': round(soc_after_discharge, 2), 'cost': discharge_cost, 'obj': discharge_cost
                                }

                        if next_path_tags:
                            new_symbolic_soc = round(sym_soc_prev + charge_soc - drive_to["soc_consume"] - drive_from[
                                "soc_consume"] - next_mean, 2)
                            new_total_cost = prev_total_cost + total_next_cost
                            new_total_obj = prev_total_obj + total_next_obj

                            if new_symbolic_soc not in next_dp or new_total_obj < next_dp[new_symbolic_soc][
                                'total_obj']:
                                new_a_smt = a_smt_prev.copy()
                                new_a_smt[s_idx, m, t_start:t_end] = 1.0
                                next_dp[new_symbolic_soc] = {
                                    'total_cost': new_total_cost, 'total_obj': new_total_obj,
                                    'path_tags': next_path_tags, 'a_smt': new_a_smt,
                                    'charge_amount': charge_soc, 'charge_mode': "慢充" if m == 0 else "快充",
                                    'station': s_real, 'prev_soc': sym_soc_prev,
                                    't_start': t_start, 't_end': t_end,
                                    'unscaled_elec': min_elec, 'unscaled_drive': total_drive_cost,
                                    'exp_elec': exp_stage_elec, 'exp_drive': exp_stage_drive, 'exp_wear': exp_stage_wear
                                }

        dp[k + 1] = next_dp

    # ---------------------- 最后一次充电 ----------------------
    final_dp = dp[-1]
    if not final_dp:
        return {"is_feasible": False, "min_rc": np.inf, "best_scheme": None}

    final_candidates = []

    for sym_soc_final, state_final in final_dp.items():
        for s_real in range(1, STATION_NUM + 1):
            s_idx = s_real - 1
            for m in [0, 1]:
                drive_to_key = (vehicle_id, trip_num - 1, s_real, "to")
                if drive_to_key not in drive_data:
                    continue

                drive_to_soc = drive_data[drive_to_key]["soc_consume"]
                total_drive_cost = drive_data[drive_to_key]["cost"]
                drive_time = drive_data[drive_to_key]["time"]

                if any(tag['soc'] - drive_to_soc < SOC_MIN - EPS for tag in state_final['path_tags'].values()):
                    continue

                valid_paths = list(state_final['path_tags'].items())

                charge_steps_list = []
                for seq, tag in valid_paths:
                    soc_start = round(tag['soc'] - drive_to_soc, 1)
                    charge_steps_list.append(int(cached_charge_time(m, soc_start, SOC_MAX)))

                max_charge_steps = max(charge_steps_list) if charge_steps_list else 0
                if max_charge_steps == 0:
                    continue

                min_start_step = int(round((trips[-1]["t_end"] + drive_time) / T_STEP))
                max_start_step = MAX_TOU_TIME_STEP - max_charge_steps

                if min_start_step > max_start_step:
                    continue

                window_costs = comprehensive_cumsum[s_idx, m,
                               min_start_step + max_charge_steps: max_start_step + max_charge_steps + 1] - comprehensive_cumsum[
                                                                                                           s_idx, m,
                                                                                                           min_start_step: max_start_step + 1]

                if len(window_costs) == 0:
                    continue

                min_cost_idx = np.argmin(window_costs)
                t_start = min_start_step + min_cost_idx
                t_end = min_start_step + min_cost_idx + max_charge_steps
                min_elec = elec_cumsum[t_end] - elec_cumsum[t_start]

                exp_stage_drive = 0.0
                exp_stage_elec = 0.0
                exp_stage_wear = 0.0
                final_obj_total = 0.0

                for seq, tag in valid_paths:
                    final_prob = tag['prob']
                    final_soc = tag['soc']
                    soc_at_final_station = round(final_soc - drive_to_soc, 1)
                    final_wear = cached_wear_cost(m, soc_at_final_station, SOC_MAX)

                    exp_stage_drive += total_drive_cost * final_prob
                    exp_stage_elec += min_elec * final_prob
                    exp_stage_wear += final_wear * final_prob
                    final_obj_total += (total_drive_cost + final_wear + window_costs[min_cost_idx]) * final_prob

                new_a_smt = state_final['a_smt'].copy()
                new_a_smt[s_idx, m, t_start:t_end] = 1.0

                final_candidates.append({
                    'total_cost': state_final['total_cost'] + exp_stage_drive + exp_stage_elec + exp_stage_wear,
                    'total_obj': state_final['total_obj'] + final_obj_total,
                    'a_smt': new_a_smt, 'symbolic_soc': sym_soc_final,
                    'final_mode': "慢充" if m == 0 else "快充", 'final_station': s_real,
                    'path_count': len(state_final['path_tags']), 'valid_path_count': len(valid_paths),
                    't_start': t_start, 't_end': t_end,
                    'unscaled_elec': min_elec, 'unscaled_drive': total_drive_cost,
                    'exp_elec': exp_stage_elec, 'exp_drive': exp_stage_drive, 'exp_wear': exp_stage_wear
                })

    if not final_candidates:
        return {"is_feasible": False, "min_rc": np.inf, "best_scheme": None}

    best_candidate = min(final_candidates, key=lambda x: x['total_obj'] - pi_v)

    # ================= 完美路径回溯 =================
    scheme_detail = []

    scheme_detail.append({
        "stage": trip_num - 1, "type": "charge",
        "station": best_candidate['final_station'], "mode": best_candidate['final_mode'],
        "charge_amount": 100.0,
        "charge_start_minute": best_candidate['t_start'] * T_STEP,
        "charge_end_minute": best_candidate['t_end'] * T_STEP,
        "charge_time": (best_candidate['t_end'] - best_candidate['t_start']) * T_STEP,
        "elec_cost": round(best_candidate['unscaled_elec'], 2),
        "drive_cost": round(best_candidate['unscaled_drive'], 2),
        "exp_elec_cost": round(best_candidate['exp_elec'], 4),
        "exp_drive_cost": round(best_candidate['exp_drive'], 4),
        "wear_cost": round(best_candidate['exp_wear'], 4)
    })

    curr_sym_soc = best_candidate['symbolic_soc']
    for k in range(trip_num - 1, 0, -1):
        state = dp[k][curr_sym_soc]
        if state['charge_mode'] == "不充电":
            scheme_detail.append({
                "stage": k - 1, "type": "no_charge",
                "drive_cost": round(state['unscaled_drive'], 2),
                "exp_drive_cost": round(state['exp_drive'], 4),
                "exp_elec_cost": 0.0,
                "wear_cost": round(state['exp_wear'], 4)
            })
        else:
            scheme_detail.append({
                "stage": k - 1, "type": "charge",
                "station": state['station'], "mode": state['charge_mode'],
                "charge_amount": state['charge_amount'],
                "charge_start_minute": state['t_start'] * T_STEP,
                "charge_end_minute": state['t_end'] * T_STEP,
                "charge_time": (state['t_end'] - state['t_start']) * T_STEP,
                "elec_cost": round(state['unscaled_elec'], 2),
                "drive_cost": round(state['unscaled_drive'], 2),
                "exp_elec_cost": round(state['exp_elec'], 4),
                "exp_drive_cost": round(state['exp_drive'], 4),
                "wear_cost": round(state['exp_wear'], 4)
            })
        curr_sym_soc = state['prev_soc']

    first_state = dp[curr_sym_soc]
    scheme_detail.append({
        "stage": 0, "type": "discharge",
        "wear_cost": round(first_state['exp_wear'], 4),
        "exp_elec_cost": 0.0, "exp_drive_cost": 0.0, "drive_cost": 0.0
    })

    scheme_detail.reverse()

    return {
        "is_feasible": True,
        "min_rc": best_candidate['total_obj'] - pi_v,
        "best_scheme": {
            "c_omega": round(best_candidate['total_cost'], 2),
            "a_smt": best_candidate['a_smt'],
            "scheme_detail": scheme_detail,
            "final_soc": 100.0,
            "vehicle_id": vehicle_id,
            "valid_path_ratio": f"{best_candidate['valid_path_count']}/{best_candidate['path_count']}"
        }
    }