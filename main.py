import numpy as np
from gurobipy import GRB
from config import *
from data_preprocess import (
    read_vehicle_trips, read_tou_price, read_wear_data,
    read_nonlinear_data, compute_drive_params
)
from master_problem import init_initial_columns, build_rmp, solve_rmp
from subproblem_dp import init_subproblem_cache, solve_subproblem

def main():
    # ---------------------- 1. 数据路径配置（用户需根据实际路径修改） ----------------------
    VEHICLE_EXCEL_PATH = "D:/multipower/papercode/车辆运营计划.xlsx"
    TOU_EXCEL_PATH = "D:/multipower/papercode/TOU3.xlsx"
    WEAR_EXCEL_PATH = "D:/multipower/papercode/电池损耗对应表.xlsx"
    NONLINEAR_EXCEL_PATH = "D:/multipower/papercode/非线性充电对应表.xlsx"

    try:
        # ---------------------- 2. 数据预处理 ----------------------
        if VERBOSE:
            print("=" * 80)
            print("📊 多公交充电调度列生成求解器（仅小数解）")
            print("=" * 80 + "\n")
            print("🔧 数据预处理中...")

        vehicle_trips = read_vehicle_trips(VEHICLE_EXCEL_PATH)
        tou_array = read_tou_price(TOU_EXCEL_PATH)
        wear_dict = read_wear_data(WEAR_EXCEL_PATH)
        nonlinear_dict = read_nonlinear_data(NONLINEAR_EXCEL_PATH)
        drive_data = compute_drive_params(vehicle_trips)

        # 初始化子问题缓存（全局唯一一次）
        init_subproblem_cache(wear_dict, nonlinear_dict)

        if VERBOSE:
            print(f"✅ 读取到 {len(vehicle_trips)} 辆车，共 {sum(len(t) for t in vehicle_trips.values())} 个车次")
            print(f"✅ 充电站配置：{STATION_NUM}个站，每站1慢充1快充")
            print(f"✅ 空驶模型：{SOC_CONSUME_PER_MIN}%/分钟耗电，{COST_PER_MIN}元/分钟成本\n")

        # ---------------------- 3. 列生成初始化 ----------------------
        if VERBOSE:
            print("=" * 80)
            print("🚀 初始化初始列")
            print("=" * 80 + "\n")

        initial_columns = init_initial_columns(vehicle_trips, drive_data, tou_array, wear_dict, nonlinear_dict)
        # 检查初始列是否足够
        for v in initial_columns.keys():
            if len(initial_columns[v]) == 0:
                raise ValueError(f"❌ 车辆{v}无初始可行方案，无法继续")

        if VERBOSE:
            print(f"\n✅ 初始列生成完成，共生成 {sum(len(cols) for cols in initial_columns.values())} 个初始方案\n")

        # ---------------------- 4. 列生成迭代 ----------------------
        if VERBOSE:
            print("=" * 80)
            print("🔄 列生成迭代（线性规划小数解）")
            print("=" * 80 + "\n")

        current_columns = {v: list(cols) for v, cols in initial_columns.items()}
        iter_count = 0
        prev_obj = np.inf
        # 保存最后一次迭代的RMP解
        final_model = None
        final_x = None
        final_pi = None
        final_sigma = None

        while iter_count < MAX_ITER:
            iter_count += 1
            # 构建并求解RMP
            model, x, vehicle_constr, pile_constr = build_rmp(current_columns)
            rmp_obj, pi, sigma = solve_rmp(model, vehicle_constr, pile_constr)

            # 保存最后一次迭代的解
            final_model = model
            final_x = x
            final_pi = pi
            final_sigma = sigma

            # 打印迭代信息
            if VERBOSE:
                print(f"📌 迭代 {iter_count:2d} | RMP目标值：{rmp_obj:10.4f} 元 | σ范围：{sigma.min():8.2f} ~ {sigma.max():8.2f}")

            # 收敛判断
            if abs(prev_obj - rmp_obj) < EPS:
                if VERBOSE:
                    print(f"\n🎉 迭代收敛（目标值变化<{EPS}），共迭代 {iter_count} 次")
                break
            prev_obj = rmp_obj

            # 求解子问题，生成新列
            new_columns_added = 0
            for v in vehicle_trips.keys():
                res = solve_subproblem(v, vehicle_trips[v], drive_data, tou_array, sigma, pi[v])

                # None值判断和警告
                if res is None:
                    print(f"   ⚠️ 车辆{v}子问题返回None，跳过该车辆")
                    continue

                if res["is_feasible"] and res["min_rc"] < -EPS:
                    # 加入新列
                    current_columns[v].append(res["best_scheme"])
                    new_columns_added += 1
                    if VERBOSE:
                        print(f"   ✅ 车辆{v}新增有效列，检验数={res['min_rc']:10.4f}")

            if new_columns_added == 0:
                if VERBOSE:
                    print(f"\n🎉 无新有效列生成，迭代收敛")
                break

        # ---------------------- 5. 列生成小数解结果输出 ----------------------
        if VERBOSE:
            print("\n" + "=" * 80)
            print("🏆 列生成最终小数解结果")
            print("=" * 80 + "\n")
            print(f"📊 全局最小期望总成本：{final_model.ObjVal:.4f} 元")
            print(f"📊 平均每车期望成本：{final_model.ObjVal / len(vehicle_trips):.4f} 元/车")
            print(f"📊 总方案数：{sum(len(cols) for cols in current_columns.values())} 个")
            print(f"📊 初始列数：{sum(len(cols) for cols in initial_columns.values())} 个")
            print(f"📊 子问题生成新列数：{sum(len(cols) - len(initial_columns[v]) for v, cols in current_columns.items())} 个\n")

            print("🚗 各车辆方案权重与详情：")
            print("-" * 80)

            for v in sorted(vehicle_trips.keys()):
                print(f"\n🚍 车辆 {v}：")
                print(f"    总方案数：{len(current_columns[v])} 个（初始列{len(initial_columns[v])}个，新列{len(current_columns[v])-len(initial_columns[v])}个）")
                print("    " + "-" * 70)
                print(f"    {'方案编号':<8} {'类型':<10} {'权重λ':<12} {'总成本(元)':<12} {'有效路径占比':<15}")
                print("    " + "-" * 70)

                # 收集所有方案的信息，按权重从高到低排序
                scheme_info = []
                for i, scheme in enumerate(current_columns[v]):
                    weight = final_x[v][i].X
                    cost = scheme["c_omega"]
                    scheme_type = "初始列" if i < len(initial_columns[v]) else "新列"
                    valid_ratio = scheme.get("valid_path_ratio", "未知")
                    scheme_info.append((-weight, i, scheme_type, weight, cost, valid_ratio, scheme))

                # 按权重降序排序
                scheme_info.sort()

                # 打印所有方案
                for neg_weight, i, scheme_type, weight, cost, valid_ratio, scheme in scheme_info:
                    weight_str = f"{weight:.6f}"
                    cost_str = f"{cost:.2f}"
                    # 高亮权重>0的方案
                    if weight > EPS:
                        print(f"    ✅ {i:<6} {scheme_type:<10} {weight_str:<12} {cost_str:<12} {valid_ratio:<15}")
                    else:
                        print(f"    ⚪ {i:<6} {scheme_type:<10} {weight_str:<12} {cost_str:<12} {valid_ratio:<15}")

                # 打印权重>0的方案的详细调度记录
                print("\n    📋 权重>0的方案详细信息：")
                print("    " + "-" * 70)
                for neg_weight, i, scheme_type, weight, cost, valid_ratio, scheme in scheme_info:
                    if weight <= EPS:
                        continue

                    detail = scheme["scheme_detail"]
                    print(f"\n    【方案{i}】{scheme_type} | 权重={weight:.6f} | 总成本={cost:.2f}元 | 有效路径={valid_ratio}")
                    print("    " + "-" * 60)

                    # 空detail容错处理
                    if not detail:
                        print("    ⚠️ 该方案为子问题生成的新列，暂未生成详细调度记录")
                        print("    " + "-" * 60)
                        continue

                    # 统计信息
                    charge_count = len([d for d in detail if d.get("type") == "charge"])
                    no_charge_count = len([d for d in detail if d.get("type") == "no_charge"])
                    fast_count = len([d for d in detail if d.get("type") == "charge" and d.get("mode") == "快充"])
                    slow_count = len([d for d in detail if d.get("type") == "charge" and d.get("mode") == "慢充"])

                    # 取出物理绝对成本（假设 100% 发生时的定价）
                    total_unscaled_elec = sum(d.get("elec_cost", 0.0) for d in detail if d.get("type") == "charge")
                    total_unscaled_drive = sum(d.get("drive_cost", 0.0) for d in detail)

                    # 取出按到达概率计算的期望成本（加和严丝合缝等于方案总成本）
                    total_exp_elec = sum(d.get("exp_elec_cost", 0.0) for d in detail)
                    total_exp_wear = sum(d.get("wear_cost", 0.0) for d in detail)
                    total_exp_drive = sum(d.get("exp_drive_cost", 0.0) for d in detail)

                    print(f"    总车次：{len(vehicle_trips[v])} 个")
                    print(f"    充电次数：{charge_count} 次（快充{fast_count}次，慢充{slow_count}次）")
                    print(f"    不充电次数：{no_charge_count} 次")
                    print(f"    --- 期望成本拆解 (完美加和 = 总成本 {cost:.2f} 元) ---")
                    print(f"    总期望电费：{total_exp_elec:.2f} 元")
                    print(f"    总期望损耗：{total_exp_wear:.2f} 元")
                    print(f"    总期望空驶：{total_exp_drive:.2f} 元")
                    print(f"    (注：上述期望值由【实际标价 × 路径到达概率】计算得出)")

                    # 详细调度记录
                    print("\n    详细调度时间轴（倒推）：")
                    print("    " + "-" * 60)
                    for d in detail:
                        d_type = d.get("type", "unknown")
                        if d_type == "discharge":
                            print(f"    初始状态运营期：")
                            print(f"      初始放电产生的期望损耗：{d.get('wear_cost', 0.0):.2f} 元")
                        elif d_type == "charge":
                            print(f"    第{d.get('stage', 0) + 1}车次后：决定【充电】")
                            print(f"      前往站点：充电站 {d.get('station', '未知')} | {d.get('mode', '未知')}")
                            print(
                                f"      充电目标：{d.get('charge_amount', 0.0):.1f}% | 占桩时长：{d.get('charge_time', 0.0):.0f}分钟")
                            print(
                                f"      时间窗口：第 {d.get('charge_start_minute', 0.0):.0f} ~ {d.get('charge_end_minute', 0.0):.0f} 分钟")
                            print(
                                f"      此动作标价 -> 电费：{d.get('elec_cost', 0.0):.2f}元 | 空驶：{d.get('drive_cost', 0.0):.2f}元")
                            print(
                                f"      计入期望 -> 电费：{d.get('exp_elec_cost', 0.0):.2f}元 | 空驶：{d.get('exp_drive_cost', 0.0):.2f}元 | 损耗：{d.get('wear_cost', 0.0):.2f}元")
                        elif d_type == "no_charge":
                            print(f"    第{d.get('stage', 0) + 1}车次后：决定【不充电】(空驶去下一站)")
                            print(f"      此动作标价 -> 空驶：{d.get('drive_cost', 0.0):.2f}元")
                            print(
                                f"      计入期望 -> 空驶：{d.get('exp_drive_cost', 0.0):.2f}元 | 损耗：{d.get('wear_cost', 0.0):.2f}元")
                        print("    " + "-" * 60)

        # 返回列生成结果
        return {
            "status": "success",
            "total_cost": final_model.ObjVal,
            "vehicle_schemes": current_columns,
            "weights": {v: [final_x[v][i].X for i in range(len(current_columns[v]))] for v in vehicle_trips.keys()},
            "iter_count": iter_count,
            "pi": final_pi,
            "sigma": final_sigma
        }

    except Exception as e:
        print(f"\n❌ 程序运行失败：{str(e)}")
        import traceback
        traceback.print_exc()  # 打印详细错误栈，方便排查
        return {"status": "failed", "error": str(e)}

if __name__ == "__main__":
        main()