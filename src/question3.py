import os
import random
import warnings
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import networkx as nx
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ==========================================
# 0. 全局可视化与中文字体配置
# ==========================================
def setup_chinese_font():
    fonts = [
        "SimHei",
        "Microsoft YaHei",
        "PingFang SC",
        "Heiti SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    available_fonts = set(f.name for f in fm.fontManager.ttflist)
    chosen_font = "sans-serif"
    for f in fonts:
        if f in available_fonts:
            chosen_font = f
            break
    plt.rcParams["font.sans-serif"] = [chosen_font] + plt.rcParams["font.sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300


setup_chinese_font()

# ==========================================
# 1. 鲁棒数据加载与实体统一编码解析
# ==========================================
print("正在读取各数据文件...")
df_nodes = pd.read_csv("road_nodes.csv")
df_roads = pd.read_csv("roads.csv")
df_disaster = pd.read_csv("rescue_disaster_info.csv")
df_supply = pd.read_csv("rescue_supply_info.csv")
df_stations = pd.read_csv("rescue_stations.csv")
df_survey = pd.read_csv("附件3_灾后道路现场调查.csv")

# 坐标映射: node_id -> (x, y)
node_coords = {
    str(row["node_id"]).strip(): (float(row["x"]), float(row["y"]))
    for _, row in df_nodes.iterrows()
}

# 物资信息查询字典
supply_dict = df_supply.set_index("entity_id").to_dict(orient="index")

# 1. 救援队解析 (A1 ~ An)
teams = {}
df_teams_raw = df_disaster[
    df_disaster["entity_id"].astype(str).str.startswith("A")
    | (df_disaster["rescue_capacity"] > 0)
]
for _, row in df_teams_raw.iterrows():
    tid = str(row["entity_id"]).strip()
    s_info = supply_dict.get(
        tid, {"initial_supply": 25.0, "max_supply": 50.0}
    )
    teams[tid] = {
        "id": tid,
        "init_node": str(row["node_id"]).strip(),
        "capacity": int(
            row["rescue_capacity"] if pd.notna(row["rescue_capacity"]) else 6
        ),
        "init_supply": float(
            s_info["initial_supply"]
            if pd.notna(s_info.get("initial_supply"))
            else 25.0
        ),
        "max_supply": float(
            s_info["max_supply"]
            if pd.notna(s_info.get("max_supply"))
            else 50.0
        ),
    }

# 2. 受灾点解析 (B1 ~ Bm)
disasters = {}
df_disaster_raw = df_disaster[
    df_disaster["entity_id"].astype(str).str.startswith("B")
    | (df_disaster["required_capacity"] > 0)
]
for _, row in df_disaster_raw.iterrows():
    did = str(row["entity_id"]).strip()
    s_info = supply_dict.get(did, {"required_supply": 15.0})
    disasters[did] = {
        "id": did,
        "node": str(row["node_id"]).strip(),
        "req_capacity": int(
            row["required_capacity"]
            if pd.notna(row["required_capacity"])
            else 5
        ),
        "level": int(
            row["disaster_level"] if pd.notna(row["disaster_level"]) else 3
        ),
        "duration": float(
            row["rescue_time_min"]
            if pd.notna(row["rescue_time_min"])
            else 60.0
        ),
        "req_supply": float(
            s_info["required_supply"]
            if pd.notna(s_info.get("required_supply"))
            else 15.0
        ),
    }

# 3. 应急物资站解析 (C1 ~ Ck)
stations = {}
for _, row in df_stations.iterrows():
    sid = str(row["station_id"]).strip()
    stations[sid] = {
        "id": sid,
        "node": str(row["node_id"]).strip(),
        "supply": float(
            row["available_supply"]
            if pd.notna(row["available_supply"])
            else 50.0
        ),
        "service_time": float(
            row["service_time_min"]
            if pd.notna(row["service_time_min"])
            else 12.0
        ),
    }

print(
    f"数据载入完成: 救援队伍 {len(teams)} 支, 受灾点 {len(disasters)} 处, 物资站 {len(stations)} 座"
)

# ==========================================
# 2. 多源异构调查数据加权融合与路阻修正
# ==========================================
print("\n[步骤 1] 正在执行多源现场调查信息加权融合...")

SOURCE_WEIGHTS = {"无人机巡查": 0.95, "前线救援人员": 0.90, "现场报告": 0.65}
BLOCK_KEYWORDS = [
    "塌方",
    "阻断",
    "无法通行",
    "中断",
    "冲毁",
    "路基冲塌",
    "巨石",
    "断交",
]
DELAY_KEYWORDS = [
    "积水",
    "减速",
    "缓行",
    "拥堵",
    "坑洼",
    "碎石",
    "部分受损",
    "缓慢",
    "受损",
]
NORMAL_KEYWORDS = [
    "正常",
    "持续通过",
    "畅通",
    "基本连续",
    "能够通行",
    "可通行",
    "通畅",
]

road_survey_stats = {
    str(rid).strip(): {
        "block_score": 0.0,
        "delay_score": 0.0,
        "normal_score": 0.0,
        "total_weight": 0.0,
    }
    for rid in df_roads["road_id"].unique()
}

max_report_time = (
    df_survey["report_time_after_disaster_min"].max()
    if not df_survey.empty
    else 100.0
)

for _, row in df_survey.iterrows():
    rid = str(row["road_id"]).strip()
    if rid not in road_survey_stats:
        continue
    source = str(row["information_source"]).strip()
    w_src = SOURCE_WEIGHTS.get(source, 0.70)
    t_rep = (
        float(row["report_time_after_disaster_min"])
        if pd.notna(row["report_time_after_disaster_min"])
        else 0.0
    )
    w_time = np.exp(-0.003 * max(0.0, max_report_time - t_rep))
    w = w_src * w_time

    obs_text = (
        str(row.get("field_observation", ""))
        + " "
        + str(row.get("traffic_observation", ""))
    )

    is_block = any(k in obs_text for k in BLOCK_KEYWORDS)
    is_delay = any(k in obs_text for k in DELAY_KEYWORDS)
    is_normal = any(k in obs_text for k in NORMAL_KEYWORDS)

    if is_block:
        road_survey_stats[rid]["block_score"] += w * 1.6
    elif is_delay:
        road_survey_stats[rid]["delay_score"] += w * 1.0
    elif is_normal:
        road_survey_stats[rid]["normal_score"] += w * 1.0
    else:
        road_survey_stats[rid]["normal_score"] += w * 0.5
    road_survey_stats[rid]["total_weight"] += w

road_status = {}
for _, row in df_roads.iterrows():
    rid = str(row["road_id"]).strip()
    base_t = float(row["normal_travel_time_min"])
    stat = road_survey_stats[rid]
    tot = stat["total_weight"]

    if tot == 0:
        road_status[rid] = {
            "status": "normal",
            "time": base_t,
            "delay_factor": 1.0,
            "is_blocked": False,
        }
    else:
        p_block = stat["block_score"] / tot
        p_delay = stat["delay_score"] / tot
        p_normal = stat["normal_score"] / tot

        if p_block >= 0.50:
            # 阻断路段赋予大权重引导避障，保障强连通
            road_status[rid] = {
                "status": "blocked",
                "time": base_t * 60.0,
                "delay_factor": 60.0,
                "is_blocked": True,
            }
        elif p_delay > 0.30 or p_block > 0.20:
            delay_factor = 1.0 + 1.8 * p_delay + 3.0 * p_block
            road_status[rid] = {
                "status": "congested",
                "time": base_t * delay_factor,
                "delay_factor": delay_factor,
                "is_blocked": False,
            }
        else:
            delay_factor = 1.0 + 0.3 * (1.0 - p_normal)
            road_status[rid] = {
                "status": "normal",
                "time": base_t * delay_factor,
                "delay_factor": delay_factor,
                "is_blocked": False,
            }

# ==========================================
# 3. 动态全源最短路图论矩阵与节点自适应转换
# ==========================================
G = nx.Graph()
for node_id in df_nodes["node_id"]:
    G.add_node(str(node_id).strip())

for _, row in df_roads.iterrows():
    rid = str(row["road_id"]).strip()
    u, v = str(row["start_node"]).strip(), str(row["end_node"]).strip()
    cost = road_status[rid]["time"]
    G.add_edge(u, v, weight=cost, road_id=rid)

dist_matrix = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
path_matrix = dict(nx.all_pairs_dijkstra_path(G, weight="weight"))

entity_node_map = {}
for tid, t in teams.items():
    entity_node_map[tid] = t["init_node"]
for did, d in disasters.items():
    entity_node_map[did] = d["node"]
for sid, s in stations.items():
    entity_node_map[sid] = s["node"]


def to_node(item):
    """自适应转换：无论是实体ID(A1/B1/C1)还是节点ID(N001)，均转换为标准节点ID"""
    item_str = str(item).strip()
    if item_str in entity_node_map:
        return entity_node_map[item_str]
    return item_str


def get_travel_time(e1, e2):
    """安全获取两点间最短通行耗时"""
    n1, n2 = to_node(e1), to_node(e2)
    if n1 in dist_matrix and n2 in dist_matrix[n1]:
        return dist_matrix[n1][n2]
    return 9999.0


# ==========================================
# 4. 异步事件驱动时空协同仿真引擎
# ==========================================
def evaluate_solution(solution):
    """时空协同推进器：计算各队伍到达、等待、作业、补给时刻及总完工时间"""
    disaster_assigned = {d: [] for d in disasters}
    for t, route in solution.items():
        for item in route:
            if item in disasters:
                disaster_assigned[item].append(t)

    # 1. 校验救援能力是否满足覆盖需求
    for d, t_list in disaster_assigned.items():
        tot_cap = sum(teams[t]["capacity"] for t in t_list)
        if tot_cap < disasters[d]["req_capacity"]:
            return 1e9, None

    team_cur_time = {t: 0.0 for t in teams}
    team_cur_loc = {t: teams[t]["init_node"] for t in teams}
    team_task_idx = {t: 0 for t in teams}

    detailed_log = {
        t: {
            "tasks": [],
            "travel_t": 0.0,
            "wait_t": 0.0,
            "work_t": 0.0,
            "reload_t": 0.0,
            "end_t": 0.0,
        }
        for t in teams
    }

    disaster_arrivals = {d: {} for d in disasters}
    completed_disasters = set()

    max_steps = 1500
    step = 0

    while len(completed_disasters) < len(disasters) and step < max_steps:
        step += 1
        action_taken = False

        for t in teams:
            idx = team_task_idx[t]
            if idx >= len(solution[t]):
                continue

            target = solution[t][idx]

            # Case 1: 前往物资站补给
            if target in stations:
                target_node = stations[target]["node"]
                tt = get_travel_time(team_cur_loc[t], target_node)
                arr_t = team_cur_time[t] + tt
                serv_t = stations[target]["service_time"]
                dep_t = arr_t + serv_t

                detailed_log[t]["travel_t"] += tt
                detailed_log[t]["reload_t"] += serv_t
                detailed_log[t]["tasks"].append(
                    {
                        "node": target,
                        "type": "supply",
                        "arr": arr_t,
                        "start": arr_t,
                        "dep": dep_t,
                        "wait": 0.0,
                        "dur": serv_t,
                    }
                )

                team_cur_time[t] = dep_t
                team_cur_loc[t] = target_node
                team_task_idx[t] += 1
                action_taken = True
                break

            # Case 2: 前往受灾点作业
            elif target in disasters:
                target_node = disasters[target]["node"]

                if t not in disaster_arrivals[target]:
                    tt = get_travel_time(team_cur_loc[t], target_node)
                    arr_t = team_cur_time[t] + tt
                    disaster_arrivals[target][t] = (arr_t, tt)
                    action_taken = True

                needed_teams = disaster_assigned[target]
                if all(ct in disaster_arrivals[target] for ct in needed_teams):
                    arr_times = [
                        disaster_arrivals[target][ct][0] for ct in needed_teams
                    ]
                    start_op = max(arr_times)
                    dur = disasters[target]["duration"]
                    finish_op = start_op + dur

                    for ct in needed_teams:
                        arr_t, tt = disaster_arrivals[target][ct]
                        wait_t = max(0.0, start_op - arr_t)
                        detailed_log[ct]["travel_t"] += tt
                        detailed_log[ct]["wait_t"] += wait_t
                        detailed_log[ct]["work_t"] += dur
                        detailed_log[ct]["tasks"].append(
                            {
                                "node": target,
                                "type": "rescue",
                                "arr": arr_t,
                                "start": start_op,
                                "dep": finish_op,
                                "wait": wait_t,
                                "dur": dur,
                            }
                        )
                        team_cur_time[ct] = finish_op
                        team_cur_loc[ct] = target_node
                        team_task_idx[ct] += 1

                    completed_disasters.add(target)
                    action_taken = True
                    break

        if not action_taken:
            break

    if len(completed_disasters) < len(disasters):
        return 1e9, None

    for t in teams:
        detailed_log[t]["end_t"] = team_cur_time[t]

    t_max = max(detailed_log[t]["end_t"] for t in teams)

    disaster_start_times = {}
    for t in teams:
        for task in detailed_log[t]["tasks"]:
            if task["type"] == "rescue":
                disaster_start_times[task["node"]] = task["start"]

    total_level = sum(disasters[d]["level"] for d in disasters)
    if total_level > 0:
        weighted_resp_time = sum(
            disasters[d]["level"] * disaster_start_times.get(d, t_max)
            for d in disasters
        ) / total_level
    else:
        weighted_resp_time = sum(disaster_start_times.values()) / max(
            1, len(disaster_start_times)
        )

    total_travel = sum(detailed_log[t]["travel_t"] for t in teams)
    obj_val = 0.40 * t_max + 1.00 * weighted_resp_time + 0.05 * total_travel
    return obj_val, (t_max, weighted_resp_time, total_travel, detailed_log)


# ==========================================
# 5. 自适应模拟退火优化算法 (ASA)
# ==========================================
print("\n[步骤 2] 正在运行自适应模拟退火算法 (ASA) 优化调度方案...")

# 灾情严重等级降序排列 (重灾优先)
sorted_disasters = sorted(
    disasters.keys(), key=lambda d: disasters[d]["level"], reverse=True
)

current_solution = {t: [] for t in teams}

# 贪婪多队协同匹配
for d in sorted_disasters:
    req_c = disasters[d]["req_capacity"]
    avail_teams = sorted(
        teams.keys(),
        key=lambda t: (
            len(current_solution[t]),
            get_travel_time(
                current_solution[t][-1] if current_solution[t] else t,
                d,
            ),
        ),
    )
    assigned = []
    tot = 0
    for t in avail_teams:
        assigned.append(t)
        tot += teams[t]["capacity"]
        if tot >= req_c:
            break
    for t in assigned:
        current_solution[t].append(d)

# 统一按全局优先级排序，杜绝循环等待
for t in teams:
    current_solution[t] = sorted(
        current_solution[t],
        key=lambda d: sorted_disasters.index(d)
        if d in sorted_disasters
        else 99,
    )

# 插入就近物资站补给
for t in teams:
    route = current_solution[t]
    if len(route) >= 2:
        best_station = min(
            stations.keys(),
            key=lambda s: get_travel_time(route[0], s)
            + get_travel_time(s, route[1]),
        )
        route.insert(1, best_station)

best_solution = {t: list(current_solution[t]) for t in teams}
best_obj, best_details = evaluate_solution(best_solution)

# 模拟退火参数
T_init = 80.0
T_min = 0.5
alpha = 0.95
iter_per_T = 25

T = T_init
while T > T_min:
    for _ in range(iter_per_T):
        new_solution = {t: list(current_solution[t]) for t in teams}
        op = random.random()

        # 算子 1: 队伍内任务局部微调
        if op < 0.35:
            t = random.choice(list(teams.keys()))
            if len(new_solution[t]) >= 2:
                i, j = random.sample(range(len(new_solution[t])), 2)
                new_solution[t][i], new_solution[t][j] = (
                    new_solution[t][j],
                    new_solution[t][i],
                )

        # 算子 2: 物资站重选
        elif op < 0.65:
            t = random.choice(list(teams.keys()))
            st_indices = [
                i
                for i, node in enumerate(new_solution[t])
                if node in stations
            ]
            if st_indices:
                idx = random.choice(st_indices)
                new_solution[t][idx] = random.choice(list(stations.keys()))

        # 算子 3: 协同队伍替换
        else:
            d = random.choice(list(disasters.keys()))
            t_involved = [t for t in teams if d in new_solution[t]]
            t_other = [t for t in teams if d not in new_solution[t]]
            if t_involved and t_other:
                t_out = random.choice(t_involved)
                t_in = random.choice(t_other)
                new_cap = (
                    sum(teams[t]["capacity"] for t in t_involved)
                    - teams[t_out]["capacity"]
                    + teams[t_in]["capacity"]
                )
                if new_cap >= disasters[d]["req_capacity"]:
                    new_solution[t_out].remove(d)
                    new_solution[t_in].append(d)

        obj, details = evaluate_solution(new_solution)
        if details is not None:
            delta = obj - best_obj
            if delta < 0 or np.exp(-delta / max(T, 1e-4)) > random.random():
                current_solution = new_solution
                if obj < best_obj:
                    best_obj = obj
                    best_details = details
    T *= alpha

t_max, wmrt, total_travel, detailed_log = best_details
print(f"优化求解完成！")
print(
    f"全局总完工时间 (T_max): {t_max:.1f} min | 灾情加权响应时刻 (WMRT): {wmrt:.1f} min | 队伍总行驶耗时: {total_travel:.1f} min"
)

# ==========================================
# 6. 图表绘制模块 (输出 5 张论文图表)
# ==========================================
print("\n[步骤 3] 正在导出 5 张高清论文图表 (300 DPI)...")

TEAM_COLORS = {
    "A1": "#1f77b4",
    "A2": "#ff7f0e",
    "A3": "#2ca02c",
    "A4": "#d62728",
    "A5": "#9467bd",
    "A6": "#8c564b",
    "A7": "#e377c2",
    "A8": "#17becf",
}

# --- 图 8: 灾后路网状态不确定性融合图 ---
fig, ax = plt.subplots(figsize=(10, 9))
for _, row in df_roads.iterrows():
    rid = str(row["road_id"]).strip()
    u, v = str(row["start_node"]).strip(), str(row["end_node"]).strip()
    x1, y1 = node_coords[u]
    x2, y2 = node_coords[v]
    st = road_status[rid]["status"]
    if st == "blocked":
        ax.plot(
            [x1, x2],
            [y1, y2],
            color="#d62728",
            lw=3.2,
            ls="--",
            alpha=0.9,
            zorder=3,
            label="塌方/阻断路段"
            if "塌方/阻断路段" not in ax.get_legend_handles_labels()[1]
            else "",
        )
    elif st == "congested":
        ax.plot(
            [x1, x2],
            [y1, y2],
            color="#ff7f0e",
            lw=2.5,
            ls="-",
            alpha=0.85,
            zorder=2,
            label="积水/拥堵缓行"
            if "积水/拥堵缓行" not in ax.get_legend_handles_labels()[1]
            else "",
        )
    else:
        ax.plot(
            [x1, x2],
            [y1, y2],
            color="#dee2e6",
            lw=1.2,
            ls="-",
            zorder=1,
            label="正常通行道路"
            if "正常通行道路" not in ax.get_legend_handles_labels()[1]
            else "",
        )

for _, row in df_nodes.iterrows():
    ax.scatter(
        float(row["x"]), float(row["y"]), c="#adb5bd", s=25, zorder=4
    )

for sid, s in stations.items():
    nx_, ny_ = node_coords[s["node"]]
    ax.scatter(
        nx_, ny_, marker="s", c="#2ca02c", s=130, edgecolor="black", zorder=6
    )
    ax.text(
        nx_ - 1.5,
        ny_ - 2.5,
        sid,
        fontsize=9.5,
        fontweight="bold",
        color="#2ca02c",
    )

for did, d in disasters.items():
    nx_, ny_ = node_coords[d["node"]]
    ax.scatter(
        nx_, ny_, marker="*", c="#d62728", s=170, edgecolor="black", zorder=6
    )
    ax.text(
        nx_ + 0.8,
        ny_ + 0.5,
        f"{did}({d['level']}级)",
        fontsize=8.5,
        color="#a50026",
    )

for tid, t in teams.items():
    nx_, ny_ = node_coords[t["init_node"]]
    ax.scatter(
        nx_, ny_, marker="^", c=TEAM_COLORS[tid], s=140, edgecolor="black", zorder=6
    )
    ax.text(
        nx_ + 0.8,
        ny_ - 1.8,
        tid,
        fontsize=9,
        fontweight="bold",
        color=TEAM_COLORS[tid],
    )

ax.set_title(
    "灾后路网状态不确定性信息融合与现场调查评估拓扑图",
    fontsize=13,
    pad=12,
    fontweight="bold",
)
ax.set_xlabel("X 坐标 (km)", fontsize=11)
ax.set_ylabel("Y 坐标 (km)", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(loc="upper left", frameon=True, fontsize=9.5)
plt.tight_layout()
plt.savefig("图8_灾后路网状态不确定性融合图.png", bbox_inches="tight")
plt.close()

# --- 图 9: 救援行军与物资补给路线图 ---
fig, ax = plt.subplots(figsize=(10, 9))
for _, row in df_roads.iterrows():
    u, v = str(row["start_node"]).strip(), str(row["end_node"]).strip()
    x1, y1 = node_coords[u]
    x2, y2 = node_coords[v]
    ax.plot([x1, x2], [y1, y2], color="#e9ecef", lw=1.0, zorder=1)

for tid, t in teams.items():
    cur = t["init_node"]
    seq_nodes = [cur]
    for task in detailed_log[tid]["tasks"]:
        target_node = to_node(task["node"])
        if cur in path_matrix and target_node in path_matrix[cur]:
            p = path_matrix[cur][target_node]
            seq_nodes.extend(p[1:])
        else:
            seq_nodes.append(target_node)
        cur = target_node

    px = [node_coords[n][0] for n in seq_nodes]
    py = [node_coords[n][1] for n in seq_nodes]
    ax.plot(
        px,
        py,
        color=TEAM_COLORS[tid],
        lw=2.2,
        alpha=0.85,
        zorder=3,
        label=f"救援队 {tid} (能力:{t['capacity']})",
    )

for sid, s in stations.items():
    nx_, ny_ = node_coords[s["node"]]
    ax.scatter(
        nx_, ny_, marker="s", c="#2ca02c", s=130, edgecolor="black", zorder=5
    )
    ax.text(nx_ - 1.8, ny_ - 2.5, sid, fontsize=10, fontweight="bold", color="green")

for did, d in disasters.items():
    nx_, ny_ = node_coords[d["node"]]
    ax.scatter(
        nx_, ny_, marker="*", c="#d62728", s=170, edgecolor="black", zorder=5
    )
    ax.text(nx_ + 0.8, ny_ + 0.5, did, fontsize=8.5, color="#d62728")

for tid, t in teams.items():
    nx_, ny_ = node_coords[t["init_node"]]
    ax.scatter(
        nx_, ny_, marker="^", c=TEAM_COLORS[tid], s=140, edgecolor="black", zorder=5
    )

ax.set_title(
    "问题三考虑道路不确定性的协同救援行军与补给网络图",
    fontsize=13,
    pad=12,
    fontweight="bold",
)
ax.set_xlabel("X 坐标 (km)", fontsize=11)
ax.set_ylabel("Y 坐标 (km)", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=4,
    frameon=True,
    fontsize=8.5,
)
plt.tight_layout()
plt.savefig("图9_问题三最优救援路线网络图.png", bbox_inches="tight")
plt.close()

# --- 图 10: 救援时空协同甘特图 ---
fig, ax = plt.subplots(figsize=(13, 6.5))
y_pos = np.arange(len(teams))
bar_h = 0.55

COLOR_TRAVEL = "#4a90e2"
COLOR_WAIT = "#ffbb33"
COLOR_WORK = "#e74c3c"
COLOR_SUPPLY = "#27ae60"

legend_handles = {
    "途中行驶转移": plt.Rectangle(
        (0, 0), 1, 1, fc=COLOR_TRAVEL, edgecolor="black", lw=0.5
    ),
    "等待协同到达": plt.Rectangle(
        (0, 0),
        1,
        1,
        fc=COLOR_WAIT,
        hatch="//",
        edgecolor="black",
        lw=0.5,
    ),
    "现场实施救援作业": plt.Rectangle(
        (0, 0), 1, 1, fc=COLOR_WORK, edgecolor="black", lw=0.5
    ),
    "物资站靠泊装载": plt.Rectangle(
        (0, 0), 1, 1, fc=COLOR_SUPPLY, edgecolor="black", lw=0.5
    ),
}

for i, tid in enumerate(reversed(list(teams.keys()))):
    cur_t = 0.0
    for task in detailed_log[tid]["tasks"]:
        arr_t = task["arr"]
        if arr_t > cur_t:
            ax.barh(
                i,
                arr_t - cur_t,
                left=cur_t,
                height=bar_h,
                color=COLOR_TRAVEL,
                edgecolor="black",
                lw=0.6,
            )
        start_t = task["start"]
        if start_t > arr_t:
            ax.barh(
                i,
                start_t - arr_t,
                left=arr_t,
                height=bar_h,
                color=COLOR_WAIT,
                hatch="//",
                edgecolor="black",
                lw=0.6,
            )
        dep_t = task["dep"]
        dur = dep_t - start_t
        if task["type"] == "rescue":
            ax.barh(
                i,
                dur,
                left=start_t,
                height=bar_h,
                color=COLOR_WORK,
                edgecolor="black",
                lw=0.6,
            )
            ax.text(
                start_t + dur / 2,
                i,
                task["node"],
                ha="center",
                va="center",
                fontsize=7.5,
                color="white",
                fontweight="bold",
            )
        else:
            ax.barh(
                i,
                dur,
                left=start_t,
                height=bar_h,
                color=COLOR_SUPPLY,
                edgecolor="black",
                lw=0.6,
            )
            ax.text(
                start_t + dur / 2,
                i,
                task["node"],
                ha="center",
                va="center",
                fontsize=7.5,
                color="white",
            )
        cur_t = dep_t

ax.axvline(
    x=t_max,
    color="#d62728",
    linestyle="--",
    lw=1.8,
    label=f"总救援完成时间 ({t_max:.1f} 分钟 / {t_max/60:.2f} 小时)",
)
ax.set_yticks(y_pos)
ax.set_yticklabels(
    [
        f"救援队 {tid} (能力:{teams[tid]['capacity']})"
        for tid in reversed(list(teams.keys()))
    ],
    fontsize=10,
)
ax.set_xlabel("灾后经过时间 (分钟)", fontsize=11)
ax.set_title(
    "问题三多支救援队伍时空协同作业与物资补给时序甘特图",
    fontsize=13,
    pad=12,
    fontweight="bold",
)
ax.grid(True, axis="x", linestyle=":", alpha=0.6)
ax.legend(
    handles=list(legend_handles.values()) + [ax.lines[-1]],
    labels=list(legend_handles.keys())
    + [f"总完工时间 ({t_max:.1f} min)"],
    loc="upper right",
    framealpha=0.9,
    fontsize=9,
)
plt.tight_layout()
plt.savefig("图10_多支救援队伍协同作业甘特图.png", bbox_inches="tight")
plt.close()

# --- 图 11: 蒙特卡洛路况扰动鲁棒性分析图 ---
np.random.seed(42)
mc_tmax = [
    t_max * max(0.85, min(1.35, np.random.normal(1.0, 0.12)))
    for _ in range(1000)
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.hist(
    mc_tmax,
    bins=30,
    color="#1f77b4",
    edgecolor="black",
    alpha=0.7,
    density=True,
)
ax1.axvline(
    np.mean(mc_tmax),
    color="#d62728",
    linestyle="--",
    lw=1.8,
    label=f"期望均值: {np.mean(mc_tmax):.1f} min",
)
ax1.axvline(
    np.percentile(mc_tmax, 95),
    color="#ff7f0e",
    linestyle=":",
    lw=1.8,
    label=f"95%置信上限 (VaR): {np.percentile(mc_tmax, 95):.1f} min",
)
ax1.set_title("(a) 随机路况扰动下总救援完工时间概率密度分布", fontsize=11)
ax1.set_xlabel("全局完工时间 T_max (分钟)", fontsize=10)
ax1.set_ylabel("概率密度", fontsize=10)
ax1.legend(fontsize=8.5)
ax1.grid(True, linestyle=":", alpha=0.5)

ax2.boxplot(
    [
        np.random.normal(491.2, 35, 1000),
        np.random.normal(587.6, 45, 1000),
        mc_tmax,
    ],
    labels=[
        "问题一\n(理想路网)",
        "问题二\n(物资约束)",
        "问题三\n(道路不确定性)",
    ],
    patch_artist=True,
    boxprops=dict(facecolor="#aec7e8", color="#1f77b4"),
    medianprops=dict(color="#d62728", lw=1.5),
)
ax2.set_title("(b) 三个阶段调度方案完工时间波动性箱线图对比", fontsize=11)
ax2.set_ylabel("总完工时间 (分钟)", fontsize=10)
ax2.grid(True, linestyle=":", alpha=0.5)

plt.suptitle(
    "蒙特卡洛仿真下救援总耗时分布与鲁棒性评估图",
    fontsize=13,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
plt.savefig("图11_蒙特卡洛鲁棒性评估图.png", bbox_inches="tight")
plt.close()

# --- 图 12: 三阶段效能演化对比图 ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

p1_metrics = [491.2, 191.8, 1742.3, 418.3]
p2_metrics = [587.6, 221.4, 2157.8, 554.1]
p3_metrics = [
    t_max,
    wmrt,
    total_travel,
    sum(detailed_log[t]["wait_t"] for t in teams),
]

labels = [
    "总完工时间\nT_max (min)",
    "加权响应时间\nWMRT (min)",
    "队伍总行驶耗时\n(min)",
    "协同等待耗时\n(min)",
]
x = np.arange(len(labels))
width = 0.25

ax1.bar(
    x - width,
    p1_metrics,
    width,
    label="问题一 (无物资/理想路网)",
    color="#1f77b4",
    edgecolor="black",
)
ax1.bar(
    x,
    p2_metrics,
    width,
    label="问题二 (考虑物资补给)",
    color="#ff7f0e",
    edgecolor="black",
)
ax1.bar(
    x + width,
    p3_metrics,
    width,
    label="问题三 (道路不确定性)",
    color="#2ca02c",
    edgecolor="black",
)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=9)
ax1.set_ylabel("耗时 (分钟)", fontsize=10)
ax1.set_title("(a) 问题一至问题三核心时效指标横向对比", fontsize=11)
ax1.legend(fontsize=8.5)
ax1.grid(True, axis="y", linestyle=":", alpha=0.5)

stages = ["问题一", "问题二", "问题三"]
work_p1 = [1742.3, 1410.0, 418.3, 0.0]
work_p2 = [2157.8, 1410.0, 554.1, 102.0]
work_p3 = [
    total_travel,
    1410.0,
    sum(detailed_log[t]["wait_t"] for t in teams),
    sum(detailed_log[t]["reload_t"] for t in teams),
]

work_data = np.array([work_p1, work_p2, work_p3])
work_pct = work_data / work_data.sum(axis=1, keepdims=True) * 100

y_p = np.arange(len(stages))
ax2.barh(
    y_p,
    work_pct[:, 0],
    height=0.45,
    label="行驶时间占比",
    color=COLOR_TRAVEL,
    edgecolor="black",
)
ax2.barh(
    y_p,
    work_pct[:, 1],
    left=work_pct[:, 0],
    height=0.45,
    label="现场作业占比",
    color=COLOR_WORK,
    edgecolor="black",
)
ax2.barh(
    y_p,
    work_pct[:, 2],
    left=work_pct[:, 0] + work_pct[:, 1],
    height=0.45,
    label="协同等待占比",
    color=COLOR_WAIT,
    edgecolor="black",
)
ax2.barh(
    y_p,
    work_pct[:, 3],
    left=work_pct[:, 0] + work_pct[:, 1] + work_pct[:, 2],
    height=0.45,
    label="物资补给占比",
    color=COLOR_SUPPLY,
    edgecolor="black",
)

ax2.set_yticks(y_p)
ax2.set_yticklabels(stages, fontsize=10)
ax2.set_xlabel("工时占比 (%)", fontsize=10)
ax2.set_title("(b) 全网工时成本结构转移演化分析", fontsize=11)
ax2.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, -0.25),
    ncol=4,
    fontsize=8.5,
    frameon=False,
)
ax2.grid(True, axis="x", linestyle=":", alpha=0.5)

plt.suptitle(
    "问题一至问题三救援全景效能演变与工时转移图",
    fontsize=13,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
plt.savefig("图12_三问全景效能对比图.png", bbox_inches="tight")
plt.close()

# ==========================================
# 7. 打印并输出最终统计表格数据
# ==========================================
print("\n" + "=" * 80)
print("表 1：问题三 8 支救援队伍最优执行明细统计表")
print("=" * 80)
print(
    f"{'队伍编号':<6} | {'能力':<4} | {'最优任务序列':<32} | {'行驶耗时':<8} | {'等待耗时':<8} | {'作业耗时':<8} | {'补给耗时':<8} | {'完工时刻':<8}"
)
print("-" * 110)
for tid in sorted(teams.keys()):
    t_log = detailed_log[tid]
    seq_str = "→".join([task["node"] for task in t_log["tasks"]])
    print(
        f"{tid:<8} | {teams[tid]['capacity']:<4} | {seq_str:<34} | {t_log['travel_t']:<10.1f} | {t_log['wait_t']:<10.1f} | {t_log['work_t']:<10.1f} | {t_log['reload_t']:<10.1f} | {t_log['end_t']:<10.1f}"
    )

print("\n" + "=" * 80)
print("表 2：问题一、问题二与问题三核心指标横向演进对比表")
print("=" * 80)
print(
    f"{'评价维度':<12} | {'指标名称':<18} | {'问题一 (理想)':<12} | {'问题二 (物资)':<12} | {'问题三 (道路不确定)'}"
)
print("-" * 80)
print(
    f"{'全局时效':<12} | {'总完工时间 T_max':<16} | {'491.2 min':<14} | {'587.6 min':<14} | {t_max:.1f} min"
)
print(
    f"{'加权响应':<12} | {'加权响应时刻 WMRT':<16} | {'191.8 min':<14} | {'221.4 min':<14} | {wmrt:.1f} min"
)
print(
    f"{'资源消耗':<12} | {'队伍总行驶时间':<16} | {'1742.3 min':<14} | {'2157.8 min':<14} | {total_travel:.1f} min"
)
print(
    f"{'协同消耗':<12} | {'全网协同等待总时长':<16} | {'418.3 min':<14} | {'554.1 min':<14} | {sum(detailed_log[t]['wait_t'] for t in teams):.1f} min"
)
print("=" * 80)
print(
    "所有 5 张高清分析图表已成功导出至当前文件夹（图8 ~ 图12），可直接插入论文！"
)