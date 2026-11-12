# -*- coding: utf-8 -*-
"""
数学建模竞赛 - 问题一：灾后应急救援协同路径与调度优化 (全中文可视化)
支持系统: Windows (SimHei/YaHei), macOS (PingFang/Songti), Linux (Noto/Droid Sans)
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import itertools
import copy
import random

# -------------------------------------------------------------
# 0. 跨平台中文字体自适应配置
# -------------------------------------------------------------
plt.rcParams['font.sans-serif'] = [
    'Noto Sans CJK JP', 'Droid Sans Fallback', 'SimHei', 
    'Microsoft YaHei', 'PingFang SC', 'Heiti SC', 'WenQuanYi Micro Hei'
]
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# -------------------------------------------------------------
# 1. 数据加载与路网拓扑图构建
# -------------------------------------------------------------
nodes_df = pd.read_csv('road_nodes.csv')
roads_df = pd.read_csv('roads.csv')
info_df = pd.read_csv('rescue_disaster_info.csv')

# 构建加权无向图
G = nx.Graph()
for _, r in roads_df.iterrows():
    G.add_edge(r['start_node'], r['end_node'], 
               weight=r['normal_travel_time_min'], 
               distance=r['distance_km'])

# 基于 Dijkstra 计算全源最短通行时间与路径
dist_dict = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
path_dict = dict(nx.all_pairs_dijkstra_path(G, weight='weight'))

# 提取救援队与受灾点基础数据
teams = info_df[info_df['entity_type'] == 'rescue_team'].copy()
sites = info_df[info_df['entity_type'] == 'disaster_site'].copy()

team_caps = dict(zip(teams['entity_id'], teams['rescue_capacity']))
site_reqs = dict(zip(sites['entity_id'], sites['required_capacity']))
site_levels = dict(zip(sites['entity_id'], sites['disaster_level']))
site_times = dict(zip(sites['entity_id'], sites['rescue_time_min']))
entity_nodes = dict(zip(info_df['entity_id'], info_df['node_id']))

# 构造实体间的最短耗时矩阵
entities = list(teams['entity_id']) + list(sites['entity_id'])
time_mat = {e1: {e2: dist_dict[entity_nodes[e1]][entity_nodes[e2]] for e2 in entities} for e1 in entities}

# 枚举满足能力需求的可行队伍组合
possible_team_combos = {}
for s, req in site_reqs.items():
    combos = []
    # 单队施救
    for t, cap in team_caps.items():
        if cap >= req:
            combos.append((t,))
    # 双队协同施救
    for t1, t2 in itertools.combinations(team_caps.keys(), 2):
        if team_caps[t1] + team_caps[t2] >= req:
            combos.append((t1, t2))
    # 备选三队协同
    if len(combos) < 5:
        for t1, t2, t3 in itertools.combinations(team_caps.keys(), 3):
            if team_caps[t1] + team_caps[t2] + team_caps[t3] >= req:
                combos.append((t1, t2, t3))
    possible_team_combos[s] = combos

# -------------------------------------------------------------
# 2. 协同作业时序仿真器
# -------------------------------------------------------------
def simulate_schedule(routes):
    site_visited = {s: [] for s in site_reqs}
    for t, r in routes.items():
        for s in r:
            site_visited[s].append(t)
            
    for s, t_list in site_visited.items():
        if sum(team_caps[t] for t in t_list) < site_reqs[s]:
            return None, 1e9, f"能力不足: {s}"
            
    team_pos = {t: t for t in team_caps}
    team_avail = {t: 0.0 for t in team_caps}
    team_idx = {t: 0 for t in team_caps}
    
    site_completed = {s: False for s in site_reqs}
    site_start, site_end = {}, {}
    team_travel = {t: 0.0 for t in team_caps}
    team_wait = {t: 0.0 for t in team_caps}
    team_work = {t: 0.0 for t in team_caps}
    team_events = {t: [] for t in team_caps}
    
    max_steps, step = 300, 0
    while step < max_steps:
        step += 1
        if all(team_idx[t] >= len(routes[t]) for t in team_caps):
            break
            
        ready_sites = []
        for s in site_reqs:
            if site_completed[s]:
                continue
            assigned = site_visited[s]
            if all(team_idx[t] < len(routes[t]) and routes[t][team_idx[t]] == s for t in assigned):
                ready_sites.append(s)
                
        if not ready_sites:
            return None, 1e9, "死锁"
            
        best_site, best_start, best_arrivals = None, float('inf'), {}
        for s in ready_sites:
            assigned = site_visited[s]
            arrivals = {t: team_avail[t] + time_mat[team_pos[t]][s] for t in assigned}
            st = max(arrivals.values())
            priority_score = st - 6.0 * site_levels[s]
            if priority_score < best_start:
                best_start = priority_score
                best_site = s
                best_arrivals = arrivals
                
        s = best_site
        assigned = site_visited[s]
        actual_st = max(best_arrivals.values())
        end_st = actual_st + site_times[s]
        
        site_start[s], site_end[s], site_completed[s] = actual_st, end_st, True
        
        for t in assigned:
            prev = team_pos[t]
            tr = time_mat[prev][s]
            arr = best_arrivals[t]
            wt = actual_st - arr
            team_travel[t] += tr
            team_wait[t] += wt
            team_work[t] += site_times[s]
            
            team_events[t].append({
                'from': prev, 'to': s,
                'depart_time': team_avail[t], 'arrival_time': arr,
                'start_rescue_time': actual_st, 'end_rescue_time': end_st,
                'travel_time': tr, 'wait_time': wt, 'rescue_time': site_times[s]
            })
            team_pos[t] = s
            team_avail[t] = end_st
            team_idx[t] += 1
            
    if not all(site_completed.values()):
        return None, 1e9, "未完成"
        
    makespan = max(team_avail.values())
    w_resp = sum(site_levels[s] * site_start[s] for s in site_reqs) / sum(site_levels.values())
    total_tr = sum(team_travel.values())
    cost = w_resp * 1.0 + makespan * 0.4 + total_tr * 0.05
    
    results = {
        'cost': cost, 'makespan': makespan, 'weighted_response_time': w_resp,
        'avg_response_time': sum(site_start.values()) / len(site_reqs),
        'total_travel_time': total_tr, 'total_work_time': sum(team_work.values()),
        'total_wait_time': sum(team_wait.values()),
        'site_start': site_start, 'site_end': site_end,
        'team_avail': team_avail, 'team_travel': team_travel,
        'team_work': team_work, 'team_wait': team_wait,
        'team_events': team_events
    }
    return results, cost, "OK"

# -------------------------------------------------------------
# 3. 模拟退火优化算法
# -------------------------------------------------------------
def solve_rescue_problem(seed=42, max_iter=15000):
    random.seed(seed)
    np.random.seed(seed)
    
    def generate_init_routes():
        routes = {t: [] for t in team_caps}
        all_s = list(site_reqs.keys())
        random.shuffle(all_s)
        all_s.sort(key=lambda s: -site_levels[s])
        for s in all_s:
            avail_teams = sorted(team_caps.keys(), key=lambda t: len(routes[t]) + random.uniform(0, 0.5))
            assigned, cur_cap = [], 0
            for t in avail_teams:
                assigned.append(t)
                cur_cap += team_caps[t]
                if cur_cap >= site_reqs[s]:
                    break
            for t in assigned:
                routes[t].append(s)
        return routes

    best_routes, best_cost, best_res = None, 1e9, None
    for _ in range(500):
        r = generate_init_routes()
        res, cost, msg = simulate_schedule(r)
        if msg == "OK" and cost < best_cost:
            best_cost, best_routes, best_res = cost, r, res
            
    current_routes, current_cost = copy.deepcopy(best_routes), best_cost
    T, cooling_rate = 100.0, 0.9995
    
    for it in range(max_iter):
        new_routes = copy.deepcopy(current_routes)
        op = random.random()
        
        if op < 0.35:  # 队伍内倒序换位
            t = random.choice(list(team_caps.keys()))
            if len(new_routes[t]) >= 2:
                i, j = random.sample(range(len(new_routes[t])), 2)
                new_routes[t][i], new_routes[t][j] = new_routes[t][j], new_routes[t][i]
        elif op < 0.65:  # 更换受灾点协同队伍组合
            s = random.choice(list(site_reqs.keys()))
            teams_s = [t for t in team_caps if s in new_routes[t]]
            new_combo = random.choice(possible_team_combos[s])
            for t in teams_s:
                new_routes[t].remove(s)
            for t in new_combo:
                pos = random.randint(0, len(new_routes[t]))
                new_routes[t].insert(pos, s)
        else:  # 单任务重定位
            t = random.choice(list(team_caps.keys()))
            if len(new_routes[t]) >= 2:
                s = new_routes[t].pop(random.randrange(len(new_routes[t])))
                pos = random.randint(0, len(new_routes[t]))
                new_routes[t].insert(pos, s)
                
        res, cost, msg = simulate_schedule(new_routes)
        if msg == "OK":
            delta = cost - current_cost
            if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-4)):
                current_routes, current_cost = new_routes, cost
                if cost < best_cost:
                    best_cost, best_routes, best_res = cost, copy.deepcopy(new_routes), res
        T *= cooling_rate
        
    return best_routes, best_res, best_cost

opt_routes, opt_res, opt_cost = solve_rescue_problem()

# -------------------------------------------------------------
# 4. 生成高分辨率中文科研图表 (300 DPI)
# -------------------------------------------------------------
node_x = dict(zip(nodes_df['node_id'], nodes_df['x']))
node_y = dict(zip(nodes_df['node_id'], nodes_df['y']))

# 图 1: 救援网络空间路径拓扑图 (全中文)
fig, ax = plt.subplots(figsize=(12, 11), dpi=300)
for _, r in roads_df.iterrows():
    u, v = r['start_node'], r['end_node']
    ax.plot([node_x[u], node_x[v]], [node_y[u], node_y[v]], color='#d0d7de', lw=1.2, zorder=1, alpha=0.8)

team_colors = {
    'A1': '#1f77b4', 'A2': '#ff7f0e', 'A3': '#2ca02c', 'A4': '#d62728',
    'A5': '#9467bd', 'A6': '#8c564b', 'A7': '#e377c2', 'A8': '#17becf'
}

for t in team_caps:
    cur_node = entity_nodes[t]
    full_path = [cur_node]
    for s in opt_routes[t]:
        dest_node = entity_nodes[s]
        p = path_dict[cur_node][dest_node]
        full_path.extend(p[1:])
        cur_node = dest_node
    xs = [node_x[n] for n in full_path]
    ys = [node_y[n] for n in full_path]
    ax.plot(xs, ys, color=team_colors[t], lw=2.8, alpha=0.85, zorder=3, 
            label=f"救援队 {t} (路径: {'→'.join(opt_routes[t])})")

road_only = nodes_df[nodes_df['node_type'] == 'road']
ax.scatter(road_only['x'], road_only['y'], c='#8c95a0', s=20, zorder=2, alpha=0.6, label='道路路口节点')

for _, s_row in sites.iterrows():
    s_id, n_id = s_row['entity_id'], s_row['node_id']
    sx, sy = node_x[n_id], node_y[n_id]
    ax.scatter(sx, sy, marker='*', c='#e63946', s=240, edgecolors='black', lw=1.2, zorder=6)
    ax.text(sx + 1.2, sy - 1.2, f"{s_id} (等级{int(s_row['disaster_level'])})", fontsize=9, fontweight='bold', color='#b7094c', zorder=7)

for _, t_row in teams.iterrows():
    t_id, n_id = t_row['entity_id'], t_row['node_id']
    tx, ty = node_x[n_id], node_y[n_id]
    ax.scatter(tx, ty, marker='^', c='#0077b6', s=220, edgecolors='black', lw=1.2, zorder=6)
    ax.text(tx + 1.2, ty + 1.2, f"{t_id} (能力:{int(t_row['rescue_capacity'])})", fontsize=9, fontweight='bold', color='#023e8a', zorder=7)

ax.set_title("问题一：最优应急救援行军路径与协同调度网络图", fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel("X 坐标 (km)", fontsize=12, fontweight='bold')
ax.set_ylabel("Y 坐标 (km)", fontsize=12, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend(loc='lower left', bbox_to_anchor=(0.0, -0.28), ncol=2, fontsize=9, frameon=True)
plt.tight_layout()
plt.savefig("figure1_rescue_routes_cn.png", dpi=300, bbox_inches='tight')
plt.close()

# 图 2: 救援时序协同调度甘特图 (全中文)
fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
team_names = sorted(team_caps.keys(), reverse=True)
y_positions = {t: i for i, t in enumerate(team_names)}
c_travel, c_wait, c_work = '#8ecae6', '#ffb703', '#fb8500'

for t, events in opt_res['team_events'].items():
    y = y_positions[t]
    for ev in events:
        dep, arr, st, end, dest = ev['depart_time'], ev['arrival_time'], ev['start_rescue_time'], ev['end_rescue_time'], ev['to']
        ax.barh(y, arr - dep, left=dep, height=0.55, color=c_travel, edgecolor='black', lw=0.6)
        if st > arr:
            ax.barh(y, st - arr, left=arr, height=0.55, color=c_wait, edgecolor='black', lw=0.6, hatch='//')
        ax.barh(y, end - st, left=st, height=0.55, color=c_work, edgecolor='black', lw=0.8)
        ax.text((st + end)/2, y, dest, ha='center', va='center', fontsize=8.5, fontweight='bold', color='white')

ax.set_yticks(range(len(team_names)))
ax.set_yticklabels([f"救援队 {t} (能力:{int(team_caps[t])})" for t in team_names], fontsize=11, fontweight='bold')
ax.set_xlabel("灾后经过时间 (分钟)", fontsize=12, fontweight='bold')
ax.set_title("问题一：多支救援队伍协同作业时序调度甘特图", fontsize=15, fontweight='bold', pad=15)
ax.axvline(opt_res['makespan'], color='#d62728', linestyle='--', lw=1.8)

patch_travel = mpatches.Patch(facecolor=c_travel, edgecolor='black', label='途中行驶转移')
patch_wait = mpatches.Patch(facecolor=c_wait, hatch='//', edgecolor='black', label='等待协同队伍到达')
patch_work = mpatches.Patch(facecolor=c_work, edgecolor='black', label='现场实施救援作业')
line_makespan = plt.Line2D([0], [0], color='#d62728', linestyle='--', lw=1.8, label=f"总救援完成时间 ({opt_res['makespan']:.1f} 分钟 / {opt_res['makespan']/60:.2f} 小时)")

ax.legend(handles=[patch_travel, patch_wait, patch_work, line_makespan], loc='upper right', frameon=True, fontsize=10)
ax.grid(axis='x', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig("figure2_rescue_gantt_cn.png", dpi=300, bbox_inches='tight')
plt.close()

# 图 3: 方案综合评价看板 (全中文)
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
site_df = pd.DataFrame([{'site': s, 'level': site_levels[s], 'start_time': opt_res['site_start'][s], 'req_cap': site_reqs[s]} for s in site_reqs])

# (a) 灾情等级与响应启动时间
ax1 = axes[0]
colors_level = {1: '#2a9d8f', 2: '#e9c46a', 3: '#f4a261', 4: '#e76f51', 5: '#d62828'}
for lvl in sorted(site_df['level'].unique()):
    sub = site_df[site_df['level'] == lvl]
    ax1.scatter(sub['level'], sub['start_time'], s=sub['req_cap']*20, c=colors_level[lvl], edgecolors='black', lw=1.2, alpha=0.85, label=f"{int(lvl)}级灾情")
    for _, r in sub.iterrows():
        ax1.text(r['level'] + 0.08, r['start_time'] - 4, r['site'], fontsize=8.5, fontweight='bold', va='center')

mean_resp_per_level = site_df.groupby('level')['start_time'].mean()
ax1.plot(mean_resp_per_level.index, mean_resp_per_level.values, color='black', linestyle='--', marker='o', lw=1.8, label='各等级平均响应时刻')
ax1.set_title("(a) 受灾点等级与救援启动时刻关系", fontsize=12, fontweight='bold')
ax1.set_xlabel("受灾严重等级 (1~5级)", fontsize=10, fontweight='bold')
ax1.set_ylabel("救援开始时刻 (分钟)", fontsize=10, fontweight='bold')
ax1.set_xticks([1, 2, 3, 4, 5])
ax1.set_xticklabels(['1级(轻微)', '2级(中度)', '3级(重度)', '4级(严重)', '5级(极危)'])
ax1.set_xlim(0.6, 5.6)
ax1.set_ylim(-10, 480)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(fontsize=8, loc='upper left', framealpha=0.9)

# (b) 队伍工时分解
ax2 = axes[1]
teams_sorted = sorted(team_caps.keys())
tr_vals = [opt_res['team_travel'][t] for t in teams_sorted]
wt_vals = [opt_res['team_wait'][t] for t in teams_sorted]
wk_vals = [opt_res['team_work'][t] for t in teams_sorted]
ax2.bar(teams_sorted, tr_vals, label='行驶时间', color='#8ecae6', edgecolor='black', lw=0.6)
ax2.bar(teams_sorted, wt_vals, bottom=tr_vals, label='协同等待时间', color='#ffb703', edgecolor='black', lw=0.6, hatch='//')
ax2.bar(teams_sorted, wk_vals, bottom=np.array(tr_vals)+np.array(wt_vals), label='现场救援时间', color='#fb8500', edgecolor='black', lw=0.6)
ax2.set_title("(b) 8支救援队伍时间与工作负荷分配", fontsize=12, fontweight='bold')
ax2.set_xlabel("救援队伍编号", fontsize=10, fontweight='bold')
ax2.set_ylabel("总累计时间 (分钟)", fontsize=10, fontweight='bold')
ax2.grid(axis='y', linestyle=':', alpha=0.6)
ax2.legend(fontsize=8.5, loc='upper right')

# (c) KPI 核心指标卡
ax3 = axes[2]
ewr = opt_res['total_work_time'] / (opt_res['total_travel_time'] + opt_res['total_work_time'] + opt_res['total_wait_time']) * 100
table_data = [
    ['总救援完成时间', f"{opt_res['makespan']:.1f} 分钟 ({opt_res['makespan']/60:.2f} 小时)"],
    ['加权平均响应时间', f"{opt_res['weighted_response_time']:.1f} 分钟"],
    ['算术平均响应时间', f"{opt_res['avg_response_time']:.1f} 分钟"],
    ['所有队伍总在途行驶时间', f"{opt_res['total_travel_time']:.1f} 分钟"],
    ['有效施救时间占比', f"{ewr:.1f} %"]
]
ax3.axis('off')
ax3.set_title("(c) 救援方案综合评价指标 ", fontsize=12, fontweight='bold')
table = ax3.table(cellText=table_data, colLabels=['评价指标名称', '优化计算结果'], loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9.5)
table.scale(1.05, 2.0)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#023e8a')
        cell.set_text_props(color='white', fontweight='bold')
    else:
        cell.set_facecolor('#f0f4f8' if row % 2 == 0 else '#ffffff')

plt.tight_layout()
plt.savefig("figure3_evaluation_dashboard_cn.png", dpi=300, bbox_inches='tight')
plt.close()
print("All 3 Chinese figures generated successfully!")