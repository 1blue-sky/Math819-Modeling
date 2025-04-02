import numpy as np
import plotly.graph_objs as go

# 目标函数： 使用经典 Rastrigin 函数作为测试目标，适合评估多峰全局优化能力。
# 目标函数（Rastrigin）
def rastrigin(X):
    A = 10
    return A * len(X) + sum([x ** 2 - A * np.cos(2 * np.pi * x) for x in X])

# 粒子群优化（PSO）
def particle_swarm_optimization(dim=2, num_particles=30, max_iter=100, bounds=(-5.12, 5.12)):
    lb, ub = bounds
    w = 0.7         # 惯性权重
    c1 = c2 = 1.5   # 个体与群体学习因子

    positions = np.random.uniform(lb, ub, (num_particles, dim))
    velocities = np.random.uniform(-1, 1, (num_particles, dim))
    personal_best_pos = positions.copy()
    personal_best_val = np.array([rastrigin(p) for p in positions])
    global_best_pos = personal_best_pos[np.argmin(personal_best_val)]
    global_best_val = min(personal_best_val)

    history_positions = []
    history_global = []

    for t in range(max_iter):
        for i in range(num_particles):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            velocities[i] = w * velocities[i] + \
                            c1 * r1 * (personal_best_pos[i] - positions[i]) + \
                            c2 * r2 * (global_best_pos - positions[i])
            positions[i] += velocities[i]
            positions[i] = np.clip(positions[i], lb, ub)

            value = rastrigin(positions[i])
            if value < personal_best_val[i]:
                personal_best_val[i] = value
                personal_best_pos[i] = positions[i]

                if value < global_best_val:
                    global_best_val = value
                    global_best_pos = positions[i]

        history_positions.append(positions.copy())
        history_global.append(global_best_val)
        print(f"Iter {t+1}: Best Value = {global_best_val:.4f}")

    return global_best_pos, global_best_val, history_positions, history_global

# PSO 可视化动画

def plot_pso_animation(history_positions, bounds=(-5.12, 5.12)):
    frames = []
    lb, ub = bounds

    for t, swarm in enumerate(history_positions):
        frame = go.Frame(
            data=[
                go.Scatter(x=swarm[:, 0], y=swarm[:, 1], mode='markers', marker=dict(size=8), name=f'Iter {t+1}')
            ],
            name=f'Iter {t+1}'
        )
        frames.append(frame)

    layout = go.Layout(
        title="PSO 群体位置演化动画",
        xaxis=dict(range=[lb, ub], title='X'),
        yaxis=dict(range=[lb, ub], title='Y'),
        updatemenus=[dict(type='buttons', showactive=False,
                          buttons=[dict(label='播放', method='animate', args=[None])])]
    )

    fig = go.Figure(
        data=[go.Scatter(x=[], y=[], mode='markers')],
        layout=layout,
        frames=frames
    )
    fig.show()

# PSO 收敛曲线

def plot_convergence_curve(global_history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=global_history, mode='lines+markers', name='Global Best'))
    fig.update_layout(title='PSO 收敛曲线', xaxis_title='Iteration', yaxis_title='Best Fitness')
    fig.show()

# 运行
best_pos, best_val, history_positions, history_global = particle_swarm_optimization()
plot_pso_animation(history_positions)
plot_convergence_curve(history_global)
