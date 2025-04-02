import numpy as np
import random
import plotly.graph_objs as go
"""
蚁群算法
问题定义：TSP 旅行商问题
"""

def generate_cities(n, seed=42):
    np.random.seed(seed)
    return np.random.rand(n, 2) * 100

def distance_matrix(cities):
    n = len(cities)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i][j] = np.linalg.norm(cities[i] - cities[j])
    return dist

class AntColony:
    def __init__(self, dist_matrix, n_ants=20, n_iter=100, alpha=1.0, beta=5.0, rho=0.5, Q=100):
        self.dist = dist_matrix
        self.n = dist_matrix.shape[0]
        self.pheromone = np.ones((self.n, self.n))
        self.heuristic = 1 / (dist_matrix + 1e-10)
        self.n_ants = n_ants
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.best_cost = float('inf')
        self.best_path = []
        self.history_paths = []
        self.history_costs = []
        self.history_pheromones = []

    def run(self):
        for iteration in range(self.n_iter):
            paths = []
            costs = []

            for ant in range(self.n_ants):
                path = self.construct_path()
                cost = self.path_length(path)
                paths.append(path)
                costs.append(cost)

            min_cost_idx = np.argmin(costs)
            if costs[min_cost_idx] < self.best_cost:
                self.best_cost = costs[min_cost_idx]
                self.best_path = paths[min_cost_idx]

            self.update_pheromones(paths, costs)
            self.history_paths.append(list(self.best_path))
            self.history_costs.append(self.best_cost)
            self.history_pheromones.append(np.copy(self.pheromone))
            print(f"Iter {iteration+1}: Best Cost = {self.best_cost:.2f}")

        return self.best_path, self.best_cost

    def construct_path(self):
        path = []
        visited = set()
        current = random.randint(0, self.n - 1)
        path.append(current)
        visited.add(current)

        for _ in range(self.n - 1):
            probs = []
            for j in range(self.n):
                if j not in visited:
                    tau = self.pheromone[current][j] ** self.alpha
                    eta = self.heuristic[current][j] ** self.beta
                    probs.append((j, tau * eta))
            total = sum(p for _, p in probs)
            probs = [(city, p / total) for city, p in probs]
            next_city = random.choices([c for c, _ in probs], [p for _, p in probs])[0]
            path.append(next_city)
            visited.add(next_city)
            current = next_city

        return path

    def path_length(self, path):
        return sum(self.dist[path[i]][path[i+1]] for i in range(len(path) - 1)) + self.dist[path[-1]][path[0]]

    def update_pheromones(self, paths, costs):
        self.pheromone *= (1 - self.rho)
        for path, cost in zip(paths, costs):
            for i in range(len(path) - 1):
                self.pheromone[path[i]][path[i+1]] += self.Q / cost
            self.pheromone[path[-1]][path[0]] += self.Q / cost

def plot_path_animation(cities, history_paths):
    frames = []
    base_x = [c[0] for c in cities]
    base_y = [c[1] for c in cities]

    for step, path in enumerate(history_paths):
        path_x = [cities[i][0] for i in path] + [cities[path[0]][0]]
        path_y = [cities[i][1] for i in path] + [cities[path[0]][1]]

        frame = go.Frame(
            data=[
                # 所有城市点（固定背景）
                go.Scatter(x=base_x, y=base_y, mode='markers+text',
                           text=[str(i) for i in range(len(cities))],
                           marker=dict(size=6, color='gray'),
                           textposition='top center', name='城市'),
                # 最优路径城市（红色）
                go.Scatter(x=path_x[:-1], y=path_y[:-1], mode='markers',
                           marker=dict(size=10, color='red'), name='最优路径城市'),
                # 路径连线（蓝色）
                go.Scatter(x=path_x, y=path_y, mode='lines',
                           line=dict(color='blue', width=2), name='最优路径')
            ],
            name=f"Step {step+1}"
        )
        frames.append(frame)

    fig = go.Figure(
        data=frames[0].data,
        layout=go.Layout(
            title="ACO 路径进化动画",
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            updatemenus=[dict(type='buttons', showactive=False,
                              buttons=[dict(label='播放', method='animate', args=[None])])]
        ),
        frames=frames
    )
    fig.show()

def plot_pheromone_heatmap(pheromone_matrix):
    fig = go.Figure(data=go.Heatmap(z=pheromone_matrix, colorscale='Viridis'))
    fig.update_layout(title='Pheromone Heatmap', xaxis_title='City', yaxis_title='City')
    fig.show()

def plot_convergence_curve(costs):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=costs, mode='lines+markers', name='Best Cost'))
    fig.update_layout(title='ACO 收敛曲线', xaxis_title='Iteration', yaxis_title='Best Cost')
    fig.show()

def compare_multiple_runs(cities, param_sets):
    traces = []
    for i, (alpha, beta, rho) in enumerate(param_sets):
        dist = distance_matrix(cities)
        aco = AntColony(dist, n_ants=30, n_iter=50, alpha=alpha, beta=beta, rho=rho)
        aco.run()
        traces.append(go.Scatter(y=aco.history_costs, name=f"α={alpha}, β={beta}, ρ={rho}"))

    fig = go.Figure(traces)
    fig.update_layout(title='参数对比：ACO收敛曲线', xaxis_title='Iteration', yaxis_title='Best Cost')
    fig.show()

n_cities = 40
cities = generate_cities(n_cities)
dist = distance_matrix(cities)
aco = AntColony(dist, n_ants=30, n_iter=50, alpha=1, beta=5, rho=0.5)
best_path, best_cost = aco.run()

plot_path_animation(cities, aco.history_paths)
plot_pheromone_heatmap(aco.history_pheromones[-1])
plot_convergence_curve(aco.history_costs)
compare_multiple_runs(cities, param_sets=[(1, 5, 0.5), (1, 3, 0.3), (1, 7, 0.7)])
