import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
import time
import random

# I1 = lambda x: (1 / np.pi) * np.arctan(x) + 1 / 2
I2 = lambda x: ((2 / np.pi) * abs(np.arctan(x)))
# I3 = lambda x: (1 / np.pi) * ((np.pi / 2) - np.arctan(x))
# I4 = lambda x: -(2 / np.pi) * np.abs(np.arctan(x)) + 1

def read_excel_file(file_path):
    df = pd.read_excel(file_path, header=None)
    df.columns = [str(i) for i in range(df.shape[1])]
    return df

def preprocess_data(df):
    y = df.iloc[:, 0]
    X = df.iloc[:, 1:]

    numerical_features = X.select_dtypes(include=[np.number]).columns
    categorical_features = X.select_dtypes(exclude=[np.number]).columns

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    X_processed = preprocessor.fit_transform(X)
    return X_processed, y

# Q-learning controller class
class QLearningController:
    def __init__(self, state_size=10, action_size=6, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((state_size, action_size))
        self.state = 0
        self.omega = 0.5
        self.lambd = 0.1
        self.delta = 0.05

    def compute_state(self, best_fitness, global_best_fitness, t, t_max, epsilon_small=1e-6):
        ratio = best_fitness / (global_best_fitness + epsilon_small)
        index = int(self.state_size * ratio * (t / t_max))
        index = min(self.state_size - 1, max(0, index))
        self.state = index
        return self.state

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            return np.argmax(self.q_table[self.state])

    def update_parameters(self, action):
        if action == 0:
            self.omega = min(1.0, self.omega + self.delta)
        elif action == 1:
            self.omega = max(0.0, self.omega - self.delta)
        elif action == 2:
            self.lambd = min(1.0, self.lambd + self.delta)
        elif action == 3:
            self.lambd = max(0.0, self.lambd - self.delta)
        elif action == 4:
            pass
        elif action == 5:
            self.omega = np.clip(self.omega + np.random.uniform(-0.01, 0.01), 0.0, 1.0)
            self.lambd = np.clip(self.lambd + np.random.uniform(-0.01, 0.01), 0.0, 1.0)

    def compute_reward(self, old_fitness, new_fitness):
        if new_fitness > old_fitness:
            return 1
        elif new_fitness == old_fitness:
            return 0
        else:
            return -1

    def update_q_table(self, action, reward, next_state):
        best_next_q = np.max(self.q_table[next_state])
        old_q = self.q_table[self.state, action]
        self.q_table[self.state, action] = old_q + self.alpha * (reward + self.gamma * best_next_q - old_q)
        self.state = next_state

    def get_current_params(self):
        return self.omega, self.lambd

def binary_gwo(n_agents, dim, max_iter, obj_func):
    """
    IBGWO

    Parameters:
        n_agents: Number of individuals in the wolf pack
        dim: Search space dimension (number of features)
        max_iter: Maximum number of iterations
        obj_func: Objective function, takes a binary solution
            (0/1 array) as input and outputs fitness (accuracy)

    Returns:
        best_wolf: Global optimal binary solution (selected features)
        best_score: Corresponding fitness value (classification accuracy)
    """
    wolves = (np.random.uniform(0, 1, (n_agents, dim)) > 0.5).astype(np.float64)

    omega_init = 0.5
    lambda_val = 0.1
    best_alpha = None
    best_score = -np.inf

    for iter in tqdm(range(max_iter), desc="GWO Iterations", leave=False):

        fitness = np.array(Parallel(n_jobs=-1)(delayed(obj_func)(wolf) for wolf in wolves))

        sorted_indices = np.argsort(-fitness)
        alpha = wolves[sorted_indices[0]]
        beta = wolves[sorted_indices[1]]
        delta = wolves[sorted_indices[2]]
        if fitness[sorted_indices[0]] > best_score:
            best_alpha = alpha.copy()
            best_score = fitness[sorted_indices[0]]

        omega = omega_init * (1 - iter / max_iter)

        A1 = 2 * np.random.rand(n_agents, dim) - 1
        C1 = 2 * np.random.rand(n_agents, dim)
        A2 = 2 * np.random.rand(n_agents, dim) - 1
        C2 = 2 * np.random.rand(n_agents, dim)
        A3 = 2 * np.random.rand(n_agents, dim) - 1
        C3 = 2 * np.random.rand(n_agents, dim)

        D_alpha = np.abs(C1 * alpha - wolves)
        D_beta  = np.abs(C2 * beta - wolves)
        D_delta = np.abs(C3 * delta - wolves)

        X1 = alpha - A1 * D_alpha
        X2 = beta - A2 * D_beta
        X3 = delta - A3 * D_delta

        avg_update = X1 + X2 + X3

        rand_indices = np.random.choice(n_agents, n_agents, replace=True)
        X_rand = wolves[rand_indices]

        X_new = (1 - omega - lambda_val) * avg_update + omega * wolves + lambda_val * X_rand

        wolves = (I2(X_new) > np.random.rand(n_agents, dim)).astype(np.float64)

    print("Best score =", best_score)
    return best_alpha, best_score

def objective_function(solution):
    """
    Objective function: Given a binary solution (each bit is 0 or 1 to
    indicate whether the corresponding feature is selected),
    use the selected features to construct a KNN classifier and
    calculate the classification accuracy using five-fold cross-validation.
    If the feature is not selected, return 0 as the fitness.
    """
    selected_features = np.where(solution == 1)[0]
    if len(selected_features) == 0:
        return 0

    X_selected = X[:, selected_features]
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

    warnings.filterwarnings("ignore", category=UserWarning)

    scores = cross_val_score(knn, X_selected, y, cv=5)
    return np.mean(scores) * 100

if __name__ == '__main__':
    file_path = './Brain_Tumor2.xlsx'
    df = read_excel_file(file_path)
    X, y = preprocess_data(df)

    # Feature numbers
    # GWO parameter configuration
    dim = X.shape[1]
    n_agents = 30
    max_iter = 100
    n_runs = 30

    start_time = time.time()

    # 30 independent runs of GWO
    # each using five-fold cross-validation to evaluate accuracy in the objective function
    results = Parallel(n_jobs=3)(
        delayed(binary_gwo)(n_agents, dim, max_iter, objective_function)
        for _ in range(n_runs)
    )

    end_time = time.time()
    total_time = end_time - start_time
    avg_run_time = total_time / n_runs

    best_solutions, best_fitnesses = zip(*results)
    selected_features_count = [len(np.where(sol == 1)[0]) for sol in best_solutions]
