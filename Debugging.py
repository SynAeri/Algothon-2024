import numpy as np
import cvxpy as cp
import pandas as pd

# Initialize global variables
nInst = 50
commRate = 0.0010
dlrPosLimit = 10000
currentPos = np.zeros(nInst)
days_since_last_update = 0

# Load historical prices from a file
def loadPrices(fn):
    df = pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    (nt, nInst) = df.shape
    return df.values.T

pricesFile = "./prices.txt"
prcAll = loadPrices(pricesFile)
print("Loaded %d instruments for %d days" % (nInst, prcAll.shape[1]))

def getMyPosition(prcSoFar, gamma, update_frequency):
    global currentPos, days_since_last_update, UPDATE_FREQUENCY

    (nins, nt) = prcSoFar.shape
    if nt < 2:
        return np.zeros(nins)

    # Calculate daily returns
    returns = np.diff(prcSoFar, axis=1) / prcSoFar[:, :-1]

    # Calculate expected returns and covariance matrix
    expected_returns = np.mean(returns, axis=1)
    cov_matrix = np.cov(returns)
    
    # Regularize covariance matrix to ensure it's positive semi-definite
    cov_matrix += np.eye(nins) * 1e-6

    # Mean-variance optimization using cvxpy
    w = cp.Variable(nins)
    risk = cp.quad_form(w, cov_matrix)
    ret = expected_returns @ w

    prob = cp.Problem(cp.Maximize(ret - gamma * risk), [cp.sum(w) == 0, w >= -1, w <= 1])

    try:
        prob.solve()
        optimal_weights = w.value
    except cp.error.SolverError as e:
        print(f"Optimization failed: {e}")
        optimal_weights = np.zeros(nins)

    new_positions = np.zeros(nins)
    budget = 9800

    if days_since_last_update >= update_frequency:
        for i in range(nins):
            current_price = prcSoFar[i, -1]
            if optimal_weights[i] > 0:
                num_shares = (budget * optimal_weights[i]) // current_price
                position_value = num_shares * current_price
                if position_value > dlrPosLimit:
                    num_shares = dlrPosLimit // current_price
            elif optimal_weights[i] < 0:
                num_shares = (budget * optimal_weights[i]) // current_price
                position_value = num_shares * current_price
                if position_value < -dlrPosLimit:
                    num_shares = -dlrPosLimit // current_price
            else:
                num_shares = 0

            new_positions[i] = num_shares

        days_since_last_update = 0
    else:
        new_positions = currentPos
        days_since_last_update += 1

    currentPos = new_positions
    return currentPos

def calcPL(prcHist, gamma, update_frequency):
    global days_since_last_update
    days_since_last_update = 0  # Reset counter
    cash = 0
    curPos = np.zeros(nInst)
    totDVolume = 0
    value = 0
    todayPLL = []
    (_, nt) = prcHist.shape
    for t in range(3, nt):
        prcHistSoFar = prcHist[:, :t]
        newPosOrig = getMyPosition(prcHistSoFar, gamma, update_frequency)
        curPrices = prcHistSoFar[:, -1]
        posLimits = np.array([int(x) for x in dlrPosLimit / curPrices])
        newPos = np.clip(newPosOrig, -posLimits, posLimits)
        deltaPos = newPos - curPos
        dvolumes = curPrices * np.abs(deltaPos)
        dvolume = np.sum(dvolumes)
        totDVolume += dvolume
        comm = dvolume * commRate
        cash -= curPrices.dot(deltaPos) + comm
        curPos = np.array(newPos)
        posValue = curPos.dot(curPrices)
        todayPL = cash + posValue - value
        todayPLL.append(todayPL)
        value = cash + posValue
        ret = 0.0
        if totDVolume > 0:
            ret = value / totDVolume
        print(f"Day {t} value: {value:.2f} todayPL: {todayPL:.2f} $-traded: {totDVolume:.0f} return: {ret:.5f}")
    pll = np.array(todayPLL)
    (plmu, plstd) = (np.mean(pll), np.std(pll))
    annSharpe = 0.0
    if (plstd > 0):
        annSharpe = np.sqrt(250) * plmu / plstd
    return (plmu, ret, plstd, annSharpe, totDVolume)

def sliding_window_cross_validation(prcAll, update_frequencies, gammas, window_size=250, step_size=50):
    end_day = prcAll.shape[1]
    best_score = -np.inf
    best_params = {}
    best_train_period = {}
    best_test_period = {}

    scores_dict = {}

    for update_frequency in update_frequencies:
        for gamma in gammas:
            total_score = 0
            n_windows = 0
            param_key = (update_frequency, gamma)
            scores_dict[param_key] = []

            for start_day in range(0, end_day - 2 * window_size, step_size):
                train_end = start_day + window_size
                test_start = train_end + 1
                test_end = test_start + window_size

                if test_end >= end_day:
                    break

                train_data = prcAll[:, start_day:train_end]
                test_data = prcAll[:, test_start:test_end]

                print(f"Processing window: Train {start_day} to {train_end}, Test {test_start} to {test_end}")

                (meanpl, ret, plstd, sharpe, dvol) = calcPL(train_data, gamma, update_frequency)
                print(f"Params: (update_frequency={update_frequency}, gamma={gamma}), meanpl: {meanpl:.2f}, plstd: {plstd:.2f}")
                
                score = meanpl - 0.1 * plstd
                total_score += score
                scores_dict[param_key].append(score)
                n_windows += 1

            if n_windows > 0:  # Check to avoid division by zero
                avg_score = total_score / n_windows  # Compute the average score for the current parameter combination

                if avg_score > best_score:
                    best_score = avg_score
                    best_params = {
                        'update_frequency': update_frequency,
                        'gamma': gamma
                    }
                    best_train_period = {'start': start_day, 'end': train_end}
                    best_test_period = {'start': test_start, 'end': test_end}

    # Calculate the best parameters based on average scores across all periods
    avg_scores = {k: np.mean(v) if v else float('-inf') for k, v in scores_dict.items()}
    best_overall_params = max(avg_scores, key=avg_scores.get)
    best_overall_score = avg_scores[best_overall_params]

    return best_score, best_params, best_train_period, best_test_period, best_overall_params, best_overall_score, scores_dict

# Define parameter ranges
update_frequencies = range(40, 46)
gammas = np.linspace(2, 3, 6)

# Perform sliding window cross-validation
best_score, best_params, best_train_period, best_test_period, best_overall_params, best_overall_score, scores_dict = sliding_window_cross_validation(prcAll, update_frequencies, gammas)
print("Best Score: %.2lf" % best_score)
print("Best Parameters for Best Training Period:", best_params)
print("Best Training Period:", best_train_period)
print("Best Testing Period:", best_test_period)
print("Best Overall Parameters:", best_overall_params)
print("Best Overall Score: %.2lf" % best_overall_score)
print("Scores for all tested parameter combinations:")
for params, scores in scores_dict.items():
    print(f"Parameters: {params}, Scores: {scores_dict(scores)}")
