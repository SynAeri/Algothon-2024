import numpy as np
import cvxpy as cp   #### convex optimisation problems packages ####

##### TODO #########################################
### RENAME THIS FILE TO YOUR TEAM NAME #############
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

#### Mean Variance Optimisation Strategy ####

# Variable to track days since last position update
days_since_last_update = 0
UPDATE_FREQUENCY = 41 # Update positions every 41 days
nInst = 50
currentPos = np.zeros(nInst)
dlrPosLimit = 10000  # Dollar position limit

def getMyPosition(prcSoFar):
    global currentPos, days_since_last_update
    
    (nins, nt) = prcSoFar.shape
    if nt < 2:
        return np.zeros(nins)  # Not enough data points to calculate returns
    
    # Calculate daily returns
    returns = np.diff(prcSoFar, axis=1) / prcSoFar[:, :-1]
    
    # Calculate expected returns and covariance matrix
    expected_returns = np.mean(returns, axis=1)
    cov_matrix = np.cov(returns)
    
    # Mean-variance optimization using cvxpy package
    w = cp.Variable(nins)
    risk = cp.quad_form(w, cov_matrix)
    ret = expected_returns @ w
    gamma = 2.0  # Risk aversion parameter
    
    # We optimise weights of each stock in order to maximise expected return and minimise risk 
    prob = cp.Problem(cp.Maximize(ret - gamma * risk), [cp.sum(w) == 0, w >= -1, w <= 1]) 
    
    ### cp.sum(w) == 0 to protect bias in long and short position in case all stock prices going up or down 
    
    prob.solve()

    optimal_weights = w.value
    
    # Initialize positions
    new_positions = np.zeros(nins)
    budget = 9800   ### budget is set below 10000 to protect from a trading force in inactive interval of 40 days, in other words, 
                    ### we will trade position near 9800 so that inactive position resulting from price changes will not reach over 10000
                    ### Otherwise, we will waste unnecessary trading fees.
    # Check if it's time to update positions
    if days_since_last_update == UPDATE_FREQUENCY:
        # Calculate the number of shares for each instrument based on optimal weights
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
        
        # Reset days_since_last_update
        days_since_last_update = 0
    
    else:
        # Maintain current positions
        new_positions = currentPos
        days_since_last_update += 1
        
    # Update current positions
    currentPos = new_positions
    return currentPos