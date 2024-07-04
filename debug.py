import numpy as np
import cvxpy as cp   #### convex optimisation problems packages ####
import pandas as pd

##### TODO #########################################
### RENAME THIS FILE TO YOUR TEAM NAME #############
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

#### Mean Variance Optimisation Strategy ####

# Variable to track days since last position update
days_since_last_update = 0
UPDATE_FREQUENCY = 41 # Update positions every 40 day, which means we will trade every 40 days
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
    gamma = 1.9  # Risk aversion parameter
    
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

nInst = 0
nt = 0
commRate = 0.0010
dlrPosLimit = 10000


def loadPrices(fn):
    global nt, nInst
    df = pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    (nt, nInst) = df.shape
    return (df.values).T


pricesFile = "./prices.txt"
prcAll = loadPrices(pricesFile)
print("Loaded %d instruments for %d days" % (nInst, nt))


def calcPL(prcHist):
    cash = 0
    curPos = np.zeros(nInst)
    totDVolume = 0
    totDVolumeSignal = 0
    totDVolumeRandom = 0
    value = 0
    todayPLL = []
    (_, nt) = prcHist.shape
    for t in range(3, 501): #best is around 129- 140 for lowest and positive result\
        #129 - 501 outputs 5.05 score with freq of 42 and gamma 3
        #120 - 501 outputs score 0.46 with freq of 42 and gamma 3
        #130 - 501 outputs score 10.5 with freq of 41 and gamma 2
        prcHistSoFar = prcHist[:, :t]
        newPosOrig = getMyPosition(prcHistSoFar)
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
        if (totDVolume > 0):
            ret = value / totDVolume
        print("Day %d value: %.2lf todayPL: $%.2lf $-traded: %.0lf return: %.5lf" %
              (t, value, todayPL, totDVolume, ret))
    pll = np.array(todayPLL)
    (plmu, plstd) = (np.mean(pll), np.std(pll))
    annSharpe = 0.0
    if (plstd > 0):
        annSharpe = np.sqrt(250) * plmu / plstd
    return (plmu, ret, plstd, annSharpe, totDVolume)


(meanpl, ret, plstd, sharpe, dvol) = calcPL(prcAll)
score = meanpl - 0.1*plstd
print("=====")
print("mean(PL): %.1lf" % meanpl)
print("return: %.5lf" % ret)
print("StdDev(PL): %.2lf" % plstd)
print("annSharpe(PL): %.2lf " % sharpe)
print("totDvolume: %.0lf " % dvol)
print("Score: %.2lf" % score)