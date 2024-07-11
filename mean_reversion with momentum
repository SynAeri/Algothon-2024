import numpy as np
import pandas as pd

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

nInst = 50
currentPos = np.zeros(nInst)

def mean_reversion_signal(prcSoFar, lookback_mean):
    if prcSoFar.shape[1] < lookback_mean:
        return np.zeros(prcSoFar.shape[0])
    means = np.mean(prcSoFar[:, -lookback_mean:], axis=1)
    stds = np.std(prcSoFar[:, -lookback_mean:], axis=1)
    z_scores = (prcSoFar[:, -1] - means) / stds
    return -z_scores

def momentum_signal(prcSoFar, lookback_momentum):
    if prcSoFar.shape[1] < lookback_momentum:
        return np.zeros(prcSoFar.shape[0])
    return prcSoFar[:, -1] - prcSoFar[:, -lookback_momentum]

def getMyPosition(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape
    
    if nt < 30:
        return np.zeros(nins)
    
    lookback_mean = 44  # Experiment with different values
    lookback_momentum = 30  # Experiment with different values
    
    # Mean Reversion Signal
    signals_mean = mean_reversion_signal(prcSoFar, lookback_mean)
    
    # Momentum Signal
    signals_momentum = momentum_signal(prcSoFar, lookback_momentum)
    
    # Combine signals with dynamic weights
    combined_signals = 0.3 * signals_mean + 0.2 * signals_momentum #weight mean, weight momentum
    combined_signals = combined_signals / np.sum(np.abs(combined_signals))
    
    # Calculate target positions with dynamic sizing
    target_position = np.array([int(x) for x in dlrPosLimit * 0.5 * combined_signals / prcSoFar[:, -1]])
    
    position_change = 0.2 * (target_position - currentPos)
    currentPos = np.array([int(x) for x in currentPos + position_change])
    
    # Apply position limits
    current_prices = prcSoFar[:, -1]
    posLimits = np.array([int(x) for x in dlrPosLimit / current_prices])
    currentPos = np.clip(currentPos, -posLimits, posLimits)
    
    return currentPos

def calcPL(prcHist):
    cash = 0
    curPos = np.zeros(nInst)
    totDVolume = 0
    value = 0
    todayPLL = []
    (_, nt) = prcHist.shape
    for t in range(250, 500):
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
        if totDVolume > 0:
            ret = value / totDVolume
        print("Day %d value: %.2lf todayPL: $%.2lf $-traded: %.0lf return: %.5lf" %
              (t, value, todayPL, totDVolume, ret))
    pll = np.array(todayPLL)
    (plmu, plstd) = (np.mean(pll), np.std(pll))
    annSharpe = 0.0
    if plstd > 0:
        annSharpe = np.sqrt(250) * plmu / plstd
    return (plmu, ret, plstd, annSharpe, totDVolume)

(meanpl, ret, plstd, sharpe, dvol) = calcPL(prcAll)
score = meanpl - 0.1 * plstd

print("=====")
print("mean(PL): %.1lf" % meanpl)
print("return: %.5lf" % ret)
print("StdDev(PL): %.2lf" % plstd)
print("annSharpe(PL): %.2lf " % sharpe)
print("totDvolume: %.0lf " % dvol)
print("Score: %.2lf" % score)
