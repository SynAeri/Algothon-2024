
import numpy as np

##### TODO #########################################
### RENAME THIS FILE TO YOUR TEAM NAME #############
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50
currentPos = np.zeros(nInst)
Curr_strat = 0
#Original - do not delete
""" def getMyPosition(prcSoFar):
    global currentPos
    #Shape of the input array = prices
    (nins, nt) = prcSoFar.shape

    #Days of data less than 2 return zero positions (nins is declared in which is outputted in the shape )
    if (nt < 2):
        return np.zeros(nins)
    #Calcs log return of the 
    lastRet = np.log(prcSoFar[:, -1] / prcSoFar[:, -2])
    lNorm = np.sqrt(lastRet.dot(lastRet))
    lastRet /= lNorm
    rpos = np.array([int(x) for x in 5000 * lastRet / prcSoFar[:, -1]])
    currentPos = np.array([int(x) for x in currentPos+rpos])
    return currentPos """

#Jordan Note: Add another getMyPosition for referencing MACD

def getMyPosition(prcSoFar):
    global currentPos
    # Trade Strategy
    positions = SMA(prcSoFar)

    # Update the current positions to the trading strategy
    #print("Positions", currentPos)
    currentPos = np.array([int(x) for x in positions])
    # Return the updated positions
    return currentPos


def SMA(prcSoFar):
    #days, inst
    (nins, nt) = prcSoFar.shape
    #Make an empty array similar to the size of the price.txt
    positions = np.zeros(nins)
    
    # Initialize the SMA matrix
    SMA_13 = np.zeros(prcSoFar.shape)
    
    # Calculate SMA for each stock
    for i in range(prcSoFar.shape[0]):  # loop for 50 stocks
        for j in range(12, prcSoFar.shape[1]):  # loop from day 13 to the end
            # SMA at day 13 = (sum of prices from day 1 to 13) / 13
            SMA_13[i, j] = np.sum(prcSoFar[i, j-12:j+1]) / 13
    
    # Trading logic based on SMA
    for i in range(nins):
        #We are now comparing the values from the SMA_13 with the price.txt matrix
        #Attempt at adding shares but not the best shares to buy
        if prcSoFar[i, -1] > SMA_13[i, -1]:
            #Debug print can comment it off if you want
            #print("->", prcSoFar[i, -1], " Less than: ", SMA_13[i,-1], " so increase")
            positions[i] = 10
        else:
            positions[i] = -10
    
    return positions
