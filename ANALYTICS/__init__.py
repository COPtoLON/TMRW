###############################################################################
#                        Slopes, returns and accelerations
#                        For calculating import 
#                        intra and inter day differences
###############################################################################

def slope(array):
    """
    """
    y = np.array(array)
    x = np.arange(len(y))
    slopes, intercept, r_value, p_value, std_err = linregress(x,y)
    return(slopes)

def returns(x, logs = False):
    """
    """
    if type(x) == list: # if you've entered a list
        _return = np.zeros(len(x))
        for i in range(len(x)):
            _return[i] = ((x[i]/x[i-1])-1)
    elif type(x) == pd.DataFrame: # if you've entered a dataframe
        _return = x.pct_change().dropna()   
    else: # if you've entered neither a list or a dataframe
        _return = x.pct_change().dropna()
    
    # if you want log returns
    if logs == True:
        _return = np.log(abs(_return))
    
    df = pd.DataFrame(data=_return, dtype=np.float64)
    df.columns = ["returns"]
    return(df)

def acceleration(x, logs = False):
    """
    """
    x = returns(x, logs)
    if type(x) == list:  # if you've entered a list
        acc = np.zeros(len(x))
        for i in range(len(x)):
            acc[i] = ((x[i]/x[i-1])-1)
    elif type(x) == pd.DataFrame: # if you've entered a dataframe
        acc = x.pct_change().dropna() 
    else: # if you've entered neither a list or a dataframe
        acc = x.pct_change().dropna()
        
     # if you want log acceleration  
    if logs == True:
        acc = np.log(abs(acc))
    
    df = pd.DataFrame(data=acc, dtype=np.float64)
    df.columns = ["acceleration"]
    return(df)




###############################################################################
#                        Moving averages and stds
#                        volume weighted averages and stds
#                        bollinger bands?
###############################################################################

# Time-weighted Moving Average
def twa(data, window = 20, typ = "sma", x = True):
    """
    time-weighted moving average
    """
    if typ == "sma": # simple moving average
        if len(data) < window+1:
            raise ValueError("not enough data")
            
        if x == True:
            
            df = []
            for i in range(len(data)):
                if i < window:
                    df.append(data[i])
                else:
                    df.append(np.mean(data[i-window:i]))

            df = pd.DataFrame(df, index = data.index)
            df = df.iloc[0:len(df), 0]
            
            return(df)
    
        elif x == False:

            df = []
            for i in range(len(data)):

                ma = np.zeros(window)

                for j in range(1,window):

                    if i < window:
                        ma[j] = data[j]


                    if i >= window:

                        ma[j] = np.mean(data[i-j:i])

                df.append(ma)

            df = pd.DataFrame(df, index = data.index)
            df = df.iloc[0:len(df), 1:len(df.columns)]
            
            return(df)
    
    elif typ == "ema": # exponential moving average

        df = pd.DataFrame(index=data.index, dtype=np.float64)
        df.loc[:,0] = data.ewm(span=window, min_periods=0, adjust=True, ignore_na=False).mean().values.flatten() # ????
    
        return(df)
      
#Time-Price Moving STD
def twstd(data, window = 20, x = True):
    """
    time-weighted moving standard deviation of prices
    also called volatility
    """
    if len(data) < window+1:
        raise ValueError("not enough data")

    if x == True:
        
        df = []
        for i in range(len(data)):
            if i < window:
                df.append(data[i])
            else:
                df.append(np.std(data[i-window:i]))

        df = pd.DataFrame(df, index = data.index)
        df = df.iloc[0:len(df), 0]
        
        return(df)
        
    elif x == False:  
        
        df = []
        for i in range(len(data)):

            mstd = np.zeros(window)

            for j in range(1,window):

                if i >= j:

                    mstd[j] = np.std(data[i-j:i])

            df.append(mstd)

        df = pd.DataFrame(df, index = data.index)
        df = df.iloc[0:len(df), 1:len(df.columns)]
    
        return(df)
   
    
# Volume-weighted Moving Average
def vwa(data, weights, window = 20):
    """
    volume-weighted moving average
    """
    
    vwa = np.zeros(window)
    vwa = list(vwa)
    for i in range(len(data)):
        
        if i < window:
            
            vwa[i] = data[i]
            
        else:
    
            v_sum = sum(weights[i-window:i])
            v_w_avg = 0
            for j in range(window):
                v_w_avg = v_w_avg + data[i-j] * (weights[i-j] / v_sum)

            vwa.append(v_w_avg)
    
    df = pd.DataFrame(vwa)
    return(df)

#Volume-weighted Moving STD
def vwstd(data, weights, window = 20):
    """
    volume-weighted moving standard deviations
    """
    
    v_w_avg = 0
    vws = np.zeros(window)
    vws = list(vws)
    for i in range(window, len(data)):
    
        v_sum = sum(weights[i-window:i])
        v_w_std = 0
        for j in range(window):
            v_w_std = v_w_std + ((data[i-j] - np.mean(data[i-window:i]))**2 )* (weights[i-j] / v_sum)

        vws.append(np.sqrt(v_w_std))
    
    df = pd.DataFrame(vws)
    return(df)



###############################################################################
#                        Technical indicators
#                        RSI, Stochastic Oscillator
#
###############################################################################

def RSI(data, window = 20):
    """
    
    """
    delta = data.diff().dropna() # Close_now - Close_yesterday
    delta = delta.reset_index(drop = True)

    u = pd.DataFrame(np.zeros(len(delta))) # make an array of 0s for the up returns
    u = u[0]
    d = u.copy() # make an array of 0s for the down returns   

    u[delta > 0] = delta[delta > 0] # for all the days where delta is up, transfer them to U
    d[delta < 0] = -delta[delta < 0] # for all the days where delta is down, transfer them to D

    u[u.index[window-1]] = np.mean( u[:window] ) #first value is sum of avg gains
    u = u.drop(u.index[:(window-1)]) #drop the days before the window opens

    d[d.index[window-1]] = np.mean( d[:window] ) #first value is sum of avg losses
    d = d.drop(d.index[:(window-1)]) #drop the days before the window opens

    RS = pd.DataFrame.ewm(u, com=window-1, adjust=False).mean() / pd.DataFrame.ewm(d, com=window-1, adjust=False).mean() # EMA(up) / EMA(down)

    RSI_ = 100 - (100 / (1 + RS))
    return(RSI_)

def FRSI(data, window = 20):
    """
    inverse fisher transform on RSI = 0.1*(rsi-50)
    fisher rsi = (np.exp(2*rsi)-1) / (np.exp(2*rsi)+1)
    """
    
    RSI_ = 0.1 * (RSI(data, window) - 50)
    F_RSI = (np.exp(2*RSI_)-1) / (np.exp(2*RSI_)+1)
    return(F_RSI)

def BB(data, window = 20, std = 2.5):
    """
    
    """
    mean_lst = np.zeros(len(data))
    std_lst = np.zeros(len(data))
    for i in range(0, window):
        mean_lst[i] = data[i]
        std_lst[i] = 0.05 * data[i]
    
    for i in range(window,len(data)):
        mean_lst[i] = np.mean(data[i-window:i])
        std_lst[i] = np.std(data[i-window:i])
        
    up = mean_lst + std * std_lst
    down = mean_lst - std * std_lst
    
    df = pd.DataFrame()
    df['Upper'] = up
    df['Lower'] = down
    return(df)

def STO(data, N=14, M=3):
    assert 'Low' in data.columns
    assert 'High' in data.columns
    assert 'Close' in data.columns
    
    data_ = pd.DataFrame()
    data_['low_N'] = data['Low'].rolling(N).min()
    data_['high_N'] = data['High'].rolling(N).max()
    data_['K'] = 100 * (data['Close'] - data_['low_N']) / \
        (data_['high_N'] - data_['low_N']) # The stochastic oscillator
    data_['D'] = data_['K'].rolling(M).mean() # the slow and smoothed K
    return data_

def ADX(high, low, close, lookback):
    
    df = pd.DataFrame()
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
    atr = tr.rolling(lookback).mean()
    
    plus_di = 100 * (plus_dm.ewm(alpha = 1/lookback).mean() / atr)
    minus_di = abs(100 * (minus_dm.ewm(alpha = 1/lookback).mean() / atr))
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = ((dx.shift(1) * (lookback - 1)) + dx) / lookback
    adx_smooth = adx.ewm(alpha = 1/lookback).mean()
    df['+DI'] = plus_di
    df['-DI'] = minus_di
    df['ADX'] = adx_smooth
    
    return df

def MACD(price, slow, fast, smooth):
    exp1 = price.ewm(span = fast, adjust = False).mean()
    exp2 = price.ewm(span = slow, adjust = False).mean()
    macd = pd.DataFrame(exp1 - exp2).rename(columns = {'Close':'macd'})
    signal = pd.DataFrame(macd.ewm(span = smooth, adjust = False).mean()).rename(columns = {'macd':'signal'})
    hist = pd.DataFrame(macd['macd'] - signal['signal']).rename(columns = {0:'hist'})
    frames =  [macd, signal, hist]
    df = pd.concat(frames, join = 'inner', axis = 1)
    return df

def SuperTrend(high, low, close, lookback, multiplier):
    # ATR
    df = pd.DataFrame()
    
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
    atr = tr.ewm(lookback).mean()
    
    # H/L AVG AND BASIC UPPER & LOWER BAND
    
    hl_avg = (high + low) / 2
    upper_band = (hl_avg + multiplier * atr).dropna()
    lower_band = (hl_avg - multiplier * atr).dropna()
    
    # FINAL UPPER BAND
    
    final_bands = pd.DataFrame(columns = ['upper', 'lower'])
    final_bands.iloc[:,0] = [x for x in upper_band - upper_band]
    final_bands.iloc[:,1] = final_bands.iloc[:,0]
    
    for i in range(len(final_bands)):
        if i == 0:
            final_bands.iloc[i,0] = 0
        else:
            if (upper_band[i] < final_bands.iloc[i-1,0]) | (close[i-1] > final_bands.iloc[i-1,0]):
                final_bands.iloc[i,0] = upper_band[i]
            else:
                final_bands.iloc[i,0] = final_bands.iloc[i-1,0]
    
    # FINAL LOWER BAND
    
    for i in range(len(final_bands)):
        if i == 0:
            final_bands.iloc[i, 1] = 0
        else:
            if (lower_band[i] > final_bands.iloc[i-1,1]) | (close[i-1] < final_bands.iloc[i-1,1]):
                final_bands.iloc[i,1] = lower_band[i]
            else:
                final_bands.iloc[i,1] = final_bands.iloc[i-1,1]
    
    # SUPERTREND
    
    supertrend = pd.DataFrame(columns = [f'supertrend_{lookback}'])
    supertrend.iloc[:,0] = [x for x in final_bands['upper'] - final_bands['upper']]
    
    for i in range(len(supertrend)):
        if i == 0:
            supertrend.iloc[i, 0] = 0
        elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 0] and close[i] < final_bands.iloc[i, 0]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 0]
        elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 0] and close[i] > final_bands.iloc[i, 0]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 1]
        elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 1] and close[i] > final_bands.iloc[i, 1]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 1]
        elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 1] and close[i] < final_bands.iloc[i, 1]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 0]
    
    supertrend = supertrend.set_index(upper_band.index)
    supertrend = supertrend.dropna()[1:]
    
    # ST UPTREND/DOWNTREND
    
    upt = []
    dt = []
    close = close.iloc[len(close) - len(supertrend):]

    for i in range(len(supertrend)):
        if close[i] > supertrend.iloc[i, 0]:
            upt.append(supertrend.iloc[i, 0])
            dt.append(np.nan)
        elif close[i] < supertrend.iloc[i, 0]:
            upt.append(np.nan)
            dt.append(supertrend.iloc[i, 0])
        else:
            upt.append(np.nan)
            dt.append(np.nan)
            
    st, upt, dt = pd.Series(supertrend.iloc[:, 0]), pd.Series(upt), pd.Series(dt)
    upt.index, dt.index = supertrend.index, supertrend.index
    
    df['ST'] = st
    df['UPT'] = upt
    df['DT'] = dt
    
    return df



def PRICE_plot(data, MA = False):
    fig, (ax1) = plt.subplots(1, 1, figsize=(10, 3))
    #ax = plt.gca()
    ax1.set_facecolor('dimgrey')
    ax1.plot(data.index, data.Close, color = "deepskyblue")
    #ax1.set_xticks([])
    plt.grid()
    plt.show()


def STO_plot(data):
    fig, (ax1) = plt.subplots(1, 1, figsize=(10, 1))
    ax1.set_facecolor('dimgrey')
    
    ax1.plot(data.index, data.D, color = "deepskyblue")
    ax1.plot(data.index, np.repeat(80,len(data)), color = "white", linestyle = "dotted")
    ax1.plot(data.index, np.repeat(20,len(data)), color = "white", linestyle = "dotted")
    plt.show()

def ADX_plot(data, components = False):    
    fig, (ax1) = plt.subplots(1, 1, figsize=(10, 1))
    ax1.set_facecolor('dimgrey')
    
    if components == False:    
        ax1.plot(data.index, data['ADX'], color = "deepskyblue")
    
    elif components == True:
        ax1.plot(data.index, data['-DI'], color = "r")
        ax1.plot(data.index, data['+DI'], color = "g")
    
    ax1.plot(data.index, np.repeat(40,len(data)), color = "white", linestyle = "dotted")
    ax1.plot(data.index, np.repeat(20,len(data)), color = "white", linestyle = "dotted")
    plt.show()

def MACD_plot(prices, macd, signal, hist):
    fig, (ax1) = plt.subplots(1, 1, figsize=(10, 1))
    ax1.set_facecolor('dimgrey')

    for i in range(len(prices)):
        if str(hist[i])[0] == '-':
            ax1.bar(prices.index[i], hist[i], color = 'gold')
        else:
            ax1.bar(prices.index[i], hist[i], color = 'deepskyblue')
    
    plt.grid()
    plt.show()
    
def TRADE_plot(data, buy_price, sell_price):
    fig, (ax1) = plt.subplots(1, 1, figsize=(10, 5))
    ax1.set_facecolor('dimgrey')
    
    ax1.plot(data.index, data['Close'], linewidth = 2, color = "deepskyblue")
    ax1.plot(data.index, buy_price, marker = 'o', color = 'lime', markersize = 8, linewidth = 0)
    ax1.plot(data.index, sell_price, marker = 'X', color = 'tomato', markersize = 8, linewidth = 0)
    plt.grid()
    plt.show()

# not useful   
def RET_plot(data, strategy):
    fig, (ax1) = plt.subplots(1, 1, figsize=(10, 5))
    ax1.set_facecolor('dimgrey')
    
    rets = data.Close.pct_change().dropna()
    strat_rets = strategy.position[1:]*rets

    ax1.plot(rets.index, rets, color = 'k', linewidth = 1)
    ax1.plot(strat_rets.index, strat_rets, color = 'deepskyblue', linewidth = 1)
    plt.grid()
    plt.show()
    
# very useful   
def SUM_plot(data, strategy):
    fig, (ax1) = plt.subplots(1, 1, figsize=(10, 5))
    ax1.set_facecolor('dimgrey')
    
    rets = data.Close.pct_change().dropna()
    strat_rets = strategy.position[1:]*rets
    
    rets_cum = (1 + rets).cumprod() - 1 
    strat_cum = (1 + strat_rets).cumprod() - 1

    ax1.plot(rets_cum.index, rets_cum, color = 'w', linewidth = 2)
    ax1.plot(strat_cum.index, strat_cum, color = 'deepskyblue', linewidth = 2)
    plt.grid()
    plt.show()
    print(strat_cum[-1])







def data(x, y, z):
    # x is a stock symbol
    # y is a start date
    # z is an end date
    # kan bruges på commodities, currencies og aktier
    
    #MSFT er en aktier der kan bruges
    #USDEUR=X er en currency der kan bruges
    #GC=F er guld priser
    
    st = pd.DataFrame()
    t = yf.Ticker(x)
    st = t.history(start=y, end=z)
    st.index = pd.to_datetime(st.index).tz_localize(None)
    return(st)






###############################################################################
#                        Portfolio level important functions
#                        VaR, CVaR, Drawdown, Annualized returns, etc.
#                        Not used for trading, but for profit considerations.
###############################################################################

def var(data, level=5):
    """
    """
    z = norm.ppf(level/100)
    return(-(data.mean() + z*data.std(ddof=0)))

def cvar(data, level=5):
    """
    """
    confidence_level = 1-(level/100)  # Set the desired confidence level
    sorted_prices = np.sort(data)
    num_samples = len(sorted_prices)
    cvar_index = int((1 - confidence_level) * num_samples)
    cvar = np.mean(sorted_prices[:cvar_index])
    return(cvar)

def drawdown(return_series):
    """
    """
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({"Drawdown": drawdowns})

def sharpe(self,x):
    """
    """
    mu = x.mean()
    sigma = np.sqrt(x.var())
    return((mu-5) / sigma)

#def sortino(risk_free,degree_of_freedom,growth_rate,minimum):
    #v=np.sqrt(np.abs(scipy.integrate.quad(lambda g ((risk_freeg)**2)*scipy.stats.t.pdf(g, degree_of_freedom), risk_free, minimum)))
    #s=(growth_rate-risk_free)/v[0]

    #return s





def UD_ind(data):
    """
    """
    result = pd.DataFrame()
    result['returns'] = data
    
    lst = ["","","",""]

    for i in range(4,len(data)):

        st = ""

        if data[i-3] < 0:
            st = "D"
        elif data[i-3] >= 0:
            st = "U"

        if data[i-2] < 0:
            st = st + "D"
        elif data[i-2] >= 0:
            st = st + "U"

        if data[i-1] < 0:
            st = st +"D"
        elif data[i-1] >= 0:
            st = st + "U"

        lst.append(st)

    result['UD3indicator'] = lst

    lst = ["", "", "", "", ""]

    for i in range(5,len(data)):

        st = ""

        if data.iloc[i-5] < 0:
            st = "D"
        elif data.iloc[i-5] >= 0:
            st = "U"

        if data.iloc[i-4] < 0:
            st = st + "D"
        elif data.iloc[i-4] >= 0:
            st = st + "U"

        if data.iloc[i-3] < 0:
            st = st +"D"
        elif data.iloc[i-3] >= 0:
            st = st + "U"

        if data.iloc[i-2] < 0:
            st = st +"D"
        elif data.iloc[i-2] >= 0:
            st = st +"U"

        if data.iloc[i-1] < 0:
            st = st + "D"
        elif data.iloc[i-1] >= 0:
            st = st + "U"

        lst.append(st)

    result['UD5indicator'] = lst
    
    return(result)

def trend_ind(data):
    """
    """
    assert data.columns[3] == "Close"
    assert data.columns[7] == "velocity"

    result = data.copy()
    Trend = [None] * len(result)
    for i in range(21, len(result)):
        Trend[i] = tm.mk_test(result['Close'][i-21:i], 'full', window = 21)[0]
    
    Trend2 = [None] * len(result)
    for i in range(21, len(result)):
        t_ind = np.mean(result['velocity'][i-21:i]) * 100
        if t_ind < -0.5:
            Trend2[i] = "decreasing"
        elif t_ind > 0.5:
            Trend2[i] = "increasing"
        else:
            Trend2[i] = "no trend"     
    result = pd.DataFrame([Trend, Trend2]).T
    return(result)

def hidden_states(data):
    data = data.dropna()
    data.replace([np.inf, -np.inf, np.nan], 1, inplace=True)

    close = np.array(data.iloc[1:,3])
    vel = np.array(data.iloc[1:,7])
    acc = np.array(data.iloc[1:,8])
    vol = np.array(data.iloc[1:,4])
    vol_vel = np.array(data.iloc[1:,9])
    vol_acc = np.array(data.iloc[1:,10])


    RSI = np.array(data.iloc[1:,11])
    MA3 = np.array(data.iloc[1:,13])
    STD3 = np.array(data.iloc[1:,14])
    MA5 = np.array(data.iloc[1:,18])
    STD5 = np.array(data.iloc[1:,19])
    MA10 = np.array(data.iloc[1:,23])
    STD10 = np.array(data.iloc[1:,24])
    MA60 = np.array(data.iloc[1:,38])
    STD60 = np.array(data.iloc[1:,39])


    X = np.column_stack([close, vel, acc, vol, vol_vel, vol_acc, RSI, MA3, STD3, MA5, STD5, MA10, STD10, MA60, STD60])
    model = GaussianHMM(n_components=6, covariance_type="diag", n_iter=10000, random_state = 123).fit(X)
    hidden_states = model.predict(X)
    data = data.iloc[1:,:]
    data['States'] = hidden_states
    return(data)



def returns(x, logs = False):

    if type(x) == list:
        returne = np.zeros(len(x))
        for i in range(len(x)):
            returne[i] = ((x[i]/x[i-1])-1)
    elif type(x) == pd.DataFrame:
        returne = x.pct_change().dropna()
        
    else:
        returne = x.pct_change().dropna()
        
    if logs == True:
        returne = np.log(abs(returne))
        #returne = np.log(1+abs(returne)).dropna()
    
    df = pd.DataFrame(data=returne, dtype=np.float64)
    df.columns = [x.name]
    return(df)

def acceleration(x, logs = False):

    
    x = returns(x, logs)
    
    
    if type(x) == list:
        returne = np.zeros(len(x))
        for i in range(len(x)):
            returne[i] = ((x[i]/x[i-1])-1)
    elif type(x) == pd.DataFrame:
        returne = x.pct_change().dropna()
        
    else:
        returne = x.pct_change().dropna()
        
    if logs == True:
        returne = np.log(abs(returne))
        #returne = np.log(1+abs(returne)).dropna()
    
    df = pd.DataFrame(data=returne, dtype=np.float64)
    df.columns = ["acceleration"]
    return(df)

def ma(data, t, typ = "sma"):

    
    if typ == "sma":
        if len(data) < t+1:
            raise ValueError("not enough data")
    
        ma_df = []
        for i in range(len(data)):
    
            mean_reversion_times = np.zeros(t)
    
            for j in range(1,t):
    
                if i >= j:
    
                    mean_reversion_times[j] = np.mean(data[i-j:i])
    
            ma_df.append(mean_reversion_times)
    
        ma_df = pd.DataFrame(ma_df, index = data.index)
        ma_df = ma_df.iloc[1:len(ma_df), 1:len(ma_df.columns)]
        return(ma_df)
    
    elif typ == "ema":

        ma_df = pd.DataFrame(index=data.index, dtype=np.float64)
        ma_df.loc[:,0] = data.ewm(span=t, min_periods=0, adjust=True, ignore_na=False).mean().values.flatten()
        return(ma_df)

def stdev(data, t):

    if len(data) < t+1:
        raise ValueError("not enough data")

    st_df = []
    for i in range(len(data)):

        mean_reversion_times = np.zeros(t)

        for j in range(1,t):

            if i >= j:

                mean_reversion_times[j] = np.std(data[i-j:i])

        st_df.append(mean_reversion_times)

    st_df = pd.DataFrame(st_df, index = data.index)
    st_df = st_df.iloc[1:len(st_df), 1:len(st_df.columns)]
    return(st_df)

def volatility(x, window = 21):

    vol = np.zeros(len(x.index))
    df = pd.DataFrame(data=vol, index=x.index, dtype='float64')
    df.loc[:, 0] = x.rolling(center=False, window=window).std().values.flatten()
    return df


def RSI(data, period):
    
    delta = data.diff().dropna()
    
    u = delta * 0
    d = u.copy()
    
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    
    u[u.index[period-1]] = np.mean( u[:period] ) #first value is sum of avg gains
    u = u.drop(u.index[:(period-1)])
    
    d[d.index[period-1]] = np.mean( d[:period] ) #first value is sum of avg losses
    d = d.drop(d.index[:(period-1)])
    rs = pd.DataFrame.ewm(u, com=period-1, adjust=False).mean() / \
         pd.DataFrame.ewm(d, com=period-1, adjust=False).mean()
    return(100 - 100 / (1 + rs))



def bollinger(self,x):
    """
    

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    smadf = self.sma(x)
    sigmas = self.volatility(x)
    diff = 2 * sigmas
    Upper = smadf.iloc[:, 0].values + diff.iloc[:,0]
    Lower = smadf.iloc[:, 0].values - diff.iloc[:,0]
    df = pd.DataFrame()
    df['Lower'] = Lower
    df['Upper'] = Upper
    return df

def sharpe(self,x):
    """
    

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    mu = x.mean()
    sigma = np.sqrt(x.var())
    return (mu-5) / sigma

def stochastic_oscillator(x):
    mini = min(x.tail(14))
    maxi = max(x.tail(14))
    return((x.iloc[-1]-mini)/(maxi-mini)*100)

# unsure of functionality
def backtest(x, window = 21):
    weights = x.shift(1).rolling(window= window).std().dropna() / np.mean(x)
    return (x * weights.shift(1)).sum()  

def slope(array):
    y = np.array(array)
    x = np.arange(len(y))
    slopes, intercept, r_value, p_value, std_err = linregress(x,y)
    return slopes
    
def var(data, level=5):
    z = norm.ppf(level/100)
    return -(data.mean() + z*data.std(ddof=0))

def cvar(data, level=5):

    confidence_level = 1-(level/100)  # Set the desired confidence level
    sorted_prices = np.sort(data)
    num_samples = len(sorted_prices)
    cvar_index = int((1 - confidence_level) * num_samples)
    cvar = np.mean(sorted_prices[:cvar_index])
    return(cvar)

def drawdown(return_series):
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({"Drawdown": drawdowns})


class financial_analytics:
    
    def __init__(self):
        print("hi")




class tech:
    
    def __init__(self, ticker):
        ########################################################################################################
        # Download base data for long period, daily-1m and monthly-1h
        ########################################################################################################
        self.st = pd.DataFrame()
        t = yf.Ticker(ticker)
        self.st = t.history(start='2020-04-01', end='2024-04-15')
        self.st.index = pd.to_datetime(self.st.index).tz_localize(None)
        self.st_daily = yf.download(ticker, start='2024-04-10', period='1d', interval="1m")
        self.st_month = yf.download(ticker, start='2024-03-01', period='1d', interval="1h")
        ########################################################################################################
        # extract price differentiation
        ########################################################################################################
        self.returns = self.st['Close'].pct_change().dropna()
        self.returns_daily = self.st_daily['Close'].pct_change().dropna()
        self.returns_month = self.st_month['Close'].pct_change().dropna()
        ########################################################################################################
        # extract price accelerations
        ########################################################################################################
        self.acc = self.returns.pct_change().dropna()
        self.acc_daily = self.returns_daily.pct_change().dropna()
        self.acc_month = self.returns_month.pct_change().dropna()
        ########################################################################################################
        # extract volume differentiations
        ########################################################################################################
        self.volume_ret = self.st['Volume'].pct_change().dropna()
        self.volume_ret = self.volume_ret[self.volume_ret != np.inf]
        self.volume_ret_daily = self.st_daily['Volume'].pct_change().dropna()
        self.volume_ret_daily = self.volume_ret_daily[self.volume_ret_daily != np.inf]
        self.volume_ret_month = self.st_month['Volume'].pct_change().dropna()
        self.volume_ret_month = self.volume_ret_month[self.volume_ret_month != np.inf]
        ########################################################################################################
        # extract volume accelerations
        ########################################################################################################
        self.volume_acc = self.volume_ret.pct_change().dropna()
        self.volume_acc_daily = self.volume_ret_daily.pct_change().dropna()
        self.volume_acc_month = self.volume_ret_month.pct_change().dropna()

        ########################################################################################################
        # Moving averages for the long term time series
        ########################################################################################################
        self.ma3_l = self.ma(self.st['Close'], 3)
        self.ma5_l = self.ma(self.st['Close'], 5)
        self.ma10_l = self.ma(self.st['Close'], 10)
        self.ma21_l = self.ma(self.st['Close'], 21)
        self.ma35_l = self.ma(self.st['Close'], 35)
        self.ma60_l = self.ma(self.st['Close'], 60)
        self.ma120_l = self.ma(self.st['Close'], 120)
        ########################################################################################################
        # Moving averages for the last month time series
        ########################################################################################################
        self.ma3_m = self.ma(self.st_month['Close'], 3)
        self.ma5_m = self.ma(self.st_month['Close'], 5)
        self.ma10_m = self.ma(self.st_month['Close'], 10)
        self.ma21_m = self.ma(self.st_month['Close'], 21)
        self.ma35_m = self.ma(self.st_month['Close'], 35)
        self.ma60_m = self.ma(self.st_month['Close'], 60)
        self.ma120_m = self.ma(self.st_month['Close'], 120)
        ########################################################################################################
        # Moving averages for the last 7 days time series
        ########################################################################################################
        self.ma3_s = self.ma(self.st_daily['Close'], 3)
        self.ma5_s = self.ma(self.st_daily['Close'], 5)
        self.ma10_s = self.ma(self.st_daily['Close'], 10)
        self.ma21_s = self.ma(self.st_daily['Close'], 21)
        self.ma35_s = self.ma(self.st_daily['Close'], 35)
        self.ma60_s = self.ma(self.st_daily['Close'], 60)
        self.ma120_s = self.ma(self.st_daily['Close'], 120)
        
        ########################################################################################################
        # Empirical distribution of 
        ########################################################################################################
        
        # are the means and stds the same? are any of the means above 0 or below 0 and is the current situation good, positive mean and low std?
        self.ed_1m_ret = self.ed(self.returns_month,100)
        self.ed_3m_ret = self.ed(self.returns.tail(90),100)
        self.ed_1y_ret = self.ed(self.returns.tail(200),100)
        self.ed_3y_ret = self.ed(self.returns.tail(780),100)
        self.ed_5y_ret = self.ed(self.returns.tail(1250),100)
        
        # expected price range
        self.ed_3y_price = self.ed(self.st.tail(780)['Close'],100) # we want to know mu and sigma
          
            
        ########################################################################################################
        # Simple Volume indicators, we just want to know if it's increasing regularly
        ######################################################################################################## 
            
        ####
        print("Average volume past 5 years: ", np.mean(self.st['Volume'].tail(1250))) 
        print("Average volume past 3 years: ", np.mean(self.st['Volume'].tail(780)))
        print("Average volume past 1 year: ", np.mean(self.st['Volume'].tail(250)))
        
        print("Average return past 5 years: ", np.mean(self.returns.tail(1250))) 
        print("Average return past 3 years: ", np.mean(self.returns.tail(780)))
        print("Average return past 1 year: ", np.mean(self.returns.tail(250)))
        
        ###
        print("Volume/return increase overall: ",  np.mean(self.volume_ret) * np.mean(self.returns))
        print("Volume/return increase past month: ",  np.mean(self.volume_ret_month) * np.mean(self.returns_month))
        print("Volume/return increase past week: ",  np.mean(self.volume_ret_daily) * np.mean(self.returns_daily))    

        self.indicator(self.returns)
        
class portfolio:
    
    def __init__(self, file):
        
        self.portf = pd.read_excel(open(file, 'rb'))
        self.portf = self.portf['Symbols'].tolist()
        
        self.scores = pd.DataFrame(0.0, index = self.portf, columns = ["financials", "holders", "dividend", "technical", "collective"])
        
        today = date.today() 
        self.today = datetime(today.year,today.month,today.day) #today
        self.one = datetime(today.year-1,today.month,today.day) #one year ago
        self.three = datetime(today.year-3,today.month,today.day) #three years ago
        self.five = datetime(today.year-5,today.month,today.day) #five years ago
        self.ten = datetime(today.year-10,today.month,today.day) #ten years ago
        self.twenty = datetime(today.year-20,today.month,today.day) #twenty years ago
        
    def dividend_score(self, ticker):
        try:
            self.dp = 0
            t = yf.Ticker(ticker)   
            div = t.dividends
            div = div.to_frame()
            div = div.tz_localize(None)
            price = td.sdata(ticker, str(div.index[0])[0:10], self.today)

            if len(div) > 0:    

                divs = np.zeros(len(div))
                dive = np.zeros(len(div))
                dives = np.zeros(len(div))

                for i in range(1,len(div)):
                    divs[i] = div.iloc[i,0] / div.iloc[i-1,0]

                for i in range(1, len(div)):
                    dive[i] = div.iloc[i,0] / price.iloc[i]
                    if i > 1:
                        dives[i] = dive[i] / dive[i-1]

                if np.mean(divs) > 1.0:
                    self.dp = self.dp + 1

                if np.mean(dives) > 1.0:
                    self.dp = self.dp + 1
        except:
            self.dp = 0
            
        for i in range(len(self.scores)):
            if self.scores.index[i] == ticker:
                self.scores.iloc[i,2] = self.dp
        
    def holder_score(self,ticker):
        
        self.hp = 0
        t = yf.Ticker(ticker)   
        self.mh = t.major_holders
        self.ih = t.institutional_holders 
    
        if float(self.mh.iloc[0,0][0:3]) > 3:
            self.hp = self.hp + 1

        if float(self.mh.iloc[1,0][0:3]) > 50:
            self.hp = self.hp + 1

        if float(self.mh.iloc[2,0][0:3]) > 50:
            self.hp = self.hp + 1

        if float(self.mh.iloc[3,0]) > 1000:
            self.hp = self.hp + 1

        GIH = ["Vanguard Group Inc", "Blackrock Inc.", "Berkshire Hathaway, Inc", "State Street Corporation", "Morgan Stanley", "Norges Bank Investment Management", "JP Morgan Chase & Company", "Goldman Sachs Group Inc", "Bank of America Corporation", "Charles Schwab Investment Management, Inc.", "Bank Of New York Mellon Corporation", "Citadel Advisors Llc", "Dimensional Fund Advisors LP"]

        for i in range(len(GIH)):
            for j in range(len(self.ih['Holder'])):
                if GIH[i] == self.ih['Holder'][j]:
                    self.hp = self.hp + 1
        
        for i in range(len(self.scores)):
            if self.scores.index[i] == ticker:
                self.scores.iloc[i,1] = self.hp

    def financial_score(self,ticker):
        
        self.fp = 0
        t = yf.Ticker(ticker)    
        bs = t.balance_sheet
        fin = t.financials
        cs = t.cashflow
        for i in range(3):
            #
            if 'Net Debt' in bs.index:
                if bs[bs.columns[i]]['Net Debt'] / bs[bs.columns[i+1]]['Net Debt'] < 1:
                    self.fp = self.fp + 1
                elif bs[bs.columns[i]]['Net Debt'] / bs[bs.columns[i+1]]['Net Debt'] > 1:
                    self.fp = self.fp - 1
            #
            if bs[bs.columns[i]]['Total Debt'] / bs[bs.columns[i+1]]['Total Debt'] < 1:
                self.fp = self.fp + 1
            elif bs[bs.columns[i]]['Total Debt'] / bs[bs.columns[i+1]]['Total Debt'] > 1:
                self.fp = self.fp - 1
            #
            if 'Long Term Debt' in bs.index:
                if bs[bs.columns[i]]['Long Term Debt'] / bs[bs.columns[i+1]]['Long Term Debt'] < 1:
                    self.fp = self.fp + 1
                elif bs[bs.columns[i]]['Long Term Debt'] / bs[bs.columns[i+1]]['Long Term Debt'] > 1:
                    self.fp = self.fp - 1
            #
            if 'Current Liabilities' in bs.index:
                if bs[bs.columns[i]]['Current Liabilities'] / bs[bs.columns[i+1]]['Current Liabilities'] < 1:
                    self.fp = self.fp + 1
                elif bs[bs.columns[i]]['Current Liabilities'] / bs[bs.columns[i+1]]['Current Liabilities'] > 1:
                    self.fp = self.fp - 1
            #
            if 'Current Debt' in bs.index and bs[bs.columns[i+1]]['Current Debt'] > 0:
                if bs[bs.columns[i]]['Current Debt'] / bs[bs.columns[i+1]]['Current Debt'] < 1:
                    self.fp = self.fp + 1
                elif bs[bs.columns[i]]['Current Debt'] / bs[bs.columns[i+1]]['Current Debt'] > 1:
                    self.fp = self.fp - 1
            #
            if 'Total Assets' in bs.index:
                if bs[bs.columns[i]]['Total Assets'] / bs[bs.columns[i+1]]['Total Assets'] > 1:
                    self.fp = self.fp + 1
                elif bs[bs.columns[i]]['Total Assets'] / bs[bs.columns[i+1]]['Total Assets'] < 1:
                    self.fp = self.fp - 1
            #
            if 'Current Assets' in bs.index:
                if bs[bs.columns[i]]['Current Assets'] / bs[bs.columns[i+1]]['Current Assets'] > 1:
                    self.fp = self.fp + 1
                elif bs[bs.columns[i]]['Current Assets'] / bs[bs.columns[i+1]]['Current Assets'] < 1:
                    self.fp = self.fp - 1
            
            if 'Total Capitalization' in bs.index:
                if bs[bs.columns[i]]['Total Capitalization'] / bs[bs.columns[i+1]]['Total Capitalization'] > 1:
                    self.fp = self.fp + 1
                elif bs[bs.columns[i]]['Total Capitalization'] / bs[bs.columns[i+1]]['Total Capitalization'] < 1:
                    self.fp = self.fp - 1
                    
            if 'Total Liabilities Net Minority Interest' in bs.index:
                if bs[bs.columns[i]]['Total Liabilities Net Minority Interest'] / bs[bs.columns[i+1]]['Total Liabilities Net Minority Interest'] > 1:
                    self.fp = self.fp + 1
                elif bs[bs.columns[i]]['Total Liabilities Net Minority Interest'] / bs[bs.columns[i+1]]['Total Liabilities Net Minority Interest'] < 1:
                    self.fp = self.fp - 1

            if 'Net Income' in fin.index:
                if fin[fin.columns[i]]['Net Income'] / fin[fin.columns[i+1]]['Net Income'] > 1:
                    self.fp = self.fp + 1
                elif fin[fin.columns[i]]['Net Income'] / fin[fin.columns[i+1]]['Net Income'] < 1:
                    self.fp = self.fp - 1

            if 'Total Revenue' in fin.index:
                if fin[fin.columns[i]]['Total Revenue'] / fin[fin.columns[i+1]]['Total Revenue'] > 1:
                    self.fp = self.fp + 1
                elif fin[fin.columns[i]]['Total Revenue'] / fin[fin.columns[i+1]]['Total Revenue'] < 1:
                    self.fp = self.fp - 1

            if 'Operating Revenue' in fin.index:
                if fin[fin.columns[i]]['Operating Revenue'] / fin[fin.columns[i+1]]['Operating Revenue'] > 1:
                    self.fp = self.fp + 1
                elif fin[fin.columns[i]]['Operating Revenue'] / fin[fin.columns[i+1]]['Operating Revenue'] < 1:
                    self.fp = self.fp - 1
                    
            if 'Normalized Income' in fin.index:
                if fin[fin.columns[i]]['Normalized Income'] / fin[fin.columns[i+1]]['Normalized Income'] > 1:
                    self.fp = self.fp + 1
                elif fin[fin.columns[i]]['Normalized Income'] / fin[fin.columns[i+1]]['Normalized Income'] < 1:
                    self.fp = self.fp - 1

            if 'Issuance Of Debt' in cs.index:
                if str(cs[cs.columns[i]]['Issuance Of Debt']) == "NaN":
                    cs[cs.columns[i]]['Issuance Of Debt'] = 0

            if 'Long Term Debt Issuance' in cs.index:
                if str(cs[cs.columns[i]]['Long Term Debt Issuance']) == "NaN":
                    cs[cs.columns[i]]['Long Term Debt Issuance'] = 0

            if cs[cs.columns[i]]['Operating Cash Flow'] > cs[cs.columns[i+1]]['Operating Cash Flow']:
                self.fp = self.fp + 1

            if 'Long Term Debt Payments' in cs.index and 'Long Term Debt Issuance' in cs.index:
                if abs(cs[cs.columns[i]]['Long Term Debt Payments']) > 0.5 * cs[cs.columns[i]]['Long Term Debt Issuance']:
                    self.fp = self.fp + 1

            if 'Repayment Of Debt' in cs.index and 'Issuance Of Debt' in cs.index:
                if abs(cs[cs.columns[i]]['Repayment Of Debt']) > 0.5 * cs[cs.columns[i]]['Issuance Of Debt']:
                    self.fp = self.fp + 1
        
        for i in range(len(self.scores)):
            if self.scores.index[i] == ticker:
                self.scores.iloc[i,0] = self.fp
    
    def technical_score(self,ticker):
        
        self.tp = 0
        df = pd.DataFrame()
        t = yf.Ticker(ticker)
        try:
            df = t.history(start=self.twenty, end=self.today)['Close']
            df.index = pd.to_datetime(df.index).tz_localize(None)
        except:
            df = t.history(start=self.five, end=self.today)['Close']
            df.index = pd.to_datetime(df.index).tz_localize(None)


        self.means = []
        self.stds = []

        l = int(round(len(df.index)/5,0))

        for i in range(5):
            if i < 4:
                self.means.append(np.mean(df[df.index > df.index[l*(i+1)]]))
                self.stds.append(np.std(df[df.index > df.index[l*(i+1)]]))
            if i == 4:
                self.means.append(np.mean(df[df.index > df.index[l*(i+1) - 150]]))
                self.stds.append(np.std(df[df.index > df.index[l*(i+1) - 150]]))
        
        if self.stds[0] < self.stds[len(self.stds)-1] or self.stds[0] < self.stds[len(self.stds)-2]:
            self.tp = self.tp + 1
        if self.means[0] < self.means[len(self.means)-1] or self.means[0] < self.means[len(self.means)-2]:
            self.tp = self.tp + 1
            
        if (self.means[len(self.means)-1]-5) / self.stds[len(self.stds)-1] > 1:
            self.tp = self.tp + 1
        elif (self.means[len(self.means)-1]-5) / self.stds[len(self.stds)-1] < 1:
            self.tp = self.tp - 1
        
        for i in range(len(self.scores)):
            if self.scores.index[i] == ticker:
                self.scores.iloc[i,3] = self.tp
    
    def score(self):
        
        for ticker in self.portf:
            
            try:
                self.financial_score(ticker)
                self.holder_score(ticker)
                self.dividend_score(ticker)
                
            except:
                for i in range(len(self.scores)):
                    
                    if self.scores.index[i] == ticker:
                        
                        self.scores.iloc[i,0] = 0
                        self.scores.iloc[i,1] = 0
                        self.scores.iloc[i,2] = 0
                
            try:
                
                self.technical_score(ticker)
    
            except:
            
                for i in range(len(self.scores)):
                    
                    if self.scores.index[i] == ticker:
                        
                        self.scores.iloc[i,3] = 0
            
            for i in range(len(self.scores)):
                
                 if self.scores.index[i] == ticker:
                    
                    if self.scores.iloc[i,0] > 8.0:

                        self.scores.iloc[i,4] = self.scores.iloc[i,4] + 1

                    if self.scores.iloc[i,1] > 7.0:

                        self.scores.iloc[i,4] = self.scores.iloc[i,4] + 1

                    self.scores.iloc[i,4] = self.scores.iloc[i,4] + 0.6 * self.scores.iloc[i,2] + 0.4 * self.scores.iloc[i,3]

        def narrow_scores(self):
            self.scores = self.scores[self.scores['financials'] > np.mean(self.scores['financials'])]
            self.scores = self.scores[self.scores['holders'] > np.mean(self.scores['holders'])]
            self.scores = self.scores[self.scores['dividend'] > 0.0]         


    
class financial_analytics:
    
    def __init__(self):
        self.initials = "pp"
    
    def fractionated_interest(self, i,m):
        ir = m *((1+i)**(1/m)-1)
        return ir
    
    def summarized_interest(self, i,m):
        ir = (1+i/m)**m - 1
        return ir
    
    def continous_interest(self, i,t):
        cont = 2.7182818**(math.log(1+i)*t)
        return cont
    
    def continous_discounting(self, i,n,t):
        cont = 2.7182818**(-math.log(1+i)*(n-t))
        return cont
    
    def presentvalue_c(self, c,i,t):
        pv = c * (1+i)**t
        return pv
    
    def presentvalue_b(self, b,i,t):
        pv = b*(1/(1+i))**t
        return pv
    
    #time value of c
    def time_value_c(self, c,b,i):
        m = 0
        M = 0
        for j in range(0,len(c)):
            m = (c[j] - b[j])*(1+i[j])
            M = M+m
        return(M)
    
    #time value of v
    def time_value_v(self, c,b,i):
        m = 0
        M = 0
        for j in range(0,len(c)):
            m = (c[j] - b[j])*1/(1+i[j])
            M = M+m
        return(M)
    
    #essentially difference between timevalues
    def fair_stream(self, c,b,i,n):
        t = len(c)
        n = t
        U = 0
        V = 0
        for j in range(0,n):
            U = U + (c[j] - b[j])*(1+i[j])**(t-j)
            V = V + (b[j] - c[j])*(1+i[j])**(-(t+1-t))
        return(U+V)
    
    
    def yrlygrowth(self, total_growth, years):
        """
        Determine the annual growth from the growth over an
        arbitrary time span.
        """
        return math.exp(math.log(total_growth) / years)
    
    def pvannuity(self, option = "annual", rate = 1, nrmts = 1, amt=1):
        if option == "annual":
            amt * (1. - (1. + rate)**-nrmts) / rate
            
        elif option == "semi-annual":
            amt * (1. - (1. + rate)**-nrmts) / rate
            
        elif option == "quarter":
            amt * (1. - (1. + rate)**-nrmts) / rate
            
        return amt * (1. - (1. + rate)**-nrmts) / rate
    
    def loanpayment(self, amount, Interest_rate, nr_payments):
        return float(amount) / self.pvannuity(Interest_rate, nr_payments)
    
    
    def inst_to_ann(self, r):
        """
        Convert an instantaneous interest rate to an annual interest rate
        """
        return np.expm1(r)
    
    def ann_to_inst(self, r):
        """
        Convert an instantaneous interest rate to an annual interest rate
        """
        return np.log1p(r)
    
    # Define the interest rate simulation based on the CIR model
    def cir(self, n_years = 10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05, steps_per_year=12, r_0=None):
        """
        Generate random interest rate evolution over time using the CIR model
        b and r_0 are assumed to be annualized rates
        output values are annualized rates as well
        """
        if r_0 is None: r_0 = b 
        r_0 = self.ann_to_inst(r_0)
        dt = 1 / steps_per_year
        num_steps = int(n_years * steps_per_year) + 1 # because n_years might be a float
        
        shock = np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
        rates = np.empty_like(shock)
        rates[0] = r_0
        for step in range(1, num_steps):
            r_t = rates[step-1]
            d_r_t = a*(b-r_t)*dt + sigma*np.sqrt(r_t)*shock[step]
            rates[step] = abs(r_t + d_r_t) # just in case of roundoff errors going negative
            
        return pd.DataFrame(data=self.inst_to_ann(rates), index=range(num_steps))
    
    
    def annuity_in_arrear(self, v0, t, i):
        res = (1-(1/(1+i))**t)/i
        return res
    
    def annuity_due(self, v0,t,i):
        res = (1-(1/(1+i))**t)/(1-(1/(1+i)))
        return res
    
    # needs correcting
    def deferred_annuity_in_arrear(self, v0, t, k, i):
        if t < k:
            (1/(1+i))**k * self.annuity_in_arrear(v0,k,i)
        elif t >= k:
            self.annuity_in_arrear(v0,t,i) - self.annuity_in_arrear(v0, k ,i)
            
    def deferred_annuity_due(self, v0,t,k,i):
        if t < k:
            (1/(1+i))**k * self.annuity_in_due(v0,k,i)
        elif t >= k:
            self.annuity_in_arrear(v0,t,i) - self.annuity_in_due(v0, k ,i)
            
    def cont_a(self, i,n):
        a = (1 - self.cont_dis(i,n,0))/math.log(1+i)
        return a
    
    def cont_def_a(self, i,n,k):
        a_n = (1 - self.cont_dis(i,n,0))/math.log(1+i)
        a_k = (1 - self.cont_dis(i,k,0))/math.log(1+i)
        return a_n - a_k




            

