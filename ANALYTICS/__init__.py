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



import yfinance as yf
import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt

from datetime import datetime, date, timedelta, timezone
import sqlalchemy as sa

from scipy.stats import norm
from scipy.stats import linregress
from scipy import integrate
from statsmodels.tsa.stattools import adfuller





class data_analytics:
    
    def __init__(self, data = None):
        
        try:
            
            if type(data) == pd.DataFrame:
                self.dataset = data

            elif isinstance(data, Simulations): # Hvis vi har simuleringer
                self.dataset = data.paths # Så tag paths

            #elif isinstance(data, Credit):

            #elif not isinstance(data, Simulations):
                #self.dataset = data

            else: 
                self.dataset = data
        
        #take into account that the Simulations class might be empty or none existant
        except NameError:
        
            self.dataset = data
        
        
        # take into account that datasets can be empty
        if len(self.dataset) <= 1:
                raise ValueError('This class needs a dataset to proceed')
        
        self.median()
        self.mean()
        self.variance()
        self.std()
        self.min = min(self.dataset) # skal rettes
        self.max = max(self.dataset) # skal rettes
        
    def median(self):
        
        if type(self.dataset) == pd.DataFrame:
            
            self.median = pd.DataFrame(columns = self.dataset.columns)
            self.median.loc['Median'] = None
            
            for i in range(len(self.dataset.columns)):
                
                lst = list(self.dataset[i])
                sortedLst = sorted(lst)
                lstLen = len(lst)
                index = (lstLen - 1) // 2

                if (lstLen % 2):
                    
                    self.median.iloc[0,i] = sortedLst[index]
                    
                else:
                    
                    self.median.iloc[0,i] = (sortedLst[index] + sortedLst[index + 1])/2.0
        
        elif type(self.dataset) == list:
            
            sortedLst = sorted(self.dataset)
            lstLen = len(self.dataset)
            index = (lstLen - 1) // 2

            if (lstLen % 2):
                
                self.median = sortedLst[index]
                
            else:
                
                self.median = (sortedLst[index] + sortedLst[index + 1])/2.0

    def mean(self):
       
        if type(self.dataset) == pd.DataFrame:
            
            self.mean = pd.DataFrame(columns = self.dataset.columns)
            
            self.mean.loc['Mean'] = None
            
            for i in range(len(self.dataset.columns)):
                
                self.mean.iloc[0,i] = np.float64(sum(self.dataset[i])/len(self.dataset))
        
        elif type(self.dataset) == list:
            
            self.mean = sum(self.dataset)/len(self.dataset)

        return(self.mean)
    
    
    def ma(data, t):

        if len(data) < t+1:
            raise ValueError("not enough data")

        mrt_df = []
        for i in range(len(data)):

            mean_reversion_times = np.zeros(t)

            for j in range(1,t):

                if i >= j:

                    mean_reversion_times[j] = np.mean(data[i-j:i])

            mrt_df.append(mean_reversion_times)

        mrt_df = pd.DataFrame(mrt_df, index = data.index)
        mrt_df = mrt_df.iloc[1:len(mrt_df), 1:len(mrt_df.columns)]
        return(mrt_df)
    
    
    
        
    def variance(self, typ = "population"):
        
        self.typ = typ
       
        if type(self.dataset) == pd.DataFrame:
            
            self.var = pd.DataFrame(columns = self.dataset.columns)
            self.var.loc['Variance'] = None
            
            for i in range(len(self.dataset.columns)):
                self.var.iloc[0,i] = 0
                
                for j in range(len(self.dataset)):
                    
                    self.var.iloc[0,i] = self.var.iloc[0,i] + (self.dataset.iloc[j,i] - self.mean.iloc[0,i])**2
                
                if typ == "population":
                    
                    self.var.iloc[0,i] = self.var.iloc[0,i] / len(self.dataset)
                    
                elif typ == "sample":
                    
                    self.var.iloc[0,i] = self.var.iloc[0,i] / (len(self.dataset) - 1)
            
        
        elif type(self.dataset) == list:
            
            self.var = None
            
            for i in range(len(self.dataset)):
                
                self.var = self.var + (self.dataset[i]-self.mean)**2
            
            if typ == "population":
                
                self.var = self.var / len(self.dataset)
                
            elif typ == "sample":
                
                self.var = self.var / (len(self.dataset) - 1)
        
        return(self.var)
        
    def std(self, typ = "population"):
        
        self.typ = typ
        self.variance(self.typ)
        
        self.stdev = self.var ** (1/2)
        self.stdev.index = ['Standard Deviation']
        return(self.stdev)
    
    def stdev(data, t):

        if len(data) < t+1:
            raise ValueError("not enough data")

        mrt_df = []
        for i in range(len(data)):

            mean_reversion_times = np.zeros(t)

            for j in range(1,t):

                if i >= j:

                    mean_reversion_times[j] = np.std(data[i-j:i])

            mrt_df.append(mean_reversion_times)

        mrt_df = pd.DataFrame(mrt_df, index = data.index)
        mrt_df = mrt_df.iloc[1:len(mrt_df), 1:len(mrt_df.columns)]
        return(mrt_df)
    
    
    def slope(self, length = 1):
        
        if type(self.dataset) == pd.DataFrame:
            
            self.slope = pd.DataFrame(None,index = self.dataset.index, columns = self.dataset.columns)
            
            for i in range(len(self.dataset.columns)):
                for j in range(0+length,len(self.dataset)):
                    
                    self.slope.iloc[j,i] = (self.dataset.iloc[j,i] - self.dataset.iloc[j-length,i]) / (self.dataset.index[j] - self.dataset.index[j-length])

    def stationarity(self):
        result = adfuller(self.dataset[0])
        res = "Stationary"
        # = pd.DataFrame(result[4].items())[1]
        if result[1] > 0.1:
            res = "Non-stationary"
        return res
     
    # estimate best probability distribution
    
    # estimate parameters
    
    # regression models
    
    #empirical distribution
     
    def empirical(self, data, bins = 10):
    
        bine = np.linspace(min(data), max(data), bins)
        counts = []
        
        for i in range(0,len(bine)):
    
            bine[i] = round(bine[i],3)
            count = 0
    
            for j in range(len(data)):
    
                if data.iloc[j] >= bine[i-1] and data.iloc[j] < bine[i]:
    
                    count = count + 1
    
            counts.append(count/len(data))
    
        counts_df = pd.DataFrame(counts, index = bine)
        return(counts_df)
    
    
    
    
    
    
class analysis:
    
    def __init__(self):
        print("hello world")
        today = date.today() 
        self.today = datetime(today.year,today.month,today.day) #today
        self.one = datetime(today.year-1,today.month,today.day) #one year ago
        self.three = datetime(today.year-3,today.month,today.day) #three years ago
        self.five = datetime(today.year-5,today.month,today.day) #five years ago
        self.ten = datetime(today.year-10,today.month,today.day) #ten years ago
        self.twenty = datetime(today.year-20,today.month,today.day) #twenty years ago
        
    def updown(data):
     
         u_d_count = []
         
         for i in range(len(data)):
             
             if data.iloc[i] < 0:
                 
                 u_d_count.append(-1)
             
             elif data.iloc[i] > 0:
                 
                 u_d_count.append(1)
                 
             else:
                 
                 u_d_count.append(0)
                 
         return(u_d_count) 
   
    def updown_prob(data, k = 1):
        
        prob = pd.DataFrame(0,index = ['up', 'down'], columns = ['up', 'down'])
        
        # columns are "future"
        # rows are "past"
        
        for i in range(len(data)):
        
            if data[i] > 0 and data[i-k] > 0:
        
                prob['up']['up'] = prob['up']['up'] + 1
        
            elif data[i] > 0 and data[i-k] < 0:
        
                prob['up']['down'] = prob['up']['down'] + 1
        
            elif data[i] < 0 and data[i-k] < 0:
        
                prob['down']['down'] = prob['down']['down'] + 1
        
            elif data[i] < 0 and data[i-k] > 0:
        
                prob['down']['up'] = prob['down']['up'] + 1
        
        prob = prob / len(data)
        return(prob)
   
    #markov chain probabilities

    def counts(data):
        ind = ['U','UU','UUU','UUD','UD','UDU','UDD','D','DU','DUU','DUD','DD','DDU','DDD'] #index
        
        counts = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        
        for i in range(1, len(data)):
    
            # if [-1] is up. Add +1 to counts[0]
            if data[i-1] > 0:   
                counts[0] = counts[0] + 1
    
            # UU
            if data[i-1] > 0 and data[i-2] > 0:
                counts[1] = counts[1] + 1
    
            # UUU
            if data[i-1] > 0 and data[i-2] > 0 and data[i-3] > 0:
                counts[2] = counts[2] + 1
    
            # UUD
            if data[i-1] > 0 and data[i-2] > 0 and data[i-3] < 0:
                counts[3] = counts[3] + 1
    
            # UD
            if data[i-1] > 0 and data[i-2] < 0:
                counts[4] = counts[4] + 1
    
            # UDU
            if data[i-1] > 0 and data[i-2] < 0 and data[i-3] < 0:
                counts[5] = counts[5] + 1
    
            # UDD
            if data[i-1] > 0 and data[i-2] < 0 and data[i-3] > 0:
                counts[6] = counts[6] + 1
    
    
            # if [-1] is down. Add +1 to counts[7]
            if data[i-1] < 0:   
                counts[7] = counts[7] + 1
    
            # DU
            if data[i-1] < 0 and data[i-2] > 0:
                counts[8] = counts[8] + 1
    
            # DUU
            if data[i-1] < 0 and data[i-2] > 0 and data[i-3] > 0:
                counts[9] = counts[9] + 1
    
            # DUD
            if data[i-1] < 0 and data[i-2] > 0 and data[i-3] < 0:
                counts[10] = counts[10] + 1
    
            # DD
            if data[i-1] < 0 and data[i-2] < 0:
                counts[11] = counts[11] + 1
    
            # DDU
            if data[i-1] < 0 and data[i-2] < 0 and data[i-3] < 0:
                counts[12] = counts[12] + 1
    
            # DDD
            if data[i-1] < 0 and data[i-2] < 0 and data[i-3] > 0:
                counts[13] = counts[13] + 1
            
        count = pd.DataFrame(counts, index = ind, columns = ['count']) # count table
        return(count)
    
    def probs(data):
        ind = ['U','UU','UUU','UUD','UD','UDU','UDD','D','DU','DUU','DUD','DD','DDU','DDD'] #index
        
        up = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        down = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        
        for i in range(1, len(data)):
            
            if data[i] > 0:
    
                # if [-1] is up. Add +1 to counts[0]
                if data[i-1] > 0:   
                    up[0] = up[0] + 1
    
                # UU
                if data[i-1] > 0 and data[i-2] > 0:
                    up[1] = up[1] + 1
    
                # UUU
                if data[i-1] > 0 and data[i-2] > 0 and data[i-3] > 0:
                    up[2] = up[2] + 1
    
                # UUD
                if data[i-1] > 0 and data[i-2] > 0 and data[i-3] < 0:
                    up[3] = up[3] + 1
    
                # UD
                if data[i-1] > 0 and data[i-2] < 0:
                    up[4] = up[4] + 1
    
                # UDU
                if data[i-1] > 0 and data[i-2] < 0 and data[i-3] < 0:
                    up[5] = up[5] + 1
    
                # UDD
                if data[i-1] > 0 and data[i-2] < 0 and data[i-3] > 0:
                    up[6] = up[6] + 1
    
    
                # if [-1] is down. Add +1 to counts[7]
                if data[i-1] < 0:   
                    up[7] = up[7] + 1
    
                # DU
                if data[i-1] < 0 and data[i-2] > 0:
                    up[8] = up[8] + 1
    
                # DUU
                if data[i-1] < 0 and data[i-2] > 0 and data[i-3] > 0:
                    up[9] = up[9] + 1
    
                # DUD
                if data[i-1] < 0 and data[i-2] > 0 and data[i-3] < 0:
                    up[10] = up[10] + 1
    
                # DD
                if data[i-1] < 0 and data[i-2] < 0:
                    up[11] = up[11] + 1
    
                # DDU
                if data[i-1] < 0 and data[i-2] < 0 and data[i-3] < 0:
                    up[12] = up[12] + 1
    
                # DDD
                if data[i-1] < 0 and data[i-2] < 0 and data[i-3] > 0:
                    up[13] = up[13] + 1
            
            elif data[i] < 0:
                
                            # if [-1] is up. Add +1 to counts[0]
                if data[i-1] > 0:   
                    down[0] = down[0] + 1
    
                # UU
                if data[i-1] > 0 and data[i-2] > 0:
                    down[1] = down[1] + 1
    
                # UUU
                if data[i-1] > 0 and data[i-2] > 0 and data[i-3] > 0:
                    down[2] = down[2] + 1
    
                # UUD
                if data[i-1] > 0 and data[i-2] > 0 and data[i-3] < 0:
                    down[3] = down[3] + 1
    
                # UD
                if data[i-1] > 0 and data[i-2] < 0:
                    down[4] = down[4] + 1
    
                # UDU
                if data[i-1] > 0 and data[i-2] < 0 and data[i-3] < 0:
                    down[5] = down[5] + 1
    
                # UDD
                if data[i-1] > 0 and data[i-2] < 0 and data[i-3] > 0:
                    down[6] = down[6] + 1
    
    
                # if [-1] is down. Add +1 to counts[7]
                if data[i-1] < 0:   
                    down[7] = down[7] + 1
    
                # DU
                if data[i-1] < 0 and data[i-2] > 0:
                    down[8] = down[8] + 1
    
                # DUU
                if data[i-1] < 0 and data[i-2] > 0 and data[i-3] > 0:
                    down[9] = down[9] + 1
    
                # DUD
                if data[i-1] < 0 and data[i-2] > 0 and data[i-3] < 0:
                    down[10] = down[10] + 1
    
                # DD
                if data[i-1] < 0 and data[i-2] < 0:
                    down[11] = down[11] + 1
    
                # DDU
                if data[i-1] < 0 and data[i-2] < 0 and data[i-3] < 0:
                    down[12] = down[12] + 1
    
                # DDD
                if data[i-1] < 0 and data[i-2] < 0 and data[i-3] > 0:
                    down[13] = down[13] + 1
            
        count = pd.DataFrame(up, index = ind, columns = ['up']) # count table
        count['down'] = down
        return(count)
   
    
        def moving_averages(data, t):
            
            if len(data) < t+1:
                raise ValueError("not enough data")
                
            mrt_df = []
            for i in range(len(data)):
                
                mean_reversion_times = np.zeros(t)
                
                for j in range(1,t):
                
                    if i >= j:
                    
                        mean_reversion_times[j] = np.mean(data[i-j:i])
                
                mrt_df.append(mean_reversion_times)
        
            mrt_df = pd.DataFrame(mrt_df, index = data.index)
            mrt_df = mrt_df.iloc[1:len(mrt_df), 1:len(mrt_df.columns)]
            #lst[0] = list(x)
            return(mrt_df)
        
        def mean_sizes(data, t):
            
            sizes = moving_averages(data, t)
            for i in range(len(sizes)):
                
                for j in range(len(sizes.columns)):
                    
                    sizes.iloc[i,j] = data.iloc[i] - sizes.iloc[i,j] 
                    
            return(sizes)
        
        def mean_size_dist(data, t):
            
            ms = mean_sizes(data,t)
            msd = pd.DataFrame(0, index = ['upper+2s', 'upper-m', 'upper-2s','lower+2s','lower-m','lower-2s'], columns = ms.columns)
            for ti in range(1, len(ms.columns)):
                
                msp = ms[ti+1][ms[ti+1] > 0]
                msn = ms[ti+1][ms[ti+1] < 0]
                
                msd[ti+1]['upper+2s'] = np.mean(msp) + 2 * np.std(msp)
                msd[ti+1]['upper-m'] = np.mean(msp)
                msd[ti+1]['upper-2s'] = min(msp)
                
                msd[ti+1]['lower+2s'] = max(msn)
                msd[ti+1]['lower-m'] = np.mean(msn)
                msd[ti+1]['lower-2s'] = np.mean(msn) - 2 * np.std(msn)
                
            return(msd)
                
        def mean_times(data, t):
            ms = mean_sizes(data, t)
            mrt = pd.DataFrame(index = ["mrt"], columns = ms.columns)
            for i in range(len(ms.columns)):
                n = 0
                for j in range(len(ms)):
                    if ms.iloc[j,i] > 0 and ms.iloc[j-1,i] < 0:
                        n = n + 1
                    elif ms.iloc[j,i] < 0 and ms.iloc[j-1,i] > 0:
                        n = n + 1
                    else:
                        n = n
                mrt[i+1]['mrt'] = len(ms) / (n+1)
            return(mrt)      
        
        def month_probabilities(self,Tickers):
            p = pd.DataFrame(0.0, index = ['p'], columns = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
            for j in range(1,13):
                russel_prob = 0.0
            
                for tick in Tickers:
            
                    Aktie = sdata(tick, '2000-01-01', self.today)
                    aar = list(Aktie.index)
                    for i in range(len(aar)):
                        aar[i] = str(aar[i])[0:4]
            
                    aar = list(dict.fromkeys(aar))
            
                    r_prob = 0.0
            
                    if len(str(j)) == 2 and (j-1) != 9:
                        l_s = "-"+str(j) + "-01"
                        u_s = "-"+str(j-1) + "-01"
                    elif len(str(j)) == 2 and (j-1) == 9:
                        l_s = "-"+str(j) + "-01"
                        u_s = "-0"+str(j-1) + "-01"
                    elif len(str(j)) != 2 and (j-1) != 0:
                        l_s = "-0"+str(j)+"-01"
                        u_s = "-0"+str(j-1)+"-01"
                    elif len(str(j)) != 2 and (j-1) == 0:
                        l_s = "-01-31"
                        u_s = "-01-01"
            
            
            
            
                    for i in range(1,len(aar)-1):
            
                        june_july = Aktie[Aktie.index < str(aar[i]) + l_s]
                        june_july = june_july[june_july.index >= str(aar[i]) + u_s]
                        if june_july.iloc[0] < june_july.iloc[len(june_july)-1]:
                            r_prob = r_prob + 1.0
            
                    russel_prob = russel_prob + r_prob / len(aar)
            
                p.iloc[0,j-1] = np.float64(russel_prob / len(Tickers))
            
            return(p)

        def extremum_probabilities(self,Tickers):
            probabilities = pd.DataFrame(index = ['min', 'max'], columns = ['7dage', '1måned'])
            probabilities.iloc[0,0] = 0.0
            probabilities.iloc[1,0] = 0.0
            probabilities.iloc[0,1] = 0.0
            probabilities.iloc[1,1] = 0.0
        
            for tick in Tickers:
        
                Aktie = sdata(tick, '2014-01-01', self.today)
                aar = list(Aktie.index)
                for i in range(len(aar)):
                    aar[i] = str(aar[i])[0:4]
        
                aar = list(dict.fromkeys(aar))
        
                probs = pd.DataFrame(0, index = ['min', 'max'], columns = ['7dage', '1måned'])
        
                for i in range(len(aar)-1):
        
                    temp_aktie = Aktie[Aktie.index < str(aar[i+1]) + "-01-30"]
                    temp_aktie = temp_aktie[temp_aktie.index >= str(aar[i]) + "-01-01"]
        
                    for j in range(len(temp_aktie)):
        
                        if str(temp_aktie.index[j]) < str(aar[i])+"12-29":
                            if temp_aktie.iloc[j] == min(temp_aktie) and temp_aktie.iloc[j + 6] > 1.01 * temp_aktie.iloc[j]:
                                probs.iloc[0,0] = probs.iloc[0,0] + 1.0
        
                            if temp_aktie.iloc[j] == min(temp_aktie) and temp_aktie.iloc[j + 19] > 1.05 * temp_aktie.iloc[j]:
                                probs.iloc[0,1] = probs.iloc[0,1] + 1.0
        
                            if temp_aktie.iloc[j] == max(temp_aktie) and temp_aktie.iloc[j + 6] < 0.99 * temp_aktie.iloc[j]:
                                probs.iloc[1,0] = probs.iloc[1,0] + 1.0
        
                            if temp_aktie.iloc[j] == max(temp_aktie) and temp_aktie.iloc[j + 19] < 0.95 * temp_aktie.iloc[j]:
                                probs.iloc[1,1] = probs.iloc[1,1] + 1.0
        
                probs = probs / len(aar)
        
                probabilities.iloc[0,0] = probabilities.iloc[0,0] + probs.iloc[0,0]
                probabilities.iloc[1,0] = probabilities.iloc[1,0] + probs.iloc[1,0]
                probabilities.iloc[0,1] = probabilities.iloc[0,1] + probs.iloc[0,1]
                probabilities.iloc[1,1] = probabilities.iloc[1,1] + probs.iloc[1,1]
        
            probabilities = probabilities / len(Tickers)         
            return(probabilities)
    
    
    
    
    
    def wealth_plot(x):
        titel = "Wealth plot " + str(x.columns[0])
        plt.figure(figsize=(18, 8))
        plt.plot((x + 1).cumprod())
        plt.grid()
        plt.title(titel)
        plt.xlabel("Date")
        plt.ylabel("Wealth")
        plt.show()

    def frequencies(data, bins, wp = False):
        #binning strategien skal opdateres på et senere tidspunkt
        if type(data) != pd.DataFrame:
            data = pd.DataFrame(data)
        
        m1 = min(data.iloc[:,0])
        m2 = max(data.iloc[:,0])    
        res1 = []
        res2 = []
        xlist = []
            
        for i in range(bins):
            k = 0

            inter1 = m1 + (i/bins) * (m2 - m1)
            inter2 = m1 + ((i+1) /bins) * (m2 -m1)
            xlist.append(round(inter2,5))

            for j in range(len(data)):
                if float(data.iloc[j,0]) > float(inter1) and float(data.iloc[j,0]) < float(inter2) :
                    k = k + 1

            res1.append(round(int(k),5))
            res2.append(round(k/len(data),5))
        
        
        res = pd.DataFrame([res1,res2],index = ['Frequencies', 'Percentages'], columns = xlist)
        lst = np.linspace(m1, m2, bins)

        if wp == True:
            y_pos = np.arange(len(res.columns))
            plt.bar(y_pos, res.iloc[0], color ='navy', width = 1)
            plt.xticks(y_pos,res.columns)
            plt.ylabel("Frequencies")
            plt.xlabel("Percentage returns")
            plt.title("Frequencies")
            plt.plot(y_pos, res.iloc[0], color = 'maroon')
            plt.xticks(y_pos,res.columns)
            plt.tick_params(axis='both', which='major', labelsize=7)
        return res

    def sstat(data, txt = "Stat"):
        m = pd.DataFrame([0])
        for n in range(0, 4):
            s = 0
            if n == 0:
                for j in range(len(data)):
                    if type(data) == pd.Series:
                        s = s + data[j]
                    elif type(data) == pd.DataFrame:
                        s = s + data.iloc[j,0]
                s = s/ len(data)

            elif n > 0:
                for j in range(len(data)):
                    if type(data) == pd.Series:
                        s = s + (data[j] - m[0][0])**(n+1)
                    elif type(data) == pd.DataFrame:
                        s = s + (data.iloc[j,0] - m[0][0])**(n+1)
                s = s / len(data)

            if n == 0 or n == 1:
                m[n] = s
            elif n>1:
                m[n] = s / (np.sqrt(m[1][0])**(n+1))
            
            if type(data) == pd.DataFrame:
                p = [data.describe().squeeze()[3],data.describe().squeeze()[4],data.describe().squeeze()[5],data.describe().squeeze()[6], data.describe().squeeze()[7]]
            else:
                p = [data.describe()[3],data.describe()[4],data.describe()[5],data.describe()[6],data.describe()[7]]

        mean = {'min':[p[0]],'25':[p[1]],'50':[p[2]],'75':[p[3]],'max':[p[4]],'Mean':[m[0][0]],'STD':[np.sqrt(m[1][0])], 'Variance':[m[1][0]], 'Skew':[m[2][0]], 'Kurtosis':[m[3][0]]}
        m = pd.DataFrame(mean, index = [str(txt)])
        return m

    def seasonality(S):
        lst = [0,0,0,0,0,0,0,0,0,0,0,0]
        liste = []
        j = 0
        k = 0
        f = 0
        for i in range(len(S)):

            if int(str(S.index[i])[0:4]) > int(str(S.index[i-1])[0:4]):
                liste.append(lst)
                j = 0
                k = 0
                lst = [0,0,0,0,0,0,0,0,0,0,0,0]

            if int(str(S.index[i])[5:7]) > int(str(S.index[i-1])[5:7]) :
                j = 0
                k = 0

            j = j + 1   
            k = k + S[i]
            f = k/j

            lst[int(str(S.index[i])[5:7])-1] = f

        SS = pd.DataFrame(liste, columns = ['JAN', 'FEB', 'MAR', 'APR', 'MAJ', 'JUN', 'JUL', 'AUG', 'SEP', 'OKT', 'NOV', 'DEC'])
        return SS

    def seasonmax(x):

        lstmin = [0,0,0,0,0,0,0,0,0,0,0,0]
        lstmax = [0,0,0,0,0,0,0,0,0,0,0,0]
        for aar in range(len(x)): 

            for j in range(len(x.columns)):
                maan = x.columns[j]
                if x[maan][aar] == 0:
                    break

                if x[maan][aar] == min(x.loc[aar]):
                    lstmin[j] = lstmin[j] + 1

                elif x[maan][aar] == max(x.loc[aar]):
                    lstmax[j] = lstmax[j] + 1

        LL = pd.DataFrame([lstmin,lstmax], columns = ['JAN', 'FEB', 'MAR', 'APR', 'MAJ', 'JUN', 'JUL', 'AUG', 'SEP', 'OKT', 'NOV', 'DEC'], index = ['Min', 'Max'])
        return LL




    ########################################################################################################
    # Empirical distribution function
    ########################################################################################################
    def ed(data, bins = 10):

        bine = np.linspace(min(data), max(data), bins)
        counts = []

        for i in range(0,len(bine)):

            bine[i] = round(bine[i],3)
            count = 0

            for j in range(len(data)):

                if data.iloc[j] >= bine[i-1] and data.iloc[j] < bine[i]:

                    count = count + 1

            counts.append(count/len(data))

        counts_df = pd.DataFrame(counts, index = bine)
        return(counts_df)



        
    def indicator(self, returns):
        self.indic = returns.copy()
        for i in range(len(self.indic)):
            if self.indic.iloc[i] < 0:
                self.indic.iloc[i] = -1
            elif self.indic.iloc[i] > 0:
                self.indic.iloc[i] = 1
            else:
                self.indic.iloc[i] = 0
        return(self.indic)  

    def n_d_indicator(self, indic, t):
        
        indic_d = indic.copy()
        
        for i in range(t, len(indic)):
            
            indic_d.iloc[i] = sum(indic[i-t:i])
            
        return(indic_d)
    
    
    
            
def updown(data):
    
    u_d_count = []
    
    for i in range(len(data)):
        
        if data.iloc[i] < 0:
            
            u_d_count.append(-1)
        
        elif data.iloc[i] > 0:
            
            u_d_count.append(1)
            
        else:
            
            u_d_count.append(0)
            
    return(u_d_count)
    
def updown_prob(data, k = 1):

    prob = pd.DataFrame(0,index = ['up', 'down'], columns = ['up', 'down'])

    # columns are "future"
    # rows are "past"

    for i in range(len(data)):

        if data[i] > 0 and data[i-k] > 0:

            prob['up']['up'] = prob['up']['up'] + 1

        elif data[i] > 0 and data[i-k] < 0:

            prob['up']['down'] = prob['up']['down'] + 1

        elif data[i] < 0 and data[i-k] < 0:

            prob['down']['down'] = prob['down']['down'] + 1

        elif data[i] < 0 and data[i-k] > 0:

            prob['down']['up'] = prob['down']['up'] + 1

    prob = prob / len(data)
    return(prob)      



#markov chain probabilities

def counts(data):
    ind = ['U','UU','UUU','UUD','UD','UDU','UDD','D','DU','DUU','DUD','DD','DDU','DDD'] #index
    
    counts = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    
    for i in range(1, len(data)):

        # if [-1] is up. Add +1 to counts[0]
        if data[i-1] > 0:   
            counts[0] = counts[0] + 1

        # UU
        if data[i-1] > 0 and data[i-2] > 0:
            counts[1] = counts[1] + 1

        # UUU
        if data[i-1] > 0 and data[i-2] > 0 and data[i-3] > 0:
            counts[2] = counts[2] + 1

        # UUD
        if data[i-1] > 0 and data[i-2] > 0 and data[i-3] < 0:
            counts[3] = counts[3] + 1

        # UD
        if data[i-1] > 0 and data[i-2] < 0:
            counts[4] = counts[4] + 1

        # UDU
        if data[i-1] > 0 and data[i-2] < 0 and data[i-3] < 0:
            counts[5] = counts[5] + 1

        # UDD
        if data[i-1] > 0 and data[i-2] < 0 and data[i-3] > 0:
            counts[6] = counts[6] + 1


        # if [-1] is down. Add +1 to counts[7]
        if data[i-1] < 0:   
            counts[7] = counts[7] + 1

        # DU
        if data[i-1] < 0 and data[i-2] > 0:
            counts[8] = counts[8] + 1

        # DUU
        if data[i-1] < 0 and data[i-2] > 0 and data[i-3] > 0:
            counts[9] = counts[9] + 1

        # DUD
        if data[i-1] < 0 and data[i-2] > 0 and data[i-3] < 0:
            counts[10] = counts[10] + 1

        # DD
        if data[i-1] < 0 and data[i-2] < 0:
            counts[11] = counts[11] + 1

        # DDU
        if data[i-1] < 0 and data[i-2] < 0 and data[i-3] < 0:
            counts[12] = counts[12] + 1

        # DDD
        if data[i-1] < 0 and data[i-2] < 0 and data[i-3] > 0:
            counts[13] = counts[13] + 1
        
    count = pd.DataFrame(counts, index = ind, columns = ['count']) # count table
    return(count)




def probs(data):
    ind = ['U','UU','UUU','UUD','UD','UDU','UDD','D','DU','DUU','DUD','DD','DDU','DDD'] #index
    
    up = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    down = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    
    for i in range(1, len(data)):
        
        if data[i] > 0:

            # if [-1] is up. Add +1 to counts[0]
            if data[i-1] > 0:   
                up[0] = up[0] + 1

            # UU
            if data[i-1] > 0 and data[i-2] > 0:
                up[1] = up[1] + 1

            # UUU
            if data[i-1] > 0 and data[i-2] > 0 and data[i-3] > 0:
                up[2] = up[2] + 1

            # UUD
            if data[i-1] > 0 and data[i-2] > 0 and data[i-3] < 0:
                up[3] = up[3] + 1

            # UD
            if data[i-1] > 0 and data[i-2] < 0:
                up[4] = up[4] + 1

            # UDU
            if data[i-1] > 0 and data[i-2] < 0 and data[i-3] < 0:
                up[5] = up[5] + 1

            # UDD
            if data[i-1] > 0 and data[i-2] < 0 and data[i-3] > 0:
                up[6] = up[6] + 1


            # if [-1] is down. Add +1 to counts[7]
            if data[i-1] < 0:   
                up[7] = up[7] + 1

            # DU
            if data[i-1] < 0 and data[i-2] > 0:
                up[8] = up[8] + 1

            # DUU
            if data[i-1] < 0 and data[i-2] > 0 and data[i-3] > 0:
                up[9] = up[9] + 1

            # DUD
            if data[i-1] < 0 and data[i-2] > 0 and data[i-3] < 0:
                up[10] = up[10] + 1

            # DD
            if data[i-1] < 0 and data[i-2] < 0:
                up[11] = up[11] + 1

            # DDU
            if data[i-1] < 0 and data[i-2] < 0 and data[i-3] < 0:
                up[12] = up[12] + 1

            # DDD
            if data[i-1] < 0 and data[i-2] < 0 and data[i-3] > 0:
                up[13] = up[13] + 1
        
        elif data[i] < 0:
            
                        # if [-1] is up. Add +1 to counts[0]
            if data[i-1] > 0:   
                down[0] = down[0] + 1

            # UU
            if data[i-1] > 0 and data[i-2] > 0:
                down[1] = down[1] + 1

            # UUU
            if data[i-1] > 0 and data[i-2] > 0 and data[i-3] > 0:
                down[2] = down[2] + 1

            # UUD
            if data[i-1] > 0 and data[i-2] > 0 and data[i-3] < 0:
                down[3] = down[3] + 1

            # UD
            if data[i-1] > 0 and data[i-2] < 0:
                down[4] = down[4] + 1

            # UDU
            if data[i-1] > 0 and data[i-2] < 0 and data[i-3] < 0:
                down[5] = down[5] + 1

            # UDD
            if data[i-1] > 0 and data[i-2] < 0 and data[i-3] > 0:
                down[6] = down[6] + 1


            # if [-1] is down. Add +1 to counts[7]
            if data[i-1] < 0:   
                down[7] = down[7] + 1

            # DU
            if data[i-1] < 0 and data[i-2] > 0:
                down[8] = down[8] + 1

            # DUU
            if data[i-1] < 0 and data[i-2] > 0 and data[i-3] > 0:
                down[9] = down[9] + 1

            # DUD
            if data[i-1] < 0 and data[i-2] > 0 and data[i-3] < 0:
                down[10] = down[10] + 1

            # DD
            if data[i-1] < 0 and data[i-2] < 0:
                down[11] = down[11] + 1

            # DDU
            if data[i-1] < 0 and data[i-2] < 0 and data[i-3] < 0:
                down[12] = down[12] + 1

            # DDD
            if data[i-1] < 0 and data[i-2] < 0 and data[i-3] > 0:
                down[13] = down[13] + 1
        
    count = pd.DataFrame(up, index = ind, columns = ['up']) # count table
    count['down'] = down
    return(count)
        
    







def moving_averages(data, t):
    
    if len(data) < t+1:
        raise ValueError("not enough data")
        
    mrt_df = []
    for i in range(len(data)):
        
        mean_reversion_times = np.zeros(t)
        
        for j in range(1,t):
        
            if i >= j:
            
                mean_reversion_times[j] = np.mean(data[i-j:i])
        
        mrt_df.append(mean_reversion_times)

    mrt_df = pd.DataFrame(mrt_df, index = data.index)
    mrt_df = mrt_df.iloc[1:len(mrt_df), 1:len(mrt_df.columns)]
    #lst[0] = list(x)
    return(mrt_df)

def mean_sizes(data, t):
    
    sizes = moving_averages(data, t)
    for i in range(len(sizes)):
        
        for j in range(len(sizes.columns)):
            
            sizes.iloc[i,j] = data.iloc[i] - sizes.iloc[i,j] 
            
    return(sizes)

def mean_size_dist(data, t):
    
    ms = mean_sizes(data,t)
    msd = pd.DataFrame(0, index = ['upper+2s', 'upper-m', 'upper-2s','lower+2s','lower-m','lower-2s'], columns = ms.columns)
    for ti in range(1, len(ms.columns)):
        
        msp = ms[ti+1][ms[ti+1] > 0]
        msn = ms[ti+1][ms[ti+1] < 0]
        
        msd[ti+1]['upper+2s'] = np.mean(msp) + 2 * np.std(msp)
        msd[ti+1]['upper-m'] = np.mean(msp)
        msd[ti+1]['upper-2s'] = min(msp)
        
        msd[ti+1]['lower+2s'] = max(msn)
        msd[ti+1]['lower-m'] = np.mean(msn)
        msd[ti+1]['lower-2s'] = np.mean(msn) - 2 * np.std(msn)
        
    return(msd)
        
def mean_times(data, t):
    ms = mean_sizes(data, t)
    mrt = pd.DataFrame(index = ["mrt"], columns = ms.columns)
    for i in range(len(ms.columns)):
        n = 0
        for j in range(len(ms)):
            if ms.iloc[j,i] > 0 and ms.iloc[j-1,i] < 0:
                n = n + 1
            elif ms.iloc[j,i] < 0 and ms.iloc[j-1,i] > 0:
                n = n + 1
            else:
                n = n
        mrt[i+1]['mrt'] = len(ms) / (n+1)
    return(mrt)      




def month_probabilities(Tickers):
    p = pd.DataFrame(0.0, index = ['p'], columns = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
    for j in range(1,13):
        russel_prob = 0.0

        for tick in Tickers:

            Aktie = tf.sdata(tick, '2000-01-01', today)
            aar = list(Aktie.index)
            for i in range(len(aar)):
                aar[i] = str(aar[i])[0:4]

            aar = list(dict.fromkeys(aar))

            r_prob = 0.0

            if len(str(j)) == 2 and (j-1) != 9:
                l_s = "-"+str(j) + "-01"
                u_s = "-"+str(j-1) + "-01"
            elif len(str(j)) == 2 and (j-1) == 9:
                l_s = "-"+str(j) + "-01"
                u_s = "-0"+str(j-1) + "-01"
            elif len(str(j)) != 2 and (j-1) != 0:
                l_s = "-0"+str(j)+"-01"
                u_s = "-0"+str(j-1)+"-01"
            elif len(str(j)) != 2 and (j-1) == 0:
                l_s = "-01-31"
                u_s = "-01-01"




            for i in range(1,len(aar)-1):

                june_july = Aktie[Aktie.index < str(aar[i]) + l_s]
                june_july = june_july[june_july.index >= str(aar[i]) + u_s]
                if june_july.iloc[0] < june_july.iloc[len(june_july)-1]:
                    r_prob = r_prob + 1.0

            russel_prob = russel_prob + r_prob / len(aar)

        p.iloc[0,j-1] = np.float64(russel_prob / len(Tickers))

    return(p)

def extremum_probabilities(Tickers):
    probabilities = pd.DataFrame(index = ['min', 'max'], columns = ['7dage', '1måned'])
    probabilities.iloc[0,0] = 0.0
    probabilities.iloc[1,0] = 0.0
    probabilities.iloc[0,1] = 0.0
    probabilities.iloc[1,1] = 0.0

    for tick in Tickers:

        Aktie = tf.sdata(tick, '2014-01-01', today)
        aar = list(Aktie.index)
        for i in range(len(aar)):
            aar[i] = str(aar[i])[0:4]

        aar = list(dict.fromkeys(aar))

        probs = pd.DataFrame(0, index = ['min', 'max'], columns = ['7dage', '1måned'])

        for i in range(len(aar)-1):

            temp_aktie = Aktie[Aktie.index < str(aar[i+1]) + "-01-30"]
            temp_aktie = temp_aktie[temp_aktie.index >= str(aar[i]) + "-01-01"]

            for j in range(len(temp_aktie)):

                if str(temp_aktie.index[j]) < str(aar[i])+"12-29":
                    if temp_aktie.iloc[j] == min(temp_aktie) and temp_aktie.iloc[j + 6] > 1.01 * temp_aktie.iloc[j]:
                        probs.iloc[0,0] = probs.iloc[0,0] + 1.0

                    if temp_aktie.iloc[j] == min(temp_aktie) and temp_aktie.iloc[j + 19] > 1.05 * temp_aktie.iloc[j]:
                        probs.iloc[0,1] = probs.iloc[0,1] + 1.0

                    if temp_aktie.iloc[j] == max(temp_aktie) and temp_aktie.iloc[j + 6] < 0.99 * temp_aktie.iloc[j]:
                        probs.iloc[1,0] = probs.iloc[1,0] + 1.0

                    if temp_aktie.iloc[j] == max(temp_aktie) and temp_aktie.iloc[j + 19] < 0.95 * temp_aktie.iloc[j]:
                        probs.iloc[1,1] = probs.iloc[1,1] + 1.0

        probs = probs / len(aar)

        probabilities.iloc[0,0] = probabilities.iloc[0,0] + probs.iloc[0,0]
        probabilities.iloc[1,0] = probabilities.iloc[1,0] + probs.iloc[1,0]
        probabilities.iloc[0,1] = probabilities.iloc[0,1] + probs.iloc[0,1]
        probabilities.iloc[1,1] = probabilities.iloc[1,1] + probs.iloc[1,1]

    probabilities = probabilities / len(Tickers)         
    return(probabilities)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            

