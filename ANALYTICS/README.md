# Base data, Open, Low, High, Close, Volume, date points
DATA = data('ISS.CO','2024-01-01','2024-09-18')

#Stochastic oscillator of DATA
DATA_S = STO(DATA, N = 20, M = 3)

#ADX of DATA
DATA_AD = ADX(DATA['High'], DATA['Low'], DATA['Close'], 20)

#MACD of DATA
DATA_MACD = MACD(DATA['Close'], 26, 12, 9)

#SuperTrend of DATA
DATA_SUP = SuperTrend(DATA['High'], DATA['Low'], DATA['Close'], 20, 2)

#price curve
PRICE_plot(DATA)
# stochastic oscillator plot
STO_plot(DATA_S)
# Average directional index plot
ADX_plot(DATA_AD)

MACD_plot(DATA.Close, DATA_MACD['macd'], DATA_MACD['signal'], DATA_MACD['hist'])

#TRADE_plot(DATA, buy_price, sell_price)

#SUM_plot(DATA, strategy)
