''' StockDataGrab.py
Class object to be run from terminal that performs the following operations:
1. Obtain new data from IB
2. Encode data to visual format
3. Train new models (although this is typically done on GPU outside of this machine)
4. Predict on specified models
5. Grab recent data to make forecasts
6. Identify most likely opportunities in options market 
7. Perform trades based on forecasts'''

# Packages
import warnings
import os
# import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from arch import arch_model  # !pip install arch==5.3.1
import ta as ta  # !pip install git+https://github.com/bukosabino/ta.git
import datetime as dt
from sklearn.model_selection import train_test_split
from ib_insync import *
import pandas as pd
# from keras import layers
from datetime import datetime, timedelta
import pickle

class Stock42():
    def __init__(self):
        # self.modName = '600x68a7sh-2.45S10.42L'
        self.tix = ['AAPL','GOOG','TSLA','NFLX','DHR',
            'MMM','PFE','AMZN','AMD','ABEV',
            'NVDA','BAC','JPM','INTC','XOM',
            'MSFT','C','CRM','BA','UNH',
            'WMT','CVS','JNJ','V','LLY',  # This is the end of the original training set
            'BRK B','AVGO','PG','MA','MRK',
            'HD','CVX','PEP','MCD','CSCO',
            'COST','TMO','ADBE','DIS','WFC',
            'KR','MCK','T','CIG','CAH',
            'ELV','MRO','WBA','VZ','PSX',
            'UPS','DELL','LOW','ADM',
            'GE','IBM','MET','PRU','RTX',
            'HUM','COR','VLO','CNC','TJX',
            'CMCSA','META','TGT','ET','FDX',
            'AA','COP','ACI','GS',
            'SYY','LMT','SNEX','MS','HP',
            'CAT','HCA','DOW','ABBV','AIG',
            'SNX','AXP','DAL','TSN','DE',
            'PGR','NKE','BBY','PBF', # 'PFGC', PFGC doesn't work
            'K','BMY','QCOM','ABT','UAL'
            ]
        self.featLength = 95
        self.p = 10
        self.trainDuration = '20 Y'
        # self.dur = 10
        self.InputLength = 252
        self.bins = 10 # Also a default for histogram
        self.extraBins = 10
        self.lookback = [252] # [22, 65, 130, 22*9, 252] # M, Q, H, 3/4Y, Y
        # self.ul = 31.437988662719725
        # self.ll = -9.06309299468994
        self.maxPositions=1 # Should be set as a global parameter
        self.stpLsRatio = 0.7
        self.profTakrRatio = 1.5
        self.binDef = [1,2,5,10,15,20,30,65,130,252]

    def connectIB(self):
        util.startLoop()  # uncomment this line when in a notebook
        self.ib = IB()
        self.ib.disconnect()
        self.ib.connect('127.0.0.1', 7496, clientId=11)
        warnings.filterwarnings('ignore', category=RuntimeWarning, module='ib_insync')

    def disconnectIB(self):
        util.startLoop()  # uncomment this line when in a notebook
        self.ib.disconnect()

    def features(self, VIX, SPY, TNX):
        warnings.filterwarnings('ignore')
        periods=self.p
        InputLength = self.InputLength
        AllData = pd.DataFrame(self.AllData)
        AllData[['VIX']] = VIX[['close']]
        AllData[['SPY']] = SPY[['close']]
        AllData[['TNX']] = TNX[['close']]
        AllData = AllData.reset_index(drop=True)
        AllData = AllData.rename(columns={"volume": "Volume"})

        rsi14 = ta.momentum.RSIIndicator(close=AllData['close'], window=14, fillna=True)
        AllData['rsi14'] = rsi14.rsi()
        rsi9 = ta.momentum.RSIIndicator(close=AllData['close'], window=9, fillna=True)
        AllData['rsi9'] = rsi9.rsi()
        rsi24 = ta.momentum.RSIIndicator(close=AllData['close'], window=24, fillna=True)
        AllData['rsi24'] = rsi24.rsi()
        # # MACD 5, 35, 5
        macd5355 = ta.trend.MACD(close=AllData['close'], window_slow=35, window_fast=5, window_sign=5, fillna=True)
        AllData['MACD5355macddiff'] = macd5355.macd_diff()
        AllData['MACD5355macddiffslope'] = macd5355.macd_diff().diff() # Slope of line
        AllData['MACD5355macd'] = macd5355.macd()
        AllData['MACD5355macdslope'] = macd5355.macd().diff()
        AllData['MACD5355macdsig'] = macd5355.macd_signal()
        AllData['MACD5355macdsigslope'] = macd5355.macd_signal().diff()

        # MACD 12,26,9
        macd12269 = ta.trend.MACD(close=AllData['close'], window_slow=26, window_fast=12, window_sign=9, fillna=True)
        AllData['MACD12269macddiff'] = macd12269.macd_diff()
        AllData['MACD12269macddiffslope'] = macd12269.macd_diff().diff() # Slope of line
        AllData['MACD12269macd'] = macd12269.macd()
        AllData['MACD12269macdslope'] = macd12269.macd().diff()
        AllData['MACD12269macdsig'] = macd12269.macd_signal()
        AllData['MACD12269macdsigslope'] = macd12269.macd_signal().diff()

        # Engulfing Bars
        AllData['lowTail'] = AllData['low'].pct_change()
        AllData['highTail'] = AllData['high'].pct_change()
        AllData['openTail'] = AllData['open'].pct_change()
        AllData['IntradayBar'] = (AllData.close.values[:] - AllData.open.values[:]) / AllData.open.values[:]
        AllData['IntradayRange'] = (AllData.high.values[:] - AllData.low.values[:]) / AllData.low.values[:]
        
        # SMA divergence
        for s in [5,10, 12 ,20, 30,65 ,50,100, 200]:
            AllData['CloseOverSMA'+str(s)] = AllData['close'] / AllData['close'].rolling(window=s).mean()

        # SMA Volume divergence
        for s in [5,10, 12 ,20, 30,65 ,50,100, 200]:
            AllData['VolOverSMA'+str(s)] = AllData['Volume'] / AllData['Volume'].rolling(window=s).mean()

        # Recent Performance
        AllData['Ret1day'] = AllData['close'].pct_change(1)  # 1
        AllData['Ret4day'] = AllData['close'].pct_change(5)  # 4
        AllData['Ret8day'] = AllData['close'].pct_change(10)  # 8
        AllData['Ret12day'] = AllData['close'].pct_change(15)# 12
        AllData['Ret24day'] = AllData['close'].pct_change(20)#24
        AllData['Ret72day'] = AllData['close'].pct_change(65)#72
        AllData['Ret240day'] = AllData['close'].pct_change(252)#240

        # Relative Strength Comparison w/ SPY

        AllData['RSC'] = AllData['Ret1day']-AllData['SPY'].pct_change(1)

        # BBs
        BB = ta.volatility.BollingerBands(close=AllData['close'], fillna=True, window=22)
        AllData['bands_l'] = BB.bollinger_lband()
        AllData['bands_u'] = BB.bollinger_hband()

        # ADX
        ADX = ta.trend.ADXIndicator(high=AllData['high'], low=AllData['low'], close=AllData['close'], window=14, fillna=True)
        AllData['ADX'] = ADX.adx()

        # # Ichimoku
        ICH = ta.trend.IchimokuIndicator(high=AllData['high'], low=AllData['low'], fillna=True)
        AllData['cloudA'] = ICH.ichimoku_a()
        AllData['cloudB'] = ICH.ichimoku_b()
        AllData['closeVsIchA'] = AllData['close'] / ICH.ichimoku_a()
        AllData['closeVsIchB'] = AllData['close'] / ICH.ichimoku_b()
        AllData['IchAvIchB'] = ICH.ichimoku_a() / ICH.ichimoku_b()

        # # Garch volatility forecast

        AllData = AllData.dropna(axis=0,how='any')

        am = arch_model(100*AllData['Ret1day'])
        res = am.fit(update_freq=1, disp="off", show_warning=False)
        AllData['CondVol_1'] = res.conditional_volatility.values.reshape(-1,1)

        am = arch_model(100*AllData['Ret4day']) # 4
        res = am.fit(update_freq=1, disp="off", show_warning=False)
        AllData['CondVol_4'] = res.conditional_volatility.values.reshape(-1,1)

        am = arch_model(100*AllData['Ret8day']) # 8
        res = am.fit(update_freq=1, disp="off", show_warning=False)
        AllData['CondVol_8'] = res.conditional_volatility.values.reshape(-1,1)

        am = arch_model(100*AllData['Ret12day']) # 12
        res = am.fit(update_freq=1, disp="off", show_warning=False)
        AllData['CondVol_12'] = res.conditional_volatility.values.reshape(-1,1)

        am = arch_model(100*AllData['Ret24day']) # 24
        res = am.fit(update_freq=1, disp="off", show_warning=False)
        AllData['CondVol_24'] = res.conditional_volatility.values.reshape(-1,1)

        am = arch_model(100*AllData['Ret72day']) # 72
        res = am.fit(update_freq=1, disp="off", show_warning=False)
        AllData['CondVol_72'] = res.conditional_volatility.values.reshape(-1,1)
        # erase below
        # Step 1: Replace inf and -inf with NaN
        AllData['Ret240day'].replace([np.inf, -np.inf], np.nan, inplace=True)
        # Step 2: Interpolate the NaN values (previously inf values)
        AllData['Ret240day'].interpolate(method='nearest', inplace=True)
        # Step 3: If any NaN values remain at the start/end, fill them with forward/backward fill
        AllData['Ret240day'].fillna(method='ffill', inplace=True)  # Forward fill
        AllData['Ret240day'].fillna(method='bfill', inplace=True)  # Backward fill
        # end erase   
        am = arch_model(100*AllData['Ret240day']) # 240
        res = am.fit(update_freq=1, disp="off", show_warning=False)
        AllData['CondVol_240'] = res.conditional_volatility.values.reshape(-1,1)

        AllData['CV1vCV4'] = AllData['CondVol_1']-AllData['CondVol_4']
        AllData['CV4vCV8'] = AllData['CondVol_4']-AllData['CondVol_8']
        AllData['CV8vCV12'] = AllData['CondVol_8']-AllData['CondVol_12']
        AllData['CV12vCV24'] = AllData['CondVol_12']-AllData['CondVol_24']
        AllData['CV8vCV24'] = AllData['CondVol_8']-AllData['CondVol_24']
        AllData['CV24vCV240'] = AllData['CondVol_24'] - AllData['CondVol_240']

        AllData['RSC_VIX'] = AllData['CondVol_1']-AllData['VIX'].pct_change(1)
        AllData['RSC_VIX_IV'] = AllData['vclose']-AllData['VIX'].pct_change(1)
        AllData['RSC_VIX_real'] = AllData['CondVol_1']-AllData['VIX']
        AllData['RSC_VIX_IV_real'] = AllData['vclose']-AllData['VIX']
        AllData['RSC_IV_gar'] = AllData['vclose']-AllData['CondVol_1']

        AllData['close_spy_corr22'] = AllData['close'].rolling(22).corr(AllData['SPY'])
        AllData['close_tnx_corr22'] = AllData['close'].rolling(22).corr(AllData['TNX'])
        AllData['vclose_VIX_corr22'] = AllData['vclose'].rolling(22).corr(AllData['VIX'])
        AllData['garch_IV_corr22'] = AllData['CondVol_1'].rolling(22).corr(AllData['vclose'])

        AllData['close_spy_corr65'] = AllData['close'].rolling(65).corr(AllData['SPY'])
        AllData['close_tnx_corr65'] = AllData['close'].rolling(65).corr(AllData['TNX'])
        AllData['vclose_VIX_corr65'] = AllData['vclose'].rolling(65).corr(AllData['VIX'])
        AllData['garch_IV_corr65'] = AllData['CondVol_1'].rolling(65).corr(AllData['vclose'])

        AllData['close_spy_corr252'] = AllData['close'].rolling(252).corr(AllData['SPY'])
        AllData['close_tnx_corr252'] = AllData['close'].rolling(252).corr(AllData['TNX'])
        AllData['vclose_VIX_corr252'] = AllData['vclose'].rolling(252).corr(AllData['VIX'])
        AllData['garch_IV_corr252'] = AllData['CondVol_1'].rolling(252).corr(AllData['vclose'])

        # Clean up output
        AllData = AllData.dropna(axis=0, how='any')
        AllData = AllData.drop(columns=['Volume'], axis=1)
        AllData = AllData.rename(columns={"close": "Close"}) # Parameter assignment for RL target

        tar = np.nan * (np.zeros(len(AllData.Close)))
        tar[:-periods] = AllData.Close.values[periods:] / AllData.open.values[1:len(AllData.Close) - periods + 1] - 1
        AllData['WasUp'] = tar
        self.AllData = AllData
        return

    def getDat(self,symbol):
        contract = Stock(symbol, 'SMART', 'USD')
        self.ib.qualifyContracts(contract)
        # Hist OLHCV data
        self.historical_data = self.ib.reqHistoricalData(
            contract, 
            endDateTime='',
            barSizeSetting='1 day', 
            durationStr=self.trainDuration, 
            whatToShow='ADJUSTED_LAST',
            useRTH=True,
            )
        AllData = util.df(self.historical_data)
        AllData=AllData.set_index(AllData['date'],drop=True)
        # Volatility
        IV_historical_data = self.ib.reqHistoricalData(
            contract, 
            endDateTime='',
            barSizeSetting='1 day', 
            durationStr=self.trainDuration, 
            whatToShow='OPTION_IMPLIED_VOLATILITY',
            useRTH=True,
            )
        AD_IV = util.df(IV_historical_data)
        AD_IV=AD_IV.set_index(AD_IV['date'], drop=True)
        AD_IV=AD_IV.drop(columns=['barCount','volume'])
        AD_IV=AD_IV.rename(columns={'open':'vopen','high':'vhigh','low':'vlow','close':'vclose','average':'vaverage'})
        AllData[['vclose','vopen','vhigh','vlow','vaverage']] = AD_IV[['vclose','vopen','vhigh','vlow','vaverage']]
        AllData=AllData.drop(columns=['date','barCount','vaverage','average']) # Drop average and vaverage because these are not accessible in live data
        self.AllData = AllData
        return
    
    def GetLastTick(self,ticker):
        # contract = Stock('AAPL', 'SMART', 'USD')
        contract = Stock(ticker, 'SMART', 'USD')
        self.ib.qualifyContracts(contract)
        self.temp = self.ib.reqMktData(contract)
        # print('O', self.temp.open,
        #         'H', self.temp.high, 
        #         'L', self.temp.low,
        #         'Last', self.temp.last)

    def GetCurrentData(self):
        # Index & VIX data
        contract = Index('SPX', 'CBOE', 'USD')
        self.ib.qualifyContracts(contract)
        historical_data = self.ib.reqHistoricalData(
            contract, 
            endDateTime='',
            barSizeSetting='1 day', 
            durationStr=self.trainDuration, 
            whatToShow='ADJUSTED_LAST',
            useRTH=True,
            )
        SPY = util.df(historical_data)
        SPY=SPY.set_index(SPY['date'], drop=True)

        contract = Index('VIX', 'CBOE', 'USD')
        self.ib.qualifyContracts(contract)
        historical_data = self.ib.reqHistoricalData(
            contract, 
            endDateTime='',
            barSizeSetting='1 day', 
            durationStr=self.trainDuration, 
            whatToShow='ADJUSTED_LAST',
            useRTH=True,
            )
        VIX = util.df(historical_data)
        VIX=VIX.set_index(VIX['date'], drop=True)
        # New treasury linked features
        contract = Index('TNX', 'CBOE', 'USD')
        self.ib.qualifyContracts(contract)
        historical_data = self.ib.reqHistoricalData(
            contract, 
            endDateTime='',
            barSizeSetting='1 day', 
            durationStr=self.trainDuration, 
            whatToShow='ADJUSTED_LAST',
            useRTH=True,
            )
        TNX = util.df(historical_data)
        TNX=TNX.set_index(TNX['date'], drop=True)

        print('Getting Data for equities...')
        self.MuSigTix = pd.DataFrame(columns=['ticker','closemu','closesig','volmu','volsig'])

        for t in self.tix:
            # print(t)
            self.getDat(t)
            self.features(VIX, SPY, TNX)
            self.AllData = self.AllData.drop(columns=['WasUp'])
            if len(self.AllData) > 1000: # Some equities are too short in length for a full train/test/oss sample
                for f in range(len(self.AllData.columns)): # features
                    # should have two classes. 1 for all non-target features that is normal scaled
                    # the other that has the target variables that are not scaled or min/max scaled.
                    feat = self.AllData.columns[f]
                    rollmu = self.AllData[feat].rolling(252).mean()
                    rollstd = self.AllData[feat].rolling(252).std()
                    self.AllData[feat]=(self.AllData[feat]-rollmu)/rollstd
                    if feat == 'Close':
                        closemu = rollmu
                        closestd = rollstd
                    if feat == 'vclose':
                        volmu = rollmu
                        volstd = rollstd

                new_row = {'ticker': t, 'closemu': closemu.iloc[-1], 'closesig': closestd.iloc[-1],
                        'volmu': volmu.iloc[-1], 'volsig': volstd.iloc[-1]}
                # self.MuSigTix = self.MuSigTix.append(new_row, ignore_index=True)

                new_row_df = pd.DataFrame([new_row])  # Convert new_row to a DataFrame
                self.MuSigTix = pd.concat([self.MuSigTix, new_row_df], ignore_index=True)

                    # if feat not in ["Close", "vclose"]:
                    #     self.AllData[feat]=(self.AllData[feat]-self.AllData[feat].rolling(252).mean())/self.AllData[feat].rolling(252).std()
                    # else: # min max scaling
                    #     self.AllData[feat]=(self.AllData[feat]-self.AllData[feat].rolling(252).min())/(self.AllData[feat].rolling(252).max()-self.AllData[feat].rolling(252).min())

                self.AllData=self.AllData.dropna(axis=0)
                self.AllData=self.AllData.reset_index(drop=True)
                # # Original
                oos_len = 252  #### Should be user defined (oos, tr, etc...)
                tr_len = int(.80*len(self.AllData[:-oos_len]))
                self.AllData[:tr_len].to_csv('./spacetimeformer/data/train/' + t + '.csv')
                self.AllData[tr_len:-oos_len].to_csv('./spacetimeformer/data/test/' + t + '.csv')
                self.AllData[-oos_len:].to_csv('./spacetimeformer/data/oos/' + t + '.csv')
                
                # shuffled train/test - This will not work for Transformer since the inputs need to be sequential (not scattered)
                # oos_len = 252
                # oos_data = self.AllData[-oos_len:]
                # train_test_data = self.AllData[:-oos_len]
                # # Shuffle and split the train_test_data into 75% train and 25% test, maintaining the shuffle
                # train_data, test_data = train_test_split(train_test_data, test_size=0.25, shuffle=True)
                # # Export to CSV
                # print(t,len(train_data),len(test_data),len(oos_data))
                # train_data.to_csv(f'./data/train/{t}.csv', index=False)
                # test_data.to_csv(f'./data/test/{t}.csv', index=False)
                # oos_data.to_csv(f'./data/oos/{t}.csv', index=False)

        self.MuSigTix.to_csv('./spacetimeformer/data/TixMuSig.csv')
        return
    
    def getOptionPositions(self):
        # self.connectIB()
        self.things = {}
        for p in range(len(self.curPreds)):
            sym = self.curPreds[p][0]
            contract = Stock(sym, 'SMART','USD')
            self.ib.qualifyContracts(contract)

            self.ib.reqMarketDataType(4)
            [ticker] = self.ib.reqTickers(contract)
            print(ticker)

            chains = self.ib.reqSecDefOptParams(contract.symbol, '', contract.secType, contract.conId)
            chain = next(c for c in chains if c.tradingClass == sym and c.exchange == 'SMART')

            strikes = [strike for strike in chain.strikes
                    if strike % 1 == 0
                    and ticker.last*.7 < strike < ticker.last*1.3]
            # strikes = [strike for strike in chain.strikes
            #         if ticker.last*.9 < strike < ticker.last*1.1]

            expirations = sorted(exp for exp in chain.expirations)[3:4] # The next 4 wkly option expirations 
            # NEED TO INDICATE WHETHER C OR P BASED ON THE PREDICTED VALUE
            # rights = ['P', 'C']
            if self.curPreds[p][1] > 0:
                rights = ['C']
            else:
                rights = ['P']

            contracts = [Option(sym, expiration, strike, right, 'SMART', tradingClass=sym)
                    for right in rights
                    for expiration in expirations
                    for strike in strikes]

            contracts = self.ib.qualifyContracts(*contracts)

            # datetime.today().strftime('%Y%m%d')

            def days_between(exp):
                d1 = datetime.strptime(datetime.today().strftime('%Y%m%d'), "%Y%m%d")
                d2 = datetime.strptime(exp, "%Y%m%d")
                return abs((d2 - d1).days)

            dur = []
            for i in range(len(expirations)):
                dur.append(days_between(expirations[i]))

            try: # sometimes this logic breaks when the strikes are sparse or nonexistent
                strikes[min(range(len(strikes)), key = lambda i: abs(strikes[i]-ticker.last))]
                strikes = np.array(strikes)
                strike_loc = np.where(strikes == strikes[min(range(len(strikes)), key = lambda i: abs(strikes[i]-ticker.last))]) # Finds closest strike

                dur = np.array(dur)
                exp_loc = np.where(dur == dur[min(range(len(dur)), key = lambda i: abs(dur[i]-22))])
                exp_loc = exp_loc[0][0]
                strike_loc = strike_loc[[0][0]]

                # print('strike: ',strikes[strike_loc][0])
                # print('expiration: ',expirations[exp_loc])
                # need to swap out the 'C' below in favor of rights
                # contract = Option(sym, str(expirations[exp_loc]), strikes[strike_loc][0], 'C', 'SMART')
                contract = Option(sym, str(expirations[exp_loc]), strikes[strike_loc][0], rights[0], 'SMART')
                contract = self.ib.qualifyContracts(contract)[0]  # To fill in other contract details
                details = self.ib.reqTickers(contract,)

                # print(details)
                # print ('Bid: ',details[0].bid)
                # print ('Ask: ',details[0].ask)
                # print ('Delta: ',details[0].lastGreeks.delta)
                # print ('Gamma: ',details[0].lastGreeks.gamma)
                # print ('IV: ',details[0].lastGreeks.impliedVol)
                # print ('Theta: ',details[0].lastGreeks.theta)

                newThings = {'Contract':contract,
                            'Trigger':self.curPreds[p][1],
                            'Bid':details[0].bid,
                            'Ask':details[0].ask,
                            'Delta':details[0].lastGreeks.delta,
                            'Gamma':details[0].lastGreeks.gamma,
                            'IV':details[0].lastGreeks.impliedVol,
                            'Theta':details[0].lastGreeks.theta
                            }
                self.things[sym] = newThings
            except:
                1+1
        d=str(datetime.strptime(datetime.today().strftime('%Y%m%d'), "%Y%m%d"))
        # save dictionary to person_data.pkl file
        with open('./optionData/'+d[:10]+ '_data.pkl', 'wb') as fp:
            pickle.dump(self.things, fp)
            print('dictionary saved successfully to file')
        # self.disconnectIB()
        return
    
    def FindOptionPositions(self):
        # self.connectIB()
        self.trainDuration = '5 Y'
        self.GetCurrentData()
        self.predModel()
        self.getOptionPositions()

    def newOrders(self):
        self.trainDuration = '5 Y'
        self.GetCurrentData()
        self.predModel()
        self.getOptionPositions()
        self.findBestPositions()
        self.placeBracketOrders()
        return

    def findBestPositions(self):
        ''' Returns the top positions for the current predictions & data
        This will need to be re-scoped to Stockerformer predictions'''

        temp = []
        for i in range(len(self.curPreds)):
            temp.append(abs(self.curPreds[i][1]))
        sort_index = np.argsort(temp)
        self.topPicks=[]
        for pos in range(self.maxPositions):
            self.topPicks.append(self.curPreds[sort_index[-(pos+1)]][0])
        return
    
    def placeBracketOrders(self):
        for o in self.topPicks:
            try:
                eq = self.things[o]
                self.ib.qualifyContracts(eq['Contract'])
                qty=1
                price=eq['Ask']
                ProfTaker = float(np.round(self.profTakrRatio*price,1))
                StpLs = float(np.round(self.stpLsRatio*price,1))
                ticker_bracket_order = self.ib.bracketOrder('BUY',qty,price,ProfTaker,StpLs)
                #place bracket order
                self.getClosingDate()
                for ord in ticker_bracket_order:
                    self.ib.sleep(1)
                    ord.tif ='GTD'
                    ord.goodTillDate = self.GTD
                    self.ib.placeOrder(eq['Contract'], ord)
            except:
                1+1
        return
    
    def getClosingDate(self):
    # Get the current date and time
        current_date = datetime.now()
        # Define a function to add business days
        def add_business_days(start_date, num_days):
            current_date = start_date
            while num_days > 0:
                current_date += timedelta(days=1)
                if current_date.weekday() < 5:  # Monday (0) to Friday (4) are business days
                    num_days -= 1
            return current_date

        # Calculate the target date with 10 business days added to the current date
        target_date = add_business_days(current_date, 11) # 11 days since the trained network is working with 10 days from tomorrow morning!

        # Format the current date and target date as strings in the desired format
        target_date_string = target_date.strftime("%Y%m%d %H:%M:%S")

        self.GTD = target_date_string
        return
    
# Run it
if __name__ == "__main__":
    stock = Stock42()
    # stock.CreateModel() # Model Scaffold
    # stock.LoadModelFromFile()  # Load Weights
    stock.connectIB()  # Connect to Interactive Brokers or your data source
    stock.trainDuration = '20 Y'
    stock.GetCurrentData()
    # Get Historic Data
    # stock.GetCurrentData()
    # Get Indiv equity
    # stock.getDat('AAPL')  # Replace with the appropriate ticker symbol