from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf


def ADF_Test(timeseries):
    """
    ADF_Test 
    print out parameters from the Dickey-Fuller test
    """
    #     print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    pvalue = dfoutput['p-value'] 
#     print(pvalue)
#     print(dfoutput)
    return pvalue

def Compare_Transformations(data, tdata):
    """ 
    Compare the data vs a trsnsformation of the  data
    drop null values, run an ADF test and return the p-value
    """
    tdiff = data - tdata
    tdiff.dropna(inplace=True)
    pvalue = ADF_Test(tdiff)

#     data.plot()
#     tdata.plot()
    return pvalue

def Plot_Compare(data, tdata):
    """
    Plot the difference between data and tdata
    To see if difference is stationary
    """
    y_diff = data - tdata
    y_diff.dropna(inplace=True)
    pval = Compare_Transformations(data,tdata)
    
    fig = plt.figure(figsize=(12,6))
    y_diff.plot()
    title = ' p-value: {0:1.2}'.format(pval)
    plt.title(title,fontsize=18)
    plt.xlabel('Date', fontsize=16)
    plt.ylabel(r'Delta data', fontsize=16)
    plt.grid(True)

def Get_Best_Model(data, p_params, q_params):
    """
    Use a SARIMAX fit to get the best model for the data
    
    Return
    Return the model with the lowest AIC
    """
    lowest_AIC =1e14
    best_param =0
    best_seasonal_param = 0
    p = p_params
    d = range(0,2)
    q = q_params

    pdq = list(itertools.product(p,d,q))
    seasonal_pdq = [(x[0], x[1],x[2],12) for x in list(itertools.product(p, d, q))]
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                
                mod = sm.tsa.statespace.SARIMAX(data,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
                results = mod.fit(maxiter=200)

                if results.aic < lowest_AIC:
                    lowest_AIC = results.aic
                    best_model = mod
                    best_param = param
                    best_seasonal_param = param_seasonal

            except:
                continue
            
    return best_model, best_param, best_seasonal_param

def Get_p_params(data):
    """
    Get the p Parameters from an ACF 
    """
    acfarray, confintarray = acf(data, alpha=.05)
    obs = len(data)
    p_params = []
    for i,v in enumerate(acfarray):
        if (v <= (-1.96/np.sqrt(obs)) or v >= (1.96/np.sqrt(obs))) and i > 0:
            p_params.append(i)
    return p_params[:4]

def Get_q_params(data):
    """
    Get the q Parameters from an PACF 
    """
    pcfarray, confintarray = pacf(data, alpha=.05)
    obs = len(data)
    q_params = []
    for i,v in enumerate(pcfarray):
        if (v <= (-1.96/np.sqrt(obs)) or v >= (1.96/np.sqrt(obs))) and i > 0:
            q_params.append(i)
    return q_params[:4]

def Trend_elimination(data):
    """
    Apply different transformation on the data, calculate the p-value 
    from a ADF test, and return the data set with the lowest p-value
    """
    data_transformed = []
    pvalues = []
    name = []
    
    # log data
#     print('Log data')
    name.append('Log_data')
    data_log = np.log(data)
    pvalue = Compare_Transformations(data, data_log)
    data_transformed.append(data_log)
    pvalues.append(pvalue)

#     print('diff 1')
    # diff
    name.append('diff_data')
    data_diff = data.shift(12)
    pvalue = Compare_Transformations(data, data_diff)
    data_transformed.append(data_diff)
    pvalues.append(pvalue)
    
#     print('rolling mean on data')
    # rolling mean
    name.append('rolling_mean_data')
    data_roll_mean = data.rolling(window=3).mean()
    pvalue = Compare_Transformations(data, data_roll_mean)
    data_transformed.append(data_roll_mean)
    pvalues.append(pvalue)
    
    # EWM
#     print('ewm on data')
    name.append('ewm_data')
    data_ewm = data.ewm(halflife= 2).mean()
    pvalue = Compare_Transformations(data, data_ewm)
    data_transformed.append(data_ewm)
    pvalues.append(pvalue)
    
    #compare all p-values and return the lowest one
    pvalue_min = np.min(pvalues)
#     print(pvalue_min)
    lowesti = pvalues.index(pvalue_min)
    return name[lowesti], pvalue_min, data_transformed[lowesti]

def Make_PredictionPlot(results, ytrain, ytest, zipcode, time_cut, print_test=True):
    fig = plt.figure(figsize=(14,7))
    pred = results.get_prediction(start=pd.to_datetime(time_cut)+pd.DateOffset(months=1),
                              end=pd.to_datetime('2019-04'),
                             dynamic=True)
    pred_ci = pred.conf_int() 
    
    ax = ytrain.plot(label='Train Data',lw=4)
    if print_test:
        ytest.plot(ax=ax, label='Test Data',lw=4, c='b')
    
    pred.predicted_mean.plot(ax = ax, label='Prediction',lw=4,
                             color='orange',linestyle='--')
    ax.fill_between(pred_ci.index,
               pred_ci.iloc[:,0],
               pred_ci.iloc[:,1], color='k', alpha=0.25)

#     pred_future = results.get_forecast(steps=16)
#     pred_ci2 = pred_future.conf_int()

#     pred_future.predicted_mean.plot(ax=ax, label=' Future Forecast')
#     ax.fill_between(pred_ci2.index, 
#                pred_ci2.iloc[:,0],
#                pred_ci2.iloc[:,1], color='k',alpha = 0.25)

    plt.title('Zipcode {}'.format(zipcode), fontsize=22)
    plt.xlabel('Date',fontsize=20)
    plt.ylabel('Price', fontsize=20)
    plt.legend(fontsize=18)
    plt.grid(True)