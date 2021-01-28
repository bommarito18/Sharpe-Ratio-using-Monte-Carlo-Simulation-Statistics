#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
from pandas_datareader import data as pdr
import scipy.stats as scs
import statsmodels.api as sm
from pylab import mpl, plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


yf.pdr_override()


# In[3]:


plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'


# In[4]:


def gen_paths(S0, r, sigma, T, M, I):
    '''Generate Monte Carlo paths for geometric Brownian motion.
    
    Parameters
    ==========
    S0: float
        intital stock/index value
    r: float
        constant short rate
    sigma: float
        constant volatility
    T: float
        final time horizon
    M: int
        number of time steps/intervals
    I: int
        number of paths to be simulated
       
    Returns
    ==========
    paths: ndarray, shape (M + 1, I)
        simulated paths given the parameters
    '''
    dt = T / M
    paths = np.zeros((M + 1, I))
    paths[0] = S0
    for t in range(1, M + 1):
        rand = np.random.standard_normal(I)
        rand = (rand - rand.mean()) / rand.std()
        paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * rand)
    return paths


# In[5]:


S0 = 100
r = 0.05
sigma = 0.2
T = 1.0
M = 50
I = 250000
np.random.seed(1000)


# In[6]:


paths = gen_paths(S0, r, sigma, T, M, I)


# In[7]:


S0 * math.exp(r * T)


# In[8]:


paths[-1].mean()


# In[9]:


plt.figure(figsize=(10,6))
plt.plot(paths[:, :10])
plt.xlabel('time steps')
plt.ylabel('index level');


# In[10]:


paths[:, 0].round(4)


# In[11]:


log_returns = np.log(paths[1:] / paths[:-1])


# In[12]:


log_returns[:, 0].round(4)


# In[13]:


def print_statistics(array):
    '''Prints selected statistics.
    
    Parameters
    ==========
    array: ndarray
        object to generate statistics on
    '''
    sta = scs.describe(array)
    print('%14s %15s' % ('statistic', 'value'))
    print(30 * '-')
    print('%14s %15.5f' % ('size', sta[0]))
    print('%14s %15.5f' % ('min', sta[1][0]))
    print('%14s %15.5f' % ('max', sta[1][1]))
    print('%14s %15.5f' % ('mean', sta[2]))
    print('%14s %15.5f' % ('std', np.sqrt(sta[3])))
    print('%14s %15.5f' % ('skew', sta[4]))
    print('%14s %15.5f' % ('kurtosis', sta[5]))


# In[14]:


print_statistics(log_returns.flatten())


# In[15]:


log_returns.mean() * M + 0.5 * sigma ** 2


# In[16]:


log_returns.std() * math.sqrt(M)


# In[17]:


plt.figure(figsize=(10,6))
plt.hist(log_returns.flatten(), bins=70, normed=True,
        label='frequency', color='b')
plt.xlabel('log_return')
plt.ylabel('frequency')
x = np.linspace(plt.axis()[0], plt.axis()[1])
plt.plot(x, scs.norm.pdf(x, loc=r / M, scale=sigma / np.sqrt(M)),
        'r', lw=2.0, label='pdf')
plt.legend();


# In[18]:


sm.qqplot(log_returns.flatten()[::500], line='s')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles');


# In[19]:


def normality_tests(arr):
    '''Tests for normality distribution of given data set.
    
    Parameters
    ===========
    array: ndarray
    object to generate statistics on
    '''
    print('Skew of data set %14.3f' % scs.skew(arr))
    print('Skew test p-value %14.3f' % scs.skewtest(arr)[1])
    print('Kurt of data set %14.3f' % scs.kurtosis(arr))
    print('Kurt test p-value %14.3f' % scs.kurtosistest(arr)[1])
    print('Norm test p-value %14.3f' % scs.normaltest(arr)[1])


# In[20]:


normality_tests(log_returns.flatten())


# In[21]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,6))
ax1.hist(paths[-1], bins=30)
ax1.set_xlabel('Index Level')
ax1.set_ylabel('Frequency')
ax1.set_title('Regular Data')
ax2.hist(np.log(paths[-1]), bins=30)
ax2.set_xlabel('Log Index Level')
ax2.set_title('Log Data')


# In[22]:


print_statistics(paths[-1])


# In[23]:


normality_tests(np.log(paths[-1]))


# In[24]:


plt.figure(figsize=(10,6))
log_data = np.log(paths[-1])
plt.hist(log_data, bins=70, normed=True,
        label='observed', color='b')
plt.xlabel('Index Levels')
plt.ylabel('Frequency')
x = np.linspace(plt.axis()[0], plt.axis()[1])
plt.plot(x, scs.norm.pdf(x, log_data.mean(), log_data.std()),
        'r', lw=2.0, label='pdf')
plt.legend();


# In[25]:


sm.qqplot(log_data, line='s')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles');


# In[26]:


symbols = ['SPY', 'GLD', 'AAPL', 'MSFT']


# In[27]:


raw = pdr.get_data_yahoo(symbols, start='2010-01-01')['Close'].dropna()
raw.head()


# In[28]:


data = raw[symbols]
data = data.dropna()


# In[29]:


data.info()


# In[30]:


(data / data.iloc[0] * 100).plot(figsize=(10,6))


# In[31]:


log_returns = np.log(data / data.shift(1))
log_returns.head()


# In[32]:


log_returns.hist(bins=50, figsize=(10,8));


# In[33]:


for sym in symbols:
    print('\nResults for symbol {}'.format(sym))
    print(30 * '-')
    log_data = np.array(log_returns[sym].dropna())
    print_statistics(log_data)


# In[34]:


sm.qqplot(log_returns['SPY'].dropna(), line='s')
plt.title('SPY')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles');
sm.qqplot(log_returns['MSFT'].dropna(), line='s')
plt.title('MSFT')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles');
sm.qqplot(log_returns['AAPL'].dropna(), line='s')
plt.title('AAPL')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
sm.qqplot(log_returns['GLD'].dropna(), line='s')
plt.title('GLD')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')


# In[35]:


for sym in symbols:
    print('\nResults for symbol {}'.format(sym))
    print(32 * '-')
    log_data = np.array(log_returns[sym].dropna())
    normality_tests(log_data)


# In[36]:


noa = len(symbols)
data = raw[symbols]
rets = np.log(data / data.shift(1))
rets.hist(bins=40, figsize=(10, 8));


# In[37]:


rets.mean() * 252


# In[38]:


rets.cov() * 252


# In[39]:


weights = np.random.random(noa)
weights /= np.sum(weights)


# In[40]:


weights


# In[41]:


weights.sum()


# In[42]:


np.sum(rets.mean() * weights) * 252


# In[43]:


np.dot(weights.T, np.dot(rets.cov() * 252, weights))


# In[44]:


math.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))


# In[45]:


def port_ret(weights):
    return np.sum(rets.mean() * weights) * 252


# In[46]:


def port_vol(weights):
    return np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))


# In[47]:


prets = []
pvols = []
for p in range (2500):
    weights = np.random.random(noa)
    weights /= np.sum(weights)
    prets.append(port_ret(weights))
    pvols.append(port_vol(weights))
prets = np.array(prets)
pvols = np.array(pvols)


# In[48]:


plt.figure(figsize=(10,6))
plt.scatter(pvols, prets, c=prets / pvols,
           marker='o', cmap='coolwarm')
plt.xlabel('Expected Volatility')
plt.ylabel('Expected Return')
plt.title('Expected Return and Volatility for Random Portfolio Weights (SPY, AAPL, MSFT, GLD)')
plt.colorbar(label='Sharpe Ratio');

