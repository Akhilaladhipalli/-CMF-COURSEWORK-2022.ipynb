#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import required libaries
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.stats as si


# In[2]:


Eth_USD = yf.download("Eth-USD", start="2020-01-01", end="2022-01-01")


# In[3]:


Eth_USD.head()


# In[4]:


Eth_USD.tail()


# In[5]:


adj_close =  Eth_USD['Adj Close'] #Take only Adj Close 
close = Eth_USD['Close']


# In[6]:


Eth_USD[['Adj Close']].head()


# In[7]:


data = Eth_USD[['Adj Close']]
data.head()


# In[8]:


data.describe().round(2)


# In[9]:


Eth_USD['SMA1'] = Eth_USD['Adj Close'].rolling(window=20).mean()
Eth_USD['SMA2'] = Eth_USD['Adj Close'].rolling(window=60).mean()
Eth_USD[['Adj Close', 'SMA1', 'SMA2']].tail()


# In[10]:


Eth_USD.dropna(inplace=True)
Eth_USD['positions'] = np.where(Eth_USD['SMA1'] > Eth_USD['SMA2'],1,-1)
ax = Eth_USD[['Adj Close', 'SMA1', 'SMA2', 'positions']].plot(figsize=(10, 6),secondary_y='positions')
ax.get_legend().set_bbox_to_anchor((0.25, 0.85))


# In[11]:


data.plot(figsize=(10, 7), subplots=True)


# In[12]:


data.plot(figsize=(10, 7), subplots=True)


# In[14]:


log_return = np.log(Eth_USD['Adj Close'] / Eth_USD['Adj Close'].shift(1))


# In[15]:


vol = log_return.std()    #daily volatility
print('The daily volatility is', round(vol*100,2), '%')


# In[16]:


Eth_adj_close = pd.DataFrame(adj_close.dropna())


# In[17]:


normal_return = Eth_adj_close.pct_change()
normal_return.head()


# In[18]:


dfnr = pd.DataFrame(normal_return, columns = ['Adj Close']) 
nr = dfnr.mean() * 730
nv = dfnr.std() * (730 ** 0.5)
print('The annualized normal return is %.8f and its annualized volatility is %.8f' % (nr,nv))


# In[19]:


log_rets = np.log(Eth_adj_close / Eth_adj_close.shift(1))
log_rets.head().round(4)


# In[20]:


dflr = pd.DataFrame(log_rets, columns = ['Adj Close']) 
lr = dflr.mean() * len(dflr)
lv = dflr.std() * (len(dflr) ** 0.5)
print('The annualized log return is %.8f and its annualized volatility is %.8f ' % (lr,lv))


# In[21]:


log_return_last_3months=log_rets[-90:]
log_return_last_3months


# In[22]:


dflr1 = pd.DataFrame(log_return_last_3months, columns = ['Adj Close']) 
lr1 = dflr1.mean() * len(dflr)
lv1 = dflr1.std() * (len(dflr) ** 0.5)
print('The annualized log return (for the last 3 months) is %.8f and its annualized volatility is %.8f' % (lr1,lv1))


# In[23]:


log_return_mid_year=log_rets[150:-90]
log_return_mid_year


# In[24]:


dflr2 = pd.DataFrame(log_return_mid_year, columns = ['Adj Close']) 
lr2 = dflr2.mean() * len(dflr)
lv2 = dflr2.std() * (len(dflr) ** 0.5)
print('The mid year annualized  log return (for Mar 1 - Sep 1) is %.8f and its annualized volatility is %.8f' % (lr2,lv2))


# In[25]:


all_lv=lv,lv2,lv1
all_lv1 = pd.DataFrame (all_lv)


# In[26]:


lv_avg=all_lv1 ['Adj Close'].mean()
print ('The combine Annualized Log volatility =', lv_avg)


# In[27]:


all_lr = lr,lr2,lr1
all_lr1 = pd.DataFrame (all_lr)


# In[28]:


lr_avg=all_lr1 ['Adj Close'].mean()
print ('The combine Annualized Log Return =', lr_avg)


# In[29]:


fig = plt.figure()
fig.set_size_inches(15.5, 8.5, forward=True)
plt.plot(Eth_USD['Close'])
plt.xlabel('Days')
plt.ylabel('Price ($)')
plt.title('Closed Price');


# In[30]:


fig = plt.figure()
fig.set_size_inches(15.5, 8.5, forward=True)
plt.plot(dflr * 100)
plt.xlabel('Days')
plt.ylabel('Percentage %')
plt.title('Log Return')


# In[31]:


str_vol = str(round(lv_avg, 4)*100)
fig, ax = plt.subplots()
dflr['Adj Close'].hist(ax=ax, bins=50, alpha=0.6, color='b')
ax.set_xlabel('Log return')
ax.set_ylabel('Freq of log return')
ax.set_title('Eth-USD Average annualized volatility:'+ str_vol + '%')


# In[32]:


S = Eth_USD['Adj Close'][-1]
print('The spot price is', round(S,2))


# In[72]:


S0 = 3769.7             # spot stock price
K = 85                 # strike
T = 1/52                # maturity 
r = 0.0169              # risk free rate 
sig = 1.53              # diffusion coefficient or volatility
N = 4                   # number of periods or number of time steps  
payoff = "put"  


# In[73]:


dT = float(T) / N                             # Delta t
u = np.exp(sig * np.sqrt(dT))                 # up factor
d = 1.0 / u                                   # down factor 


# In[74]:


print('up factor', u)


# In[75]:


print('down factor' , d)


# In[76]:


S = np.zeros((N + 1, N + 1))
S[0, 0] = S0
z = 1
for t in range(1, N + 1):
    for i in range(z):
        S[i, t] = S[i, t-1] * u
        S[i+1, t] = S[i, t-1] * d
    z += 1


# In[77]:


S


# In[78]:


a = np.exp(r * dT)    # risk free compound return
p = (a - d)/ (u - d)  # risk neutral up probability
q = 1.0 - p           # risk neutral down probability
p


# In[79]:


S_T = S[:,-1]
V = np.zeros((N + 1, N + 1))
if payoff =="call":
    V[:,-1] = np.maximum(S_T-K, 0.0)
elif payoff =="put":
    V[:,-1] = np.maximum(K-S_T, 0.0)
V


# In[80]:


# for European Option
for j in range(N-1, -1, -1):
    for i in range(j+1):
        V[i,j] = np.exp(-r*dT) * (p * V[i,j + 1] + q * V[i + 1,j + 1])
V


# In[81]:


# for European Option
for j in range(N-1, -1, -1):
    for i in range(j+1):
        V[i,j] = np.exp(-r*dT) * (p * V[i,j + 1] + q * V[i + 1,j + 1])
V


# In[82]:


print('European ' + payoff, str( V[0,0]))


# In[54]:


p = np.mean(np.maximum(K - S[:,-1],0))
print('European call', str(p))


# In[83]:


def mcs_simulation_np(n,p):
    M = n
    I = p
    dt = T / M 
    S = np.zeros((M + 1, I))
    S[0] = S0 
    rn = np.random.standard_normal(S.shape) 
    for t in range(1, M + 1): 
        S[t] = S[t-1] * np.exp((r - sigma ** 2 / 2) * dt + sigma * np.sqrt(dt) * rn[t]) 
    return S


# In[84]:


T = 1/52
r = 0.0169
sigma = 1.53
S0 = 3769.7
K = 85


# In[85]:


S = mcs_simulation_np(100,10000)


# In[86]:


S = np.transpose(S)
S


# In[87]:


n, bins, patches = plt.hist(x=S[:,-1], bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)

plt.grid(axis='y', alpha=0.75)
plt.xlabel('S_T')
plt.ylabel('Frequency')
plt.title('Frequency distribution of the simulated end-of-preiod values')


# In[88]:


p = np.mean(np.maximum(K - S[:,-1],0))
print('European put', str(p))


# In[90]:


def delta(S, K, T, r, q, vol, payoff):
    
    d1 = (np.log(S / K) + (r - q + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    if payoff == "call":
        delta = np.exp(-q * T) * si.norm.cdf(d1, 0.0, 1.0)
    elif payoff == "put":
        delta =  - np.exp(-q * T) * si.norm.cdf(-d1, 0.0, 1.0)
    
    return delta


# In[97]:


delta(3769, 3400, 1/24, 1.69, 0.0163, 0.86, 'put') # value of delta


# In[98]:


S = np.linspace(20,150,51)
Delta_Put = np.zeros((len(S),1))
for i in range(len(S)):
    Delta_Put [i] = delta(S[i], 3400, 1/52, 0.0169, 0, 1.53, 'put')


# In[99]:


fig = plt.figure()
plt.plot(S, Delta_Put, '-')
plt.grid()
plt.xlabel('Stock Price')
plt.ylabel('Delta')
plt.title('Delta')
plt.legend(['Delta for Put'])


# In[100]:


S = np.linspace(20, 150, 51)
T = np.linspace(0.5, 2, 51)
Delta = np.zeros((len(T),len(S)))
for j in range(len(S)):
    for i in range(len(T)):
        Delta[i,j] = delta(S[j], 3400, T[i], 0.0169, 0, 1.53, 'put')


# In[101]:


fig = plt.figure(figsize=(10, 6))
ax = fig.gca(projection='3d')
S, T = np.meshgrid(S, T)
surf = ax.plot_surface(S, T, Delta, rstride=2, cstride=2, cmap=plt.cm.PiYG, linewidth=0.5, antialiased=True)
ax.set_xlabel('Stock Price')
ax.set_ylabel('Time to Expiry')
ax.set_zlabel('Delta')
fig.colorbar(surf, shrink=0.5, aspect=5);


# In[103]:


d = delta(3769, 3400, 1/52, 0.0169, 0, 1.53, 'put')
print('The value of Delta is', d.round(4),'.','If the cypto eth price increase 1 dollar, then the value of the option will increase $', d.round(4), '.')


# In[104]:


def gamma(S, K, T, r, q, vol, payoff):
    
    d1 = (np.log(S / K) + (r - q + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    gamma = np.exp(- q * T) * si.norm.pdf(d1, 0.0, 1.0) / (vol * S * np.sqrt(T))
    
    return gamma


# In[105]:


gamma(3769, 3400, 1/24, 1.69, 0.0163, 0.86, 'put') # value of gamma


# In[106]:


S = np.linspace(20,150,51)
Gamma = np.zeros((len(S),1))
for i in range(len(S)):
    Gamma [i] = gamma(S[i], 3400, 1/52, 0.0169, 0, 1.53, 'put')


# In[107]:


fig = plt.figure()
plt.plot(S, Gamma, '-')
plt.grid()
plt.xlabel('Stock Price')
plt.ylabel('Gamma')
plt.title('Gamma')
plt.legend(['Gamma for Put'])


# In[108]:


S = np.linspace(20, 150, 51)
T = np.linspace(0.5, 2, 51)
Gamma = np.zeros((len(T),len(S)))
for j in range(len(S)):
    for i in range(len(T)):
        Gamma[i,j] = gamma(S[j], 3400, T[i], 0.0169, 0, 1.53, 'put')


# In[109]:


fig = plt.figure(figsize=(10, 6))
ax = fig.gca(projection='3d')
S, T = np.meshgrid(S, T)
surf = ax.plot_surface(S, T, Gamma, rstride=2, cstride=2, cmap=plt.cm.PRGn, linewidth=0.5, antialiased=True)
ax.set_xlabel('Stock Price')
ax.set_ylabel('Time to Expiry')
ax.set_zlabel('Gamma')
fig.colorbar(surf, shrink=0.5, aspect=5);


# In[110]:


def theta(S, K, T, r, q, vol, payoff):
    
    d1 = (np.log(S / K) + (r - q + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    if payoff == "call":
        theta = vol * S * np.exp(-q * T) * si.norm.pdf(d1, 0.0, 1.0) / (2 * np.sqrt(T)) - q * S * np.exp(-q * T) * si.norm.cdf(d1, 0.0, 1.0) + r * K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
    elif payoff == "put":
        theta = vol * S * np.exp(-q * T) * si.norm.pdf(-d1, 0.0, 1.0) / (2 * np.sqrt(T)) - q * S * np.exp(-q * T) * si.norm.cdf(-d1, 0.0, 1.0) + r * K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0)
    
    return theta


# In[111]:


theta(3769, 3400, 1/24, 1.69, 0.0163, 0.86, 'put') # value of theta


# In[112]:


T = np.linspace(0.25,7,12)
Theta_Put = np.zeros((len(T),1))
for i in range(len(T)):
    Theta_Put [i] = theta(3769, 28, T[i], 0.0169, 0, 1.53, 'put')


# In[113]:


fig = plt.figure()
plt.plot(T, Theta_Put, '-')
plt.grid()
plt.xlabel('Time to Expiry')
plt.ylabel('Theta')
plt.title('Theta')
plt.legend(['Theta for Put'])


# In[114]:


S = np.linspace(20, 150, 51)
T = np.linspace(0.5, 2, 51)
Theta = np.zeros((len(T),len(S)))
for j in range(len(S)):
    for i in range(len(T)):
        Theta[i,j] = theta(S[j], 3400, T[i], 0.0169, 0, 1.53, 'put')


# In[115]:


fig = plt.figure(figsize=(10, 6))
ax = fig.gca(projection='3d')
S, T = np.meshgrid(S, T)
surf = ax.plot_surface(S, T, Theta, rstride=2, cstride=2, cmap=plt.cm.Spectral, linewidth=0.5, antialiased=True)
ax.set_xlabel('Stock Price')
ax.set_ylabel('Time to Expiry')
ax.set_zlabel('Theta')
fig.colorbar(surf, shrink=0.5, aspect=5);


# In[116]:


def rho(S, K, T, r, q, vol, payoff):
    
    d1 = (np.log(S / K) + (r - q + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    if payoff == "call":
        rho =  K * T * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
    elif payoff == "put":
        rho = - K * T * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0)
    
    return rho


# In[117]:


rho(3769, 3400, 1/24, 1.69, 0.0163, 0.86, 'put') # value of rho


# In[118]:


r = np.linspace(0,0.8,51)
Rho_Put = np.zeros((len(r),1))
for i in range(len(r)):
    Rho_Put [i] = rho(3769, 3400, 1/52, r[i], 0, 1.53, 'put')


# In[119]:


fig = plt.figure()
plt.plot(r, Rho_Put, '-')
plt.grid()
plt.xlabel('Interest Rate')
plt.ylabel('Rho')
plt.title('Rho')
plt.legend(['Rho for Put'])


# In[120]:


S = np.linspace(20, 150, 51)
T = np.linspace(0.5, 2, 51)
Rho = np.zeros((len(T),len(S)))
for j in range(len(S)):
    for i in range(len(T)):
        Rho[i,j] = rho(S[j], 3400, T[i], 0.0169, 0, 1.53, 'put')


# In[121]:


fig = plt.figure(figsize=(10, 6))
ax = fig.gca(projection='3d')
S, T = np.meshgrid(S, T)
surf = ax.plot_surface(S, T, Rho, rstride=2, cstride=2, cmap=plt.cm.BrBG, linewidth=0.5, antialiased=True)
ax.set_xlabel('Stock Price')
ax.set_ylabel('Time to Expiry')
ax.set_zlabel('Rho')
fig.colorbar(surf, shrink=0.5, aspect=5);


# In[122]:


def vega(S, K, T, r, q, vol, payoff):
    
    d1 = (np.log(S / K) + (r - q + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    vega = S * np.sqrt(T) * np.exp(-q * T) * si.norm.pdf(d1, 0.0, 1.0)
    
    return vega


# In[123]:


vega(3769, 3400, 1/24, 1.69, 0.0163, 0.86, 'put') # value of vega


# In[124]:


vol = np.linspace(0.1,2,21)
Vega = np.zeros((len(vol),1))
for i in range(len(vol)):
    Vega [i] = vega(3769, 3400, 1/52, 0.0169, 0, vol[i], 'put')


# In[125]:


fig = plt.figure()
plt.plot(vol, Vega, '-')
plt.grid()
plt.xlabel('Volatility')
plt.ylabel('Vega')
plt.title('Vega')
plt.legend(['Vega for Put'])


# In[126]:


S = np.linspace(20, 150, 51)
T = np.linspace(0.5, 2, 51)
Vega = np.zeros((len(T),len(S)))
for j in range(len(S)):
    for i in range(len(T)):
        Vega[i,j] = vega(S[j], 3400, T[i], 0.0169, 0, 1.53, 'put')


# In[127]:



fig = plt.figure(figsize=(10, 6))
ax = fig.gca(projection='3d')
S, T = np.meshgrid(S, T)
surf = ax.plot_surface(S, T, Vega, rstride=2, cstride=2, cmap=plt.cm.seismic, linewidth=0.5, antialiased=True)
ax.set_xlabel('Stock Price')
ax.set_ylabel('Time to Expiry')
ax.set_zlabel('Vega')
fig.colorbar(surf, shrink=0.5, aspect=5);


# In[128]:


v = vega(3769, 3400, 1/52, 0.0169, 0, 1.53, 'put')
print('The value of Vega is', v.round(4),'.','If the volatility increases 1%, then the value of the option will increase $', v.round(4)*0.01, '.')


# In[129]:


def speed(S, K, T, r, q, vol, payoff):
    
    d1 = (np.log(S / K) + (r - q + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    speed = - np.exp(-q * T) * si.norm.pdf(d1, 0.0, 1.0) / ((vol **2) * (S**2) * np.sqrt(T)) * (d1 + vol * np.sqrt(T))
    
    return speed


# In[130]:


speed(3769, 3400, 1/24, 1.69, 0.0163, 0.86, 'put') # value of speed


# In[131]:


S = np.linspace(20,150,51)
Speed = np.zeros((len(S),1))
for i in range(len(S)):
    Speed [i] = speed(S[i], 3400, 1/52, 0.0169, 0, 1.53, 'put')


# In[132]:


fig = plt.figure()
plt.plot(S, Speed, '-')
plt.grid()
plt.xlabel('Stock Price')
plt.ylabel('Speed')
plt.title('Speed')
plt.legend(['Speed for Put'])


# In[133]:


S = np.linspace(20, 150, 51)
T = np.linspace(0.5, 2, 51)
Speed = np.zeros((len(T),len(S)))
for j in range(len(S)):
    for i in range(len(T)):
        Speed[i,j] = speed(S[j], 3400, T[i], 0.0169, 0, 1.53, 'put')


# In[134]:


fig = plt.figure(figsize=(10, 6))
ax = fig.gca(projection='3d')
S, T = np.meshgrid(S, T)
surf = ax.plot_surface(S, T, Speed, rstride=2, cstride=2, cmap=plt.cm.RdYlGn, linewidth=0.5, antialiased=True)
ax.set_xlabel('Stock Price')
ax.set_ylabel('Time to Expiry')
ax.set_zlabel('Speed')
fig.colorbar(surf, shrink=0.5, aspect=5);


# In[135]:


#END


# In[ ]:




