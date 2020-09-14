
import pandas as pd
import numpy as np

from datetime import datetime

%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns


sns.set(style="darkgrid")

mpl.rcParams['figure.figsize'] = (16, 9)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', None)

from scipy import optimize
from scipy import integrate


def data_gathering():
    population_df = pd.read_csv('C:/Users/jitin/ads_covid-19/data/raw/world_population_data.csv',sep=';', thousands=',')
    population_df = population_df.set_index(['country']).T
    df_analyse = pd.read_csv('C:/Users/jitin/ads_covid-19/data/processed/all_country_data.csv',sep=';')
    country_list = df_analyse.columns[1:]

    infected_list = []
    t=[]

    for column in df_analyse.columns:
        infected_list.append(np.array(df_analyse[column][75:]))

    t = np.arange(len(infected_list))
    infected_list = pd.DataFrame(infected_list,index=df_analyse.columns).T
    infected_list.to_csv('C:/Users/jitin/ads_covid-19/data/processed/SIR/SIR_data.csv',sep=';',index=False)
    optimized_df = pd.DataFrame(columns = df_analyse.columns[1:],
                     index = ['opt_beta', 'opt_gamma', 'std_dev_error_beta', 'std_dev_error_gamma'])


    t = []
    fitted_final_data = []

    global I0, N0, S0, R0
    for column in infected_list.columns[1:]:
        I0 = infected_list[column].loc[0]
        N0 = population_df[column].loc['population']
        S0 = N0-I0
        R0 = 0
        t  = np.arange(len(infected_list[column]))

        popt=[0.4,0.1]

        fit_odeint(t, *popt)


        popt, pcov = optimize.curve_fit(fit_odeint, t, infected_list[column], maxfev=5000)
        perr = np.sqrt(np.diag(pcov))



        optimized_df.at['opt_beta', column] = popt[0]
        optimized_df.at['opt_gamma', column] = popt[1]
        optimized_df.at['std_dev_error_beta', column] = perr[0]
        optimized_df.at['std_dev_error_gamma', column] = perr[1]

        fitted = fit_odeint(t, *popt)
        fitted_final_data.append(np.array(fitted))

    optimized_df.to_csv('C:/Users/jitin/ads_covid-19/data/processed/SIR/optimized_SIR_data.csv',sep=';',index=False)
    fitted_SIR_data_df = pd.DataFrame(fitted_final_data,index=df_analyse.columns[1:]).T
    fitted_SIR_data_df.to_csv('C:/Users/jitin/ads_covid-19/data/processed/SIR/fitted_SIR_data.csv',sep=';',index=False)
    print(' Number of rows stored in optimized df: '+str(optimized_df.shape[0]))
    print(' Number of rows stored in fitted SIR data: '+str(fitted_SIR_data_df.shape[0]))


def SIR_model_t(SIRN,t,beta,gamma):
    ''' Simple SIR model
        S: susceptible population
        t: time step, mandatory for integral.odeint
        I: infected people
        R: recovered people
        beta:

        overall condition is that the sum of changes (differnces) sum up to 0
        dS+dI+dR=0
        S+I+R= N (constant size of population)

    '''

    S,I,R,N=SIRN
    dS_dt=-beta*S*I/N          #S*I is the
    dI_dt=beta*S*I/N-gamma*I
    dR_dt=gamma*I
    dN_dt=0
    return dS_dt,dI_dt,dR_dt,dN_dt


def fit_odeint(t, beta, gamma):

    '''
    helper function for the integration
    '''
    return integrate.odeint(SIR_model_t, (S0, I0, R0, N0), t, args=(beta, gamma))[:,1] # we only would like to get dI









if __name__ == '__main__':
    # test_data_reg=np.array([2,4,6])
    # result=get_doubling_time_via_regression(test_data_reg)
    # print('the test slope is: '+str(result))
    #
    # pd_JH_data=pd.read_csv('C:/Users/jitin/ads_covid-19/data/processed//COVID_relational_confirmed.csv',sep=';',parse_dates=[0])
    # pd_JH_data=pd_JH_data.sort_values('date',ascending=True).copy()
    #
    # #test_structure=pd_JH_data[((pd_JH_data['country']=='US')|
    # #                  (pd_JH_data['country']=='Germany'))]
    #
    # pd_result_larg=calc_filtered_data(pd_JH_data)
    # pd_result_larg=calc_doubling_rate(pd_result_larg)
    # pd_result_larg=calc_doubling_rate(pd_result_larg,'confirmed_filtered')
    #
    #
    # mask=pd_result_larg['confirmed']>100
    # pd_result_larg['confirmed_filtered_DR']=pd_result_larg['confirmed_filtered_DR'].where(mask, other=np.NaN)
    # pd_result_larg.to_csv('C:/Users/jitin/ads_covid-19/data/processed/COVID_final_set.csv',sep=';',index=False)
    # print(pd_result_larg[pd_result_larg['country']=='Germany'].tail())
    data_gathering()
