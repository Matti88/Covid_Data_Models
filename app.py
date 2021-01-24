import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt_
from statsmodels.graphics.tsaplots import plot_predict
from statsmodels.tsa.arima.model import ARIMA


# web scraping data for covid
@st.cache
def load_data():

    try:
        url = """https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"""
        data = pd.read_csv(url)
    except:
        local = """.\owid-covid-data.csv"""
        data = pd.read_csv(local)

    return data

def plotting_chart_latestValues(df, positioned_x_y_hue):

    latest_date = max(df['date'].unique())
    latest_date = latest_date - pd.Timedelta(days=1)
    latest_per_country = df[df['date']== latest_date]
    print_date = str(latest_date)[:10]

    x_ , y_, hue_ = positioned_x_y_hue[0], positioned_x_y_hue[1], positioned_x_y_hue[2]
    df = df[positioned_x_y_hue]

    st.write("""### Time-line of {} by country at date {}""".format(y_, print_date))

    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    f, ax = plt.subplots(figsize=(10,6))
    ax = sns.lineplot(data=df[df.location.isin(selected_countries) ], x=x_ , y=y_, hue=hue_)
    st.pyplot(f)
 
    #Some Specifics
    print_our_new_cases = """### Latest {} at date {}:""".format(y_, print_date)
 
    for country in selected_countries:
        on_display = list(latest_per_country[latest_per_country['location']== country][y_])[0]
        print_our_new_cases = print_our_new_cases + """\n- {} : {} """.format(country, round(on_display,2))

    st.write(print_our_new_cases)

#correlation functions
def previous_days_corr_effects(df, smaller_set,  constant_columns , days_= 30, suffix='_30daysago'):
    #df = df[df.location.isin(selected_countries) ]
    sel_col_df = df[smaller_set]
    sel_col_df = sel_col_df.merge(sel_col_df.assign(date = df.date+pd.Timedelta(days=days_)), on=constant_columns, how='left', suffixes=['', suffix])
    return sel_col_df.corr()

def plot_corr_heatmap(corr):
    sns.set_theme(style="white")
 
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    ax = sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
 
    st.pyplot(f)

def search_highest_correlations(corr, min_corr=0.86):
    list_interesting_corr = []
    values_investigated = []
    columns_available = corr.columns.to_list()
    for row_ in corr.iterrows():
        for col_value in row_[1:]:
            for col_name in columns_available:
                if abs(col_value[col_name]) >= min_corr and abs(col_value[col_name]) < 1.0 and (col_name not in values_investigated):
                    tuple_with_high_correlation = [row_[0] + ' ~ ' + col_name ,col_value[col_name] ]
                    list_interesting_corr.append(tuple_with_high_correlation)

        values_investigated.append(row_[0])
    return list_interesting_corr

def lagged_days(element, col_suffix='daysago'):
    list_ofitems = element.split(' ~ ')
    item1, item2 = list_ofitems[0], list_ofitems[1]
    enditem_1, enditem_2 = item1.endswith(col_suffix), item2.endswith(col_suffix)
    if (enditem_1 == True and enditem_2 == False) or (enditem_1 == False and enditem_2 == True):
        return True
    else:
        return False

#ARIMA STUFF
def data_clean_up(ds_):
    return ds_.ffill().bfill().dropna().asfreq(freq='D')

def Arima_And_plot(y, country):
    arma_mod = ARIMA(y, order=(5, 0, 5), trend='n')
    arma_res = arma_mod.fit()
    
    SARIMAX_results = arma_res.summary()

    start_date , end_date = max(y.index)- pd.Timedelta(days=90), max(y.index) + pd.Timedelta(days=10)
 
    st.write('## ARIMA model for {}'.format(country))
    st.text(SARIMAX_results)
    
    st.write('## Forecasts for {}'.format(country))
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    f, ax = plt.subplots(figsize=(10,8))
    date_start = max(y.index)- pd.Timedelta(days=90)
    ax.plot(y[(y.index >= date_start )], 'r', label='Actuals')
    f = plot_predict(arma_res, start= start_date , end=end_date  , ax=ax)
    ax.legend(loc='upper right')
    st.pyplot(f)


#loading and displaying data
covid_data = load_data() 
covid_data = covid_data[covid_data.continent == 'Europe']
covid_data['date'] =  pd.to_datetime(covid_data ['date'])
date_criteria = covid_data['date'] > '2020-11-01'
covid_data = covid_data[date_criteria]
countries = list(covid_data.location.unique())


#at the sidebar
st.sidebar.header("Select what to predict")
selected_countries = st.sidebar.multiselect('Countries', countries, ['Italy','France', 'Slovakia', 'United Kingdom', 'Germany'])


#main page
st.write("# European Covid - Data")
st.markdown("What to expect for tomorrows Covid news\n - Data source: [Open Data Bank] (https://github.com/owid/covid-19-data) ")



#New Cases Smoothed Data
positioned_x_y_hue = ['date', 'new_cases_smoothed', 'location']
plotting_chart_latestValues(covid_data, positioned_x_y_hue)

#New Cases new_deaths_smoothed  Data
positioned_x_y_hue = ['date', 'new_deaths_smoothed', 'location']
plotting_chart_latestValues(covid_data, positioned_x_y_hue)



#section for correlation plots
smaller_set = ['date', 'continent', 'location', 'new_cases_per_million','new_deaths_per_million','reproduction_rate','icu_patients_per_million','hosp_patients_per_million'
               ,'weekly_icu_admissions_per_million','weekly_hosp_admissions_per_million','total_tests_per_thousand','new_tests_per_thousand',
               'positive_rate','tests_per_case','total_vaccinations_per_hundred','new_vaccinations','stringency_index']
const_cols =  ['date', 'continent', 'location', 'new_cases_per_million','new_deaths_per_million']



# Compute the correlation matrix
lag_days = 5
suffix_ = '_'+str(lag_days)+'daysago'

corr = previous_days_corr_effects(covid_data, smaller_set, const_cols,  days_= lag_days, suffix=suffix_)

st.write("### Correlation with {} lagged days ".format(lag_days))
plot_corr_heatmap(corr)

#search for the highest correlations
print_our_new_cases = """### Notable Correlations:"""
noted_correlations = search_highest_correlations(corr, .90)
for var in list(filter(lambda x: lagged_days(x[0]),noted_correlations)):
    print_our_new_cases = print_our_new_cases + """\n- {} : {} """.format(var[0], str(round(var[1],2)))

st.write(print_our_new_cases)

#plotting arima
covid_data_arima = load_data() 
covid_data_arima = covid_data_arima[covid_data_arima.continent == 'Europe']
covid_data_arima['date'] =  pd.to_datetime(covid_data_arima['date'])
date_criteria = covid_data_arima['date'] > '2020-02-01'
covid_data_arima = covid_data_arima[date_criteria]

st.write("# European Covid - Prediction")
italy_only = covid_data_arima[covid_data_arima.location == 'Italy']
italy_only.index = italy_only.date
italy_only_new_cases_smoothed = italy_only.new_cases_smoothed
y = data_clean_up(italy_only_new_cases_smoothed)
Arima_And_plot(y, 'Italy')




