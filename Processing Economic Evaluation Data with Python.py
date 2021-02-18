#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 07:50:38 2020

"""

import pandas as pd  
import numpy as np

import pandas as pd
import numpy as np
import bokeh as bk
import plotly as pl
import dash
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pandas_datareader import data
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

# DATA PROCESSING

# Goal: Read each CSV and save to separate dataframes 

# Read Information Related to House Price
df_houseprice = pd.read_csv('NorthernEurope.csv')

# Read information on GDP
df_gdp = pd.read_csv('GDPNorthernEurope.csv')

# Read working population
df_pop = pd.read_csv('WorkingPopulationDemographicInfo.csv')

# Read disposable income
df_d_income = pd.read_csv('HouseholdDisposableIncome.csv')

df_d_income.head()

# GOAL: Create dataframes for DNK with housing pricing index, GDP, working pop and income for the 2005 - 2019


#Dropdown
app = dash.Dash()
app.layout = html.Div([
    dcc.Dropdown(
        id='CountryGDP',
        options=[
            {'label': 'Denmark', 'value': 'DNK'},
            {'label': 'Finland', 'value': 'FIN'},
            {'label': 'Iceland', 'value': 'ISL'},
            {'label': 'Norway', 'value': 'NOR'},
            {'label': 'Sweden', 'value': 'SWE'}
        ],
        value='DNK'
    ),
    dcc.Graph(id='chart_plot'),
    dcc.Graph(id='chart_map'),
    dcc.Graph(id='chart_1'),
    dcc.Graph(id='chart_2'),
    dcc.Graph(id='chart_3'),
    dcc.Graph(id='chart_4'),
    html.Div(id = 'graph')
])


@app.callback(
    Output('chart_plot', 'figure'),
    Output('chart_map', 'figure'),
    Output('chart_1', 'figure'),
    Output('chart_2', 'figure'),
    Output('chart_3', 'figure'),
    Output('chart_4', 'figure'),
    [dash.dependencies.Input('CountryGDP', 'value')])
def update_output(value):
    country_code = value
    
    # Get the housing information for LOCATION = DNK, SUBJECT = NOMINAL #df_houseprice.LOCATION == country_code
    df_temp_houseprice = df_houseprice[(df_houseprice.SUBJECT == 'NOMINAL')]
    
    # Get the GDP for LOCATION = DNK
    df_temp_gdp = df_gdp[(df_gdp.LOCATION == country_code)]
    
    # Get working population for LOCATION = DNK
    df_temp_pop = df_pop[(df_pop.LOCATION == country_code)]
    
    # Get disposable income for LOCATION = DNK
    df_temp_d_income = df_d_income[(df_d_income.LOCATION == country_code)]
    
    df_temp_d_income
    
    # Goal merge into one dataframe with the following columns: 
    # Year, House_Price, GDP, Working_Population, Disposable_Income
    
    # All tables except housing index start with 2005, so let's just do inner join
    
    # Merge House price and GDP
    df_DNK_info = pd.merge(df_temp_houseprice[['LOCATION','TIME', 'Value']], df_temp_gdp[['TIME', 'Value']], on='TIME')
    df_DNK_info = df_DNK_info.rename(columns = {'Value_x': 'House_Price', 'Value_y': 'GDP'}, inplace = False)
    
    # Merge working population
    df_DNK_info = pd.merge(df_DNK_info, df_temp_pop[['TIME', 'Value']], on='TIME')
    df_DNK_info = df_DNK_info.rename(columns = {'Value': 'Working_Population'}, inplace = False)
    
    # Merge Disposable income
    df_DNK_info = pd.merge(df_DNK_info, df_temp_d_income[['TIME', 'Value']], on='TIME')
    df_DNK_info = df_DNK_info.rename(columns = {'Value': 'Disposable_Income'}, inplace = False)
    
    # df_DNK_info now contains all the information that we want to analyze for the country "DNK"
    df_DNK_info
    
    #countries = df_DNK_info["LOCATION"].unique()

    fig = px.scatter(df_DNK_info, x="GDP", y="House_Price")
    
    
    # For the map
    # Get the GDP and make it a DataFrame that includes all countries
    gdpprojectdata = pd.read_csv('GDPNorthernEurope.csv') 
    gdpprojectdata.columns    
    
    # Clean GDP Data
    gdpprojectdata["LOCATION"].replace({"DNK": "Denmark","FIN":"Finland","ISL":"Iceland", "NOR":"Norway", "SWE": "Sweden"}, inplace=True)
    df_gdp_map_updated = gdpprojectdata.rename(columns={'LOCATION': 'Countries','SUBJECT':'Methodology', 'TIME':'Year'})
    df_gdp_map_updated2 = df_gdp_map_updated.drop(columns=['Flag Codes','FREQUENCY'])
    
    df_gdp_map_updated2.head()


    oecd_removed_df_gdp_map = df_gdp_map_updated2[df_gdp_map_updated['Countries'] != 'OECD']
    
    #Verify that OECD does not show up in DF
    oecd_removed_df_gdp_map
    
    #Unique Values for Countries
    cn = oecd_removed_df_gdp_map['Countries'].unique()
    #cn
    
    oecd_removed_df_gdp_map['Countries'].value_counts()
    
    year = oecd_removed_df_gdp_map['Year'].unique()
    
    #year
    
    #oecd_removed_df_gdp_map.to_csv('C:/temp/data.csv')
    
    fig_map = px.choropleth(oecd_removed_df_gdp_map, locationmode = 'country names', locations = 'Countries',
                        color = 'Value',
                       hover_name = 'Countries',
                        animation_frame = oecd_removed_df_gdp_map['Year'],
                       color_continuous_scale = px.colors.sequential.Plasma)
    fig_map.update_layout(transition = {'duration': 100})  

    # Goal: Create timeseries for the housing index
    fig_ts_house_price = go.Figure([go.Scatter(x=df_DNK_info['TIME'],y=df_DNK_info['House_Price'])])
    fig_ts_house_price.update_layout(title = 'House Price Trend',
                            yaxis_title = "House Price (Nominal)")
    fig_ts_house_price.show()
   
 
    # Remove the date on the scatterplot matrix
    fig_mult_scatter = px.scatter_matrix(df_DNK_info.iloc[:,1:5])
    #fig_mult_scatter.show()
    
    # Goal: Create timeseries for GDP
    fig_ts_gdp = go.Figure([go.Scatter(x=df_DNK_info['TIME'],y=df_DNK_info['GDP'])])
    fig_ts_gdp.update_layout(title = 'GDP Trend',
                            yaxis_title = "GDP")
    #fig_ts_gdp.show()
    
    # Goal: Create timeseries for Working Population
    fig_ts_working_pop = go.Figure([go.Scatter(x=df_DNK_info['TIME'],y=df_DNK_info['Working_Population'])])
    fig_ts_working_pop.update_layout(title = 'Working Population Demographic Info',
                            yaxis_title = "WKGPOP")
    #fig_ts_working_pop.show()

    return fig, fig_map, fig_ts_house_price, fig_mult_scatter, fig_ts_gdp, fig_ts_working_pop
    #return 'You have selected "{}"'.format(value)

if __name__ == '__main__':
    app.run_server(debug=True)
