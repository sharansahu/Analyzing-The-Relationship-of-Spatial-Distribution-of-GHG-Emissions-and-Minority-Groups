#!/usr/bin/env python
# coding: utf-8

# ## ESPM 50AC
# 
# ### Final Project: Using Data Science To Illustrate The Relationship Between Concentrated Marginalized Groups and the Spatial Distribution of GHG Emissions 
# 
# 
# #### Sharan Sahu
# 
# ## Introduction
# 
#  As we have discussed multiple times in lectures and discussions, global climate change seems to impact marginalized groups the most although they have contributed the least to climate change. A recent field of study that seeks to use datasets and data analysis to create insights about the impact of climate change on different social groups is called environmental data science. Furthermore, in documentaries such as Fence Line, we have seen how oil and power companies are located near marginalized communities, thus creating disparities in terms of climate change effects on these groups. This creative project seeks to use demographic datasets and datasets from the Environmental Protection Agency regarding greenhouse gas reporting to visualize the relationship between the spatial distribution of marginalized groups and GHG Emissions. This will use technologies such as Python and various data science packages to create visualizations. As an extension of this project if time permits, I will attempt to use machine learning techniques such as forecasting to illustrate a time series of how the spatial distribution of GHG emissions has been correlated with the spatial distribution of marginalized groups and how it may change in the future. This project also seeks to create insights that we have not discussed in class such as the relationship between the spatial distribution of GHG emissions and other minority groups that we have not investigated thoroughly.
# 

# In[1]:


import numpy as np
import pandas as pd
import geopandas as gpd
import folium
# import contextily as ctx
import geoplot.crs as gcrs
import cartopy.crs as ccrs
import cartopy
import geoplot as gplt
import matplotlib as mpl
from matplotlib import pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
from folium.plugins import MarkerCluster, HeatMap

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
plt.style.use("seaborn")


# ## Exploratory Data Analysis And Cleaning For EPA GHG Reporting Dataset
# 
# We first load up the data to see what the data looks like. The columns of the dataset are wordy, so we can change th e names of the columns to be just the word following the period. We can also disregard any data that will not be used within the context of this project.
# 
# The dataset used is from [EPA Greenhouse Gas Reporting Program (GHGRP)](https://www.epa.gov/ghgreporting/ghg-reporting-program-data-sets). The GHGRP requires facilities emitting 25,000 metric tons ($mt$) of carbon dioxide equivalent ($CO_2e$) per year to report their emissions and many other related information annually.

# In[2]:


emissions_data_summary = pd.read_csv("data_summary.CSV")


# In[3]:


emissions_data_summary.head()


# In[4]:


new_column_names = []
for columns in emissions_data_summary.columns:
    new_column_names.append(columns.split('.')[1])
emissions_data_summary.columns = new_column_names


# In[5]:


data_formatted = emissions_data_summary[['ADDRESS1', 'ADDRESS2', 'CITY', 'CO2E_EMISSION', 'COUNTY',
       'FACILITY_ID', 'GAS_CODE', 'GAS_NAME', 'LATITUDE', 'LONGITUDE', 'STATE',
       'STATE_NAME', 'YEAR', 'ZIP', 'FACILITY_NAME', 'GHG_QUANTITY']]
data_formatted


# ## Data Insights and Analysis For EPA GHG Reporting Dataset
# 
# Below, we are given some insights into the GHG emissions broken down by state between the period of 2010-2019. Surprisingly, in terms of total GHG emissions during this period, Texas has been the greatest contributor of GHG emissions by a large margin compared to other states. In 2019, Texas became the nation's number 1 carbon dioxide  emitter and according to an article posted by the Federal Reserve Bank of Dallas by authors Emma Marshall and Jesse Thompson, "Texas’ high industrial share largely results from the production of energy-intensive goods such as motor fuels and petrochemicals, which yields substantial CO2 waste. The state is home to 30 percent of U.S. refining capacity and 70 percent of the nation’s basic petrochemical capacity." (Marshall and Thompson 10) [(Link to Article)](https://www.dallasfed.org/~/media/documents/research/swe/2019/swe1903c.pdf)

# In[6]:


state_data = data_formatted.groupby("STATE").aggregate(np.sum)
state_data_plot = state_data.plot(y="GHG_QUANTITY", kind ='bar', figsize = (20, 20), legend = False, fontsize = 18)
state_data_plot.set_title(r"Breakdown Of GHG Emissions By State (2010-2019)", fontsize = 20)
state_data_plot.set_xlabel("State", fontsize = 20)
state_data_plot.set_ylabel("GHG Emissions (Gigatons)", fontsize = 20)
plt.plot()


# Based on the information we have above, it is not surprising to see that Texas was the number one GHG emitter in 2019.

# In[7]:


data_2019 = data_formatted[data_formatted['YEAR'] == 2019]

state_data_2019 = data_2019.groupby("STATE").aggregate(np.sum)

state_data_2019_plot = state_data_2019.plot(y="GHG_QUANTITY", kind ='bar', figsize = (20, 20), legend = False, fontsize = 18)
state_data_2019_plot.set_title('Breakdown Of GHG Emissions By State In 2019', fontsize = 20)
state_data_2019_plot.set_xlabel("State", fontsize = 20)
state_data_2019_plot.set_ylabel(r"GHG Emissions ($10^8$)", fontsize = 20)
plt.plot()


# Below is a breakdown of the GHG emission usage from the period of 2010-2019. As we have seen before, Carbon Dioxide is the largest GHG Emission contributor, and this makes sense to why Texas has jumped up so highly when discussing GHG emissions since in 2019, they became the nation's number 1 carbon dioxide emitter.

# In[8]:


ghg_emission_breakdown = data_formatted.groupby("GAS_NAME").aggregate(np.sum)
ghg_emission_breakdown_plot = ghg_emission_breakdown.plot.pie(y="GHG_QUANTITY", figsize = (20, 20), legend = True, labels = ['' for i in range(len(data_formatted["GAS_NAME"]))])
ghg_emission_breakdown_plot.set_title("Breakdown Of GHG Emission Usage (2010-2019)", fontsize = 20)
ghg_emission_breakdown_plot.set_ylabel("GHG Emission Quantity", fontsize = 20)
total_emissions = sum(ghg_emission_breakdown['GHG_QUANTITY'])
percentages = [round(100.*i/total_emissions, 2)  for i in ghg_emission_breakdown['GHG_QUANTITY']]
ghg_emission_breakdown['GHG Percentage'] = percentages
labels = [
    "Biogenic CO2 - 5.08%",
    "Carbon Dioxide - 94.62%",
    "HFCs - 0%",
    "HFEs - 0%",
    "Methane - 0.29%",
    "Nitrogen Triflouride - 0%",
    "Nitrous Oxide - 0%",
    "Other - 0%",
    "Other Fully Fluorinated GHGs - 0%",
    "PFCs - 0%",
    "Sulfur Hexafluoride - 0%",
    "Very Short-lived Compounds - 0%",
]
plt.legend(labels, fontsize = 12)
plt.plot()


# Although the United States has been one of the leading nations when discussing GHG emissions, it looks like while we are still emitting close to 3 gigatons of GHG, we are slowly making progress to reducing our GHG emissions through policy, adaptation, and mitigation

# In[9]:


year_breakdown = data_formatted.groupby("YEAR").aggregate(np.sum)
year_breakdown_plot = year_breakdown.plot.bar(y = "GHG_QUANTITY", legend = False, figsize = (20, 20), fontsize = 18)
year_breakdown_plot.set_xlabel("Year", fontsize = 20)
year_breakdown_plot.set_ylabel("GHG Emissions (Gigatons) ", fontsize = 20)
year_breakdown_plot.set_title("Progression of GHG Emissions (2010-2019)", fontsize = 20)


# ## Exploratory Data Analysis And Cleaning For 2017 United States Census Dataset
# 
# 
# We will now take a look at the demographics of the United States, particularly with the 2017 Census Data. With this data, we can take into account the demographics of different states within the United States and deduce which states emit the most GHG and have the greatest underrepresented minority groups. Below, we are doing some exploring of the data to see which information is most relevant, and filtering out the information that is not necessary for the analysis.
# 
# The data set used below is from [United States Census Bureau](https://data.census.gov/cedsci/profile?g=0100000US)

# In[10]:


census_data_2017 = pd.read_csv("acs2017_county_data.csv")
census_data_2017.columns


# In[11]:


census_data_formatted = census_data_2017[['CountyId', 'State', 'County', 'TotalPop', 'Men', 'Women', 'Hispanic',
       'White', 'Black', 'Native', 'Asian', 'Pacific']]
census_data_formatted


# ## Data Insights and Analysis For 2017 United States Census Dataset
# 
# Below, we can see a breakdown of minority racial groups within the United States as reported by the 2017 United States Census Dataset. As we see, the majority of minority groups are found within the state of Texas, Georgia, Mississipi, Louisiana, Virginia, and Puerto Rico. In the subsequent following this, we can begin to make some insights into the spatial distribution of minority groups and the spatial distribution of GHG emissions within the United States.

# In[12]:


census_data_race_breakdown = census_data_formatted.groupby("State").aggregate(np.sum)
census_data_race_breakdown_plot = census_data_race_breakdown.plot.bar(y = ['Hispanic', 'Black', 'Native'], figsize = (15, 15), legend = True, fontsize = 18, width = 1)
census_data_race_breakdown_plot.set_title("Breakdown of Minority Racial Groups In The United States (2017 Census)", fontsize = 20)
census_data_race_breakdown_plot.set_xlabel("States", fontsize = 20)
census_data_race_breakdown_plot.set_ylabel("Population", fontsize = 20)


# As we can see below, within the year of 2017, the states of Texas, Louisiana, and Georgia were amongst the state leaders in terms of climate change. In relation to the spatial distribution of minority groups, we already know the insights of environmental racism within Louisiana and New Orleans as discussed within the first 2 units of the course; however, we also begin to see a pattern of large Hispanic population and large GHG emissions within those particular states, particularly when discussing Puerto Rico and Texas. In fact, according to the Environmental Defense Fund (EDF), "Climate change has a tremendous impact on the health, livelihoods, and overall well-being of U.S. Latinos ... Latinos are more vulnerable to the effects of climate change because of where we live and work. More than half (55%) of Latino-Americans live in three states that are already experiencing serious effects related to climate change: historic drought in California, record-breaking heat in Texas, and increased sea level rise and flooding in Florida. Furthermore, Latinos are more likely to work in the industries that are deeply affected by the effects of climate change such as agriculture, manufacturing, and construction." (EDF) [(Link To Article)](https://www.edf.org/sites/default/files/content/latinos_and_climate_change_factsheet_0317_refresh.pdf)

# In[13]:


data_2017 = data_formatted[data_formatted['YEAR'] == 2017]

state_data_2017 = data_2017.groupby("STATE").aggregate(np.sum)

state_data_2017_plot = state_data_2017.plot(y="GHG_QUANTITY", kind ='bar', figsize = (20, 20), legend = False, fontsize = 18)
state_data_2017_plot.set_title('Breakdown Of GHG Emissions By State In 2017', fontsize = 20)
state_data_2017_plot.set_xlabel("State", fontsize = 20)
state_data_2017_plot.set_ylabel(r"GHG Emissions ($10^8$)", fontsize = 20)
plt.plot()


# However, when we look into GHG emissions per capita in the United States in 2017, we can see that states such as California, Texas, and Indiana appear to be some of the smallest contributors to climate change with states like Wyoming, North Dakota, West Virginia, and Louisiana contributing the most to climate change when considering their population.

# In[14]:


import string
lowerCaseNames = []
for stateNames in data_2017['STATE_NAME']:
    lowerCaseNames.append(string.capwords(stateNames.lower()))
data_2017['STATE_NAME'] = lowerCaseNames
new_state_data_2017 = data_2017.groupby("STATE_NAME").aggregate(np.sum)
joined_table = new_state_data_2017.join(census_data_race_breakdown)
per_capita = [ghg/pop for ghg, pop in zip(joined_table['GHG_QUANTITY'], joined_table['TotalPop'])]
joined_table['PER_CAPITA'] = per_capita
joined_table_plot = joined_table.plot.bar(y="PER_CAPITA", figsize = (20, 20), legend = False, fontsize = 18)
joined_table_plot.set_xlabel("States", fontsize = 20)
joined_table_plot.set_ylabel("GHG Emissions Per Capita", fontsize = 20)
joined_table_plot.set_title("GHG Emissions Per Capita In 2017", fontsize = 20)
plt.plot()


# # Spatial Distribution of GHG Emissions
# 
# Below, we are going to illustrate the spatial distribution of GHG emissions within the United States. This is just done by taking the 2017 data we had above, turning it into a GeoDataFrame to get points that we can graph onto a US map visualization using Cartopy. As one can see, most of the facilities that promulgate GHG pollutants are located near the Gulf of Mexico and the Northeast. Incidentally, many Hispanic communities are located near this area such as in Texas, Mexico, Puerto Rico, etc.

# In[15]:


# define Albers Equal Area projection for contiguous US in cartopy format
proj_aea_contig_us = ccrs.AlbersEqualArea(central_longitude=-96.0,
                                          central_latitude=37.5,
                                          standard_parallels=(29.5, 45.5))

# define Albers Equal Area projection for contiguous US in proj4 format
crs_aea_contig_us = "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs"


# In[16]:


# convert pandas dataframe to geopandas geodataframe
data_2017_gpdf = gpd.GeoDataFrame(data_2017, crs=crs_aea_contig_us,
                                     geometry=gpd.points_from_xy(data_2017.LONGITUDE, data_2017.LATITUDE))


# In[17]:


fig, ax = plt.subplots(figsize=(16,6), ncols=2, subplot_kw={'projection': proj_aea_contig_us})

data_2017.plot.scatter("LONGITUDE", "LATITUDE", ax=ax[0], transform=ccrs.PlateCarree(),
                           c="firebrick", s=10, edgecolor="w")
for i in range(1):
    ax[i].set_extent([-123, -72, 20, 52])
    ax[i].stock_img()
    ax[i].coastlines()
    ax[i].add_feature(cartopy.feature.BORDERS)
    ax[i].add_feature(cartopy.feature.STATES)

ax[0].set_title("Distribution of the Facilities")


# Although the United States has been one of the leading nations when discussing GHG emissions, it looks like while we are still emitting close to 3 gigatons of GHG, we are slowly making progress to reducing our GHG emissions through policy, adaptation, and mitigation; however, while one may think this, our forecase predictions as illustrated later in this notebook say otherwise. This is just another way of visualizating the progression of GHG emissions within the 2010 to 2019 period.

# In[52]:


year_breakdown_ghg = year_breakdown[['GHG_QUANTITY']]
years = []
for year in data_formatted['YEAR']:
    if year not in years:
        years.append(year)
year_breakdown_ghg
years = sorted(years)
year_ghg_dict = {'YEAR': years, "GHG_QUANTITY": year_breakdown_ghg['GHG_QUANTITY'].tolist()}
year_breakdown_ghg = pd.DataFrame(data = year_ghg_dict)
year_breakdown_ghg_line = year_breakdown_ghg.plot.line(x = 'YEAR', y='GHG_QUANTITY', legend = False, figsize = (20, 20), fontsize = 18)
year_breakdown_ghg_line.set_xlabel('Year', fontsize = 20)
year_breakdown_ghg_line.set_ylabel('GHG Emissions (Gigatons)', fontsize = 20)
year_breakdown_ghg_line.set_title("Progression of GHG Emissions (2010-2019)", fontsize = 20)


# # Forecasting GHG Emissions
# 
# It is now time to use forecasting models such as the Autoregressive Integrated Moving Average (ARIMA) time series model to forecast GHG emissions in the future. In order to get a better picture and accuracy with this model, I am using a new dataset which contains U.S data from 1973 to 2016. This gives us more data to work with. The dataset can be found here: [Carbon Emissions Dataset](www.kaggle.com/txtrouble/carbon-emissions/data). Below, we are just loading in the data and doing some manipulation to make sure that our data is formatted properly.

# In[37]:


carbon_emissions_df = pd.read_csv("MER_T12_06.csv")
carbon_emissions_df


# In[40]:


carbon_emissions_df_yy = []
for items in carbon_emissions_df['YYYYMM']:
    carbon_emissions_df_yy.append(int(str(items)[:4]))
carbon_emissions_df_formatted = carbon_emissions_df[['YYYYMM', 'Value']]
carbon_emissions_df_formatted['YYYYMM'] = carbon_emissions_df_yy
carbon_emissions_df_formatted.rename(columns={'YYYYMM': 'Year'}, inplace = True)
values_float = []
for values in carbon_emissions_df_formatted['Value']:
    values_float.append(float(values))
carbon_emissions_df_formatted['Value'] = values_float
carbon_emissions_df_formatted


# In[43]:


carbon_emissions_by_year_df = carbon_emissions_df_formatted.groupby('Year').aggregate(np.sum)
carbon_emissions_by_year_df
new_data_years = []
for year in carbon_emissions_df_formatted['Year']:
    if year not in new_data_years:
        new_data_years.append(year)
new_data_years = sorted(new_data_years)
new_data_ghg_dict = {'Year': new_data_years, "Value": carbon_emissions_by_year_df['Value'].tolist()}
new_year_breakdown_df = pd.DataFrame(data = new_data_ghg_dict)
new_year_breakdown_df


# Below, we are using the Autoregressive Integrated Moving Average (ARIMA) time series model to forecast GHG emissions in the future. We see that while we will continue to decrease our GHG emissions for a short period, they will continue to grow substantially. While 9 Gigatons is an overestimate of GHG emissions, it is on par with other estimations if we continue to consume as we have been 

# In[58]:


from statsmodels.tsa.arima_model import ARIMA
train = new_year_breakdown_df.iloc[0:30]
test = new_year_breakdown_df.iloc[30:]
model = ARIMA(train['Value'], order=(1,0,0))
model = model.fit()
model.summary()
future_year_index = pd.date_range(start='2017', end='2050', freq = 'Y')
pred = model.predict(start = 0, end = 32)
pred.index = future_year_index
pred_line_plot = pred.plot(figsize=(20, 20), fontsize = 18)
pred_line_plot.set_xlabel('Year', fontsize = 20)
pred_line_plot.set_ylabel('GHG Emissions (Million Metric Tons)', fontsize = 20)
pred_line_plot.set_title('GHG Emissions Time Series Forecast Till 2050', fontsize = 20)


# Here, we are compiling data from Texas to prepare to make forecasting predications about GHG emissions within Texas.

# In[70]:


texas_data = data_formatted[data_formatted['STATE'] == 'TX']
texas_data_grouped['GHG_QUANTITY'] = texas_data.groupby('YEAR').aggregate(np.sum)['GHG_QUANTITY']
texas_data_year_dict = {'YEAR': [year for year in range(2010, 2020)], 'GHG_QUANTITY': texas_data_grouped['GHG_QUANTITY'].tolist()}
texas_data_by_year = pd.DataFrame(data=texas_data_year_dict)
texas_data_by_year


# Below, we are using the Autoregressive Integrated Moving Average (ARIMA) time series model to forecast GHG emissions in the future. We see that Texas GHG emissions will increase but will plateau. Although the GHG emissions are forecasted to plateau, this is still concerning for Hispanics 

# In[73]:


train = texas_data_by_year.iloc[0:7]
test = new_year_breakdown_df.iloc[7:]
model = ARIMA(train['GHG_QUANTITY'], order=(1,0,0))
model = model.fit()
model.summary()
future_year_index = pd.date_range(start='2020', end='2050', freq = 'Y')
pred = model.predict(start = 0, end = 29)
pred.index = future_year_index
pred_line_plot = pred.plot(figsize=(20, 20), fontsize = 18)
pred_line_plot.set_xlabel('Year', fontsize = 20)
pred_line_plot.set_ylabel('GHG Emissions (Million Metric Tons)', fontsize = 20)
pred_line_plot.set_title('GHG Emissions Time Series Forecast Till 2050 In Texas', fontsize = 20)


# # Conclusion
# 
# Through data analysis, we were able to conclude that while GHG emissions have been declining these past 10 years, maintaining our current consumption patterns and policies will inevitably cause GHG emissions to grow substantially.
# We were able to visualize the spatial distribution of GHG emissions and determine that most of the facilities that promulgate GHG pollutants are located near the Gulf of Mexico and the Northeast. Incidentally, many Hispanic communities are located near this area such as in Texas, Mexico, Puerto Rico, etc. This helped us begin to analyze how climate change has affected Hispanic communities. We found that according to the Environmental Defense Fund (EDF), “Climate change has a tremendous impact on the health, livelihoods, and overall well-being of U.S. Latinos ... More than half (55%) of Latino-Americans live in three states that are already experiencing serious effects related to climate change”. Some further research allowed us to notice  that Puerto Rico's climate change problems are largely synonymous to New Orleans and that higher Hispanic poverty levels and lower rates of health insurance mean that Hispanic communities are more prone to climate change effects. Through the United States Census Bureau's forecasting model along with our own, we saw that the Hispanic population within the United States is expected to grow to up to 100 million along the prediction that Texas GHG emissions will increase and plateau, but this is still concerning for Hispanics. In summation, policy, community outreach, adaptation, and mitigation strategies should continue to be adopted to help protect marginalized populations such as African Americans and Hispanics from the adverse effects of climate change

# In[ ]:




