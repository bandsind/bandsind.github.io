---
name: MNIST_digits dataset
tools: [numpy, pandas, pytjon, matplotlib, collab]
image: /assets/img/GroupBy/output_10_0.png
description: Practicing with GroupBy using the Covid Dataset
---




# Practice with GroupBy

```python
#Importing packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

```


```python
#Giving permission to the drive.
from google.colab import drive
drive.mount('/content/gdrive')
```

    Drive already mounted at /content/gdrive; to attempt to forcibly remount, call drive.mount("/content/gdrive", force_remount=True).



```python
#Creating a dataframe
covid_data = pd.read_csv('/content/United_States_COVID-19_Community_Levels_by_County.csv')
covid_data
```





  <div id="df-88a9dc51-450b-4ce4-862b-b985b3e1feb8">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>county</th>
      <th>county_fips</th>
      <th>state</th>
      <th>county_population</th>
      <th>health_service_area_number</th>
      <th>health_service_area</th>
      <th>health_service_area_population</th>
      <th>covid_inpatient_bed_utilization</th>
      <th>covid_hospital_admissions_per_100k</th>
      <th>covid_cases_per_100k</th>
      <th>covid-19_community_level</th>
      <th>date_updated</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Lincoln County</td>
      <td>55069</td>
      <td>Wisconsin</td>
      <td>27593.0</td>
      <td>282</td>
      <td>Marathon (Wausau), WI - Wood, WI</td>
      <td>291401.0</td>
      <td>4.7</td>
      <td>13.4</td>
      <td>177.58</td>
      <td>Medium</td>
      <td>2022-08-18</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Manitowoc County</td>
      <td>55071</td>
      <td>Wisconsin</td>
      <td>78981.0</td>
      <td>355</td>
      <td>Sheboygan (Sheboygan), WI - Manitowoc, WI</td>
      <td>244410.0</td>
      <td>3.4</td>
      <td>9.8</td>
      <td>169.66</td>
      <td>Low</td>
      <td>2022-08-18</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Marathon County</td>
      <td>55073</td>
      <td>Wisconsin</td>
      <td>135692.0</td>
      <td>282</td>
      <td>Marathon (Wausau), WI - Wood, WI</td>
      <td>291401.0</td>
      <td>4.7</td>
      <td>13.4</td>
      <td>209.30</td>
      <td>High</td>
      <td>2022-08-18</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Monroe County</td>
      <td>55081</td>
      <td>Wisconsin</td>
      <td>46253.0</td>
      <td>290</td>
      <td>La Crosse (La Crosse), WI - Monroe, WI</td>
      <td>257027.0</td>
      <td>3.9</td>
      <td>15.6</td>
      <td>216.20</td>
      <td>High</td>
      <td>2022-08-18</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Portage County</td>
      <td>55097</td>
      <td>Wisconsin</td>
      <td>70772.0</td>
      <td>400</td>
      <td>Portage, WI</td>
      <td>70772.0</td>
      <td>5.9</td>
      <td>7.1</td>
      <td>217.60</td>
      <td>Medium</td>
      <td>2022-08-18</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>112831</th>
      <td>Peñuelas Muni</td>
      <td>72111</td>
      <td>Puerto Rico</td>
      <td>19249.0</td>
      <td>904</td>
      <td>Puerto Rico</td>
      <td>3193694.0</td>
      <td>1.4</td>
      <td>2.9</td>
      <td>202.61</td>
      <td>Medium</td>
      <td>2022-10-20</td>
    </tr>
    <tr>
      <th>112832</th>
      <td>San Lorenzo Muni</td>
      <td>72129</td>
      <td>Puerto Rico</td>
      <td>35989.0</td>
      <td>904</td>
      <td>Puerto Rico</td>
      <td>3193694.0</td>
      <td>1.4</td>
      <td>2.9</td>
      <td>119.48</td>
      <td>Low</td>
      <td>2022-10-20</td>
    </tr>
    <tr>
      <th>112833</th>
      <td>Santa Isabel Muni</td>
      <td>72133</td>
      <td>Puerto Rico</td>
      <td>21209.0</td>
      <td>904</td>
      <td>Puerto Rico</td>
      <td>3193694.0</td>
      <td>1.4</td>
      <td>2.9</td>
      <td>165.02</td>
      <td>Low</td>
      <td>2022-10-20</td>
    </tr>
    <tr>
      <th>112834</th>
      <td>Utuado Muni</td>
      <td>72141</td>
      <td>Puerto Rico</td>
      <td>27395.0</td>
      <td>904</td>
      <td>Puerto Rico</td>
      <td>3193694.0</td>
      <td>1.4</td>
      <td>2.9</td>
      <td>233.62</td>
      <td>Medium</td>
      <td>2022-10-20</td>
    </tr>
    <tr>
      <th>112835</th>
      <td>Vega Baja Muni</td>
      <td>72145</td>
      <td>Puerto Rico</td>
      <td>50023.0</td>
      <td>904</td>
      <td>Puerto Rico</td>
      <td>3193694.0</td>
      <td>1.4</td>
      <td>2.9</td>
      <td>231.89</td>
      <td>Medium</td>
      <td>2022-10-20</td>
    </tr>
  </tbody>
</table>
<p>112836 rows × 12 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-88a9dc51-450b-4ce4-862b-b985b3e1feb8')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-88a9dc51-450b-4ce4-862b-b985b3e1feb8 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-88a9dc51-450b-4ce4-862b-b985b3e1feb8');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




# **Case 1**
Use .grouby on state to create a subset of this dataframe only containing entries from Washington. Then create another dataframe only containing entries from Florida.


```python
#dataframe containtaining only entries from Washington
washington_data = covid_data.groupby("state").get_group('Washington')
washington_data
```





  <div id="df-eb5e3502-c63e-46d5-850a-4935b3786298">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>county</th>
      <th>county_fips</th>
      <th>state</th>
      <th>county_population</th>
      <th>health_service_area_number</th>
      <th>health_service_area</th>
      <th>health_service_area_population</th>
      <th>covid_inpatient_bed_utilization</th>
      <th>covid_hospital_admissions_per_100k</th>
      <th>covid_cases_per_100k</th>
      <th>covid-19_community_level</th>
      <th>date_updated</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>766</th>
      <td>Adams County</td>
      <td>53001</td>
      <td>Washington</td>
      <td>19983.0</td>
      <td>698</td>
      <td>Spokane (Spokane), WA - Stevens, WA</td>
      <td>620794.0</td>
      <td>4.2</td>
      <td>8.9</td>
      <td>155.13</td>
      <td>Low</td>
      <td>2022-08-18</td>
    </tr>
    <tr>
      <th>767</th>
      <td>Ferry County</td>
      <td>53019</td>
      <td>Washington</td>
      <td>7627.0</td>
      <td>698</td>
      <td>Spokane (Spokane), WA - Stevens, WA</td>
      <td>620794.0</td>
      <td>4.2</td>
      <td>8.9</td>
      <td>196.67</td>
      <td>Low</td>
      <td>2022-08-18</td>
    </tr>
    <tr>
      <th>768</th>
      <td>Kittitas County</td>
      <td>53037</td>
      <td>Washington</td>
      <td>47935.0</td>
      <td>739</td>
      <td>Yakima (Yakima), WA - Kittitas, WA</td>
      <td>298808.0</td>
      <td>4.1</td>
      <td>3.7</td>
      <td>31.29</td>
      <td>Low</td>
      <td>2022-08-18</td>
    </tr>
    <tr>
      <th>769</th>
      <td>San Juan County</td>
      <td>53055</td>
      <td>Washington</td>
      <td>17582.0</td>
      <td>736</td>
      <td>King (Seattle), WA - Snohomish, WA</td>
      <td>3578266.0</td>
      <td>5.7</td>
      <td>6.2</td>
      <td>51.19</td>
      <td>Low</td>
      <td>2022-08-18</td>
    </tr>
    <tr>
      <th>770</th>
      <td>Snohomish County</td>
      <td>53061</td>
      <td>Washington</td>
      <td>822083.0</td>
      <td>736</td>
      <td>King (Seattle), WA - Snohomish, WA</td>
      <td>3578266.0</td>
      <td>5.7</td>
      <td>6.2</td>
      <td>151.93</td>
      <td>Low</td>
      <td>2022-08-18</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>112770</th>
      <td>Kittitas County</td>
      <td>53037</td>
      <td>Washington</td>
      <td>47935.0</td>
      <td>739</td>
      <td>Yakima (Yakima), WA - Kittitas, WA</td>
      <td>298808.0</td>
      <td>1.4</td>
      <td>1.3</td>
      <td>58.41</td>
      <td>Low</td>
      <td>2022-10-20</td>
    </tr>
    <tr>
      <th>112771</th>
      <td>San Juan County</td>
      <td>53055</td>
      <td>Washington</td>
      <td>17582.0</td>
      <td>736</td>
      <td>King (Seattle), WA - Snohomish, WA</td>
      <td>3578266.0</td>
      <td>4.0</td>
      <td>3.5</td>
      <td>28.44</td>
      <td>Low</td>
      <td>2022-10-20</td>
    </tr>
    <tr>
      <th>112772</th>
      <td>Snohomish County</td>
      <td>53061</td>
      <td>Washington</td>
      <td>822083.0</td>
      <td>736</td>
      <td>King (Seattle), WA - Snohomish, WA</td>
      <td>3578266.0</td>
      <td>4.0</td>
      <td>3.5</td>
      <td>46.47</td>
      <td>Low</td>
      <td>2022-10-20</td>
    </tr>
    <tr>
      <th>112773</th>
      <td>Whatcom County</td>
      <td>53073</td>
      <td>Washington</td>
      <td>229247.0</td>
      <td>815</td>
      <td>Whatcom, WA</td>
      <td>229247.0</td>
      <td>3.7</td>
      <td>6.5</td>
      <td>36.21</td>
      <td>Low</td>
      <td>2022-10-20</td>
    </tr>
    <tr>
      <th>112774</th>
      <td>Whitman County</td>
      <td>53075</td>
      <td>Washington</td>
      <td>50104.0</td>
      <td>784</td>
      <td>Whitman, WA - Latah, ID</td>
      <td>90212.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>9.98</td>
      <td>Low</td>
      <td>2022-10-20</td>
    </tr>
  </tbody>
</table>
<p>1365 rows × 12 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-eb5e3502-c63e-46d5-850a-4935b3786298')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-eb5e3502-c63e-46d5-850a-4935b3786298 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-eb5e3502-c63e-46d5-850a-4935b3786298');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
#dataframe containtaining only entries from Florida
florida_data = covid_data.groupby("state").get_group('Florida')
florida_data
```





  <div id="df-69fe76b0-470c-4e51-b0b7-41c30e37c1d2">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>county</th>
      <th>county_fips</th>
      <th>state</th>
      <th>county_population</th>
      <th>health_service_area_number</th>
      <th>health_service_area</th>
      <th>health_service_area_population</th>
      <th>covid_inpatient_bed_utilization</th>
      <th>covid_hospital_admissions_per_100k</th>
      <th>covid_cases_per_100k</th>
      <th>covid-19_community_level</th>
      <th>date_updated</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>115</th>
      <td>Alachua County</td>
      <td>12001</td>
      <td>Florida</td>
      <td>269043.0</td>
      <td>159</td>
      <td>Alachua (Gainesville), FL - Columbia, FL</td>
      <td>528345.0</td>
      <td>7.9</td>
      <td>33.7</td>
      <td>193.28</td>
      <td>High</td>
      <td>2022-08-18</td>
    </tr>
    <tr>
      <th>116</th>
      <td>Bradford County</td>
      <td>12007</td>
      <td>Florida</td>
      <td>28201.0</td>
      <td>159</td>
      <td>Alachua (Gainesville), FL - Columbia, FL</td>
      <td>528345.0</td>
      <td>7.9</td>
      <td>33.7</td>
      <td>195.03</td>
      <td>High</td>
      <td>2022-08-18</td>
    </tr>
    <tr>
      <th>117</th>
      <td>Citrus County</td>
      <td>12017</td>
      <td>Florida</td>
      <td>149657.0</td>
      <td>233</td>
      <td>Marion (Ocala), FL - Citrus, FL</td>
      <td>515236.0</td>
      <td>7.7</td>
      <td>24.3</td>
      <td>201.13</td>
      <td>High</td>
      <td>2022-08-18</td>
    </tr>
    <tr>
      <th>118</th>
      <td>Clay County</td>
      <td>12019</td>
      <td>Florida</td>
      <td>219252.0</td>
      <td>158</td>
      <td>Duval (Jacksonville), FL - Clay, FL</td>
      <td>1481679.0</td>
      <td>6.5</td>
      <td>19.5</td>
      <td>220.75</td>
      <td>High</td>
      <td>2022-08-18</td>
    </tr>
    <tr>
      <th>119</th>
      <td>DeSoto County</td>
      <td>12027</td>
      <td>Florida</td>
      <td>38001.0</td>
      <td>213</td>
      <td>Sarasota (Sarasota), FL - Charlotte, FL</td>
      <td>660653.0</td>
      <td>8.9</td>
      <td>31.9</td>
      <td>160.52</td>
      <td>High</td>
      <td>2022-08-18</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>112140</th>
      <td>Pasco County</td>
      <td>12101</td>
      <td>Florida</td>
      <td>553947.0</td>
      <td>227</td>
      <td>Hillsborough (Tampa), FL - Pinellas, FL</td>
      <td>3194831.0</td>
      <td>2.3</td>
      <td>7.5</td>
      <td>35.74</td>
      <td>Low</td>
      <td>2022-10-20</td>
    </tr>
    <tr>
      <th>112141</th>
      <td>Pinellas County</td>
      <td>12103</td>
      <td>Florida</td>
      <td>974996.0</td>
      <td>227</td>
      <td>Hillsborough (Tampa), FL - Pinellas, FL</td>
      <td>3194831.0</td>
      <td>2.3</td>
      <td>7.5</td>
      <td>46.46</td>
      <td>Low</td>
      <td>2022-10-20</td>
    </tr>
    <tr>
      <th>112142</th>
      <td>Seminole County</td>
      <td>12117</td>
      <td>Florida</td>
      <td>471826.0</td>
      <td>142</td>
      <td>Orange (Orlando), FL - Volusia, FL</td>
      <td>3033181.0</td>
      <td>2.8</td>
      <td>4.4</td>
      <td>32.64</td>
      <td>Low</td>
      <td>2022-10-20</td>
    </tr>
    <tr>
      <th>112143</th>
      <td>Suwannee County</td>
      <td>12121</td>
      <td>Florida</td>
      <td>44417.0</td>
      <td>159</td>
      <td>Alachua (Gainesville), FL - Columbia, FL</td>
      <td>528345.0</td>
      <td>2.4</td>
      <td>12.5</td>
      <td>47.28</td>
      <td>Medium</td>
      <td>2022-10-20</td>
    </tr>
    <tr>
      <th>112144</th>
      <td>Washington County</td>
      <td>12133</td>
      <td>Florida</td>
      <td>25473.0</td>
      <td>155</td>
      <td>Bay (Panama City), FL - Jackson, FL</td>
      <td>293953.0</td>
      <td>1.0</td>
      <td>3.4</td>
      <td>19.63</td>
      <td>Low</td>
      <td>2022-10-20</td>
    </tr>
  </tbody>
</table>
<p>2345 rows × 12 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-69fe76b0-470c-4e51-b0b7-41c30e37c1d2')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-69fe76b0-470c-4e51-b0b7-41c30e37c1d2 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-69fe76b0-470c-4e51-b0b7-41c30e37c1d2');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




# ** Case 2**
Using whichever approach you prefer, calculate the average covid hospital admissions per 100k for each state. Then calculate the standard deviations for each state. Finally, create a single histogram containing the counts from each state. Below, in 3-4 sentences, explain your observations.


```python
florida_avarage = florida_data['covid_hospital_admissions_per_100k'].mean()
florida_std = florida_data['covid_hospital_admissions_per_100k'].std()
print("mean for Florida: " + str(round(florida_avarage, 6)))
print("Standard deviation for Florida is: "+str(round(florida_std, 6)))

print("\n")

washington_average = washington_data['covid_hospital_admissions_per_100k'].mean()
washington_std = washington_data['covid_hospital_admissions_per_100k'].std()
print("mean for Washington: " + str(round(washington_average, 6)))
print("Standard deviation for Washington is: " + str(round(washington_std, 6)))
```

    mean for Florida: 12.335991
    Standard deviation for Florida is: 9.260087
    
    
    mean for Washington: 6.367399
    Standard deviation for Washington is: 3.50032



```python
plt.hist(florida_data['covid_hospital_admissions_per_100k'], label="Florida",  bins=10)
plt.hist(washington_data['covid_hospital_admissions_per_100k'],label ="Washington",  bins=10)
plt.ylabel("Number of Counts")
plt.xlabel("covid_hospital_admissions_per_100k")
plt.title("Counts vs covid hospital admissions per 100k")
plt.legend(loc="upper right")
plt.show()

```


    
![png](/assets/img/GroupBy/output_10_0.png)
    


#Observation


*   both Florida and Washington histogram are skewed to the right (positivily skewed).
*   There are more covid hospital admissions in Florida than in Washington.
*   The mean will be will slightly larger than the median for Florida.



# **Case 3**
Use .groupby again to determine average number of covid cases per 100k in each county for Florida during this time period. Which had the highest? The lowest? Use a bar graph to plot the top 20 counties.  Repeat this for Washington. Do a quick internet serach on the top county for each state. Is there anything based on this information that may explain why both of these counties had the highest rate?



```python
#Florida State
county_df = florida_data.groupby("county")
for county,group in county_df:
  print("\n")
  print(county)
  print(group['covid_cases_per_100k'].mean())



print("\n")
print("County with Highest mean: " + str(county_df['covid_cases_per_100k'].mean().idxmax()))
print("County Lowest mean: " + str(county_df['covid_cases_per_100k'].mean().idxmin()))


```

    
    
    Alachua County
    215.37749999999997
    
    
    Baker County
    141.07500000000002
    
    
    Bay County
    138.71083333333334
    
    
    Bradford County
    148.19291666666666
    
    
    Brevard County
    191.96124999999998
    
    
    Broward County
    250.07666666666668
    
    
    Calhoun County
    138.54416666666665
    
    
    Charlotte County
    182.56083333333336
    
    
    Citrus County
    169.41375
    
    
    Clay County
    156.38458333333332
    
    
    Collier County
    143.79125000000002
    
    
    Columbia County
    141.18166666666667
    
    
    DeSoto County
    137.05791666666667
    
    
    Dixie County
    138.42625
    
    
    Duval County
    179.76
    
    
    Escambia County
    136.32958333333332
    
    
    Flagler County
    153.84
    
    
    Franklin County
    90.37875000000001
    
    
    Gadsden County
    230.05124999999998
    
    
    Gilchrist County
    163.68791666666667
    
    
    Glades County
    58.5275
    
    
    Gulf County
    141.44416666666666
    
    
    Hamilton County
    130.8225
    
    
    Hardee County
    150.35083333333333
    
    
    Hendry County
    124.935
    
    
    Hernando County
    167.57291666666666
    
    
    Highlands County
    165.73166666666668
    
    
    Hillsborough County
    198.1954166666667
    
    
    Holmes County
    121.91833333333334
    
    
    Indian River County
    179.59124999999997
    
    
    Jackson County
    166.70583333333332
    
    
    Jefferson County
    196.54749999999999
    
    
    Lafayette County
    108.34583333333335
    
    
    Lake County
    177.08833333333328
    
    
    Lee County
    180.19416666666666
    
    
    Leon County
    259.43958333333336
    
    
    Levy County
    152.09750000000003
    
    
    Liberty County
    137.65833333333333
    
    
    Madison County
    173.48791666666668
    
    
    Manatee County
    189.75833333333333
    
    
    Marion County
    150.275
    
    
    Martin County
    156.65
    
    
    Miami-Dade County
    378.3233333333333
    
    
    Monroe County
    176.76541666666665
    
    
    Nassau County
    166.76125000000002
    
    
    Okaloosa County
    130.0
    
    
    Okeechobee County
    108.495
    
    
    Orange County
    207.59916666666666
    
    
    Osceola County
    230.25083333333336
    
    
    Palm Beach County
    205.16958333333332
    
    
    Pasco County
    187.11208333333335
    
    
    Pinellas County
    190.93624999999997
    
    
    Polk County
    219.59624999999997
    
    
    Putnam County
    126.19458333333334
    
    
    Santa Rosa County
    152.11958333333334
    
    
    Sarasota County
    200.14791666666667
    
    
    Seminole County
    169.87125
    
    
    St. Johns County
    155.46
    
    
    St. Lucie County
    182.29250000000002
    
    
    Sumter County
    163.715
    
    
    Suwannee County
    142.49416666666664
    
    
    Taylor County
    121.50916666666667
    
    
    Union County
    115.945
    
    
    Volusia County
    167.61291666666668
    
    
    Wakulla County
    162.64541666666665
    
    
    Walton County
    108.51125000000002
    
    
    Washington County
    101.57875
    
    
    County with Highest mean: Miami-Dade County
    County Lowest mean: Glades County



```python
top_20_counties  = county_df['covid_cases_per_100k'].mean().nlargest(20)
top_20_counties.plot(kind='bar', xlabel='County', ylabel='Avgerage covid cases per 100k', title='Top 20 Counties by Average COVID Cases per 100k in Florida')
plt.show()
```


    
![png](/assets/img/GroupBy/output_14_0.png)
    



```python
#Florida State
county_wash = washington_data.groupby("county")
for county_2,group in county_df:
  print("\n")
  print(county_2)
  print(group['covid_cases_per_100k'].mean())



print("\n")
print("County with Highest mean: " + str(county_wash['covid_cases_per_100k'].mean().idxmax()))
print("County Lowest mean: " + str(county_wash['covid_cases_per_100k'].mean().idxmin()))


top_20_counties  = county_wash['covid_cases_per_100k'].mean().nlargest(20)
top_20_counties.plot(kind='bar', xlabel='County ', ylabel='Avgerage covid cases per 100k', title='Top 20 Counties by Average COVID Cases per 100k in Florida')
plt.show()
```

#Question 3 Observtion
Miami-Dade County and king County are one of the most populated counties in the united states and they are the higest conties in the their states. This may have to led to alot of human interaction. Their averages are likely to be higher due to the higher population compared to the other counties.

# ** Case 4**

Using 1 of the 3 numerical COVID fields (covid_inpatient_bed_utilization 	covid_hospital_admissions_per_100k 	covid_cases_per_100k) for Florida. Plot a scatterplot of that feature vs.county_population. Then using Scipy, calculate the correlation coefficent for these two varaibles. Repeat for Washington dataset. In 1 to 2 paragraphs, analyze the results. Is there any correlation between the two varialbes, what kind? To what magnitude? Is this surprising or expected?





```python

#Washington scatter plot
x = washington_data["county_population"]
y = washington_data["covid_inpatient_bed_utilization"]

plt.scatter(x, y)
plt.xlabel("County Population")
plt.ylabel("Covid inpatient bed untilization")
plt.title("Covid inpatient bed utilization vs County Population")
plt.show()

```


    
![png](/assets/img/GroupBy/output_19_0.png)
    



```python
#Using the pearson method to calculate r
from scipy.stats import pearsonr
pearsonr(x, y)
```




    (0.09972930042880182, 1.3055724815058924e-06)






```python
#Florida scatter plot
x = florida_data["county_population"]
y = florida_data["covid_inpatient_bed_utilization"]

plt.scatter(x, y)
plt.xlabel("County Population")
plt.ylabel("Covid inpatient bed untilization")
plt.title("Covid inpatient bed utilization vs County Population")
plt.show()


```


    
![png](/assets/img/GroupBy/output_22_0.png)
    



```python
#Using the pearson method to calculate r
from scipy.stats import pearsonr
pearsonr(x, y)
```




    (0.12624890389766488, 3.7950156590012967e-07)



In both graphs,there is no relationship between county population and the covid inpatient bed utilization. There could be other factors would influence the above results such as the number of beds or the size the hospital. However, this the number of inpatient bed utilization may have been as a result of causation. This is because the higher the population will likely have higher inpatient bed utilization if we are in pandemic crisis.  



# Case 5
Using the Florida Dataset calculate the average covid_cases_per_100k for each of the covid-19_community_level catergories (High, Medium & Low). Plot the average value for each category using a bar plot. Repeat for Washinton and summarize your findings.

*Hint: If we explore the dataset we will notice that the covid level was entered in both upper and lower case (High and high). Use your Python and/or Pandas knowledge to fix this.*


```python
washington_data['covid-19_community_level'] = washington_data.loc[:, 'covid-19_community_level'].str.lower()

#calculating the average covid_cases_per_100k for each of the covid-19_community_level catergories
#Groupby one column and return the mean of only particular column in the group.
averagewashington = washington_data.groupby('covid-19_community_level')['covid_cases_per_100k'].mean()
print(averagewashington)

#bar plot
averagewashington.plot(kind= "bar", title = "Average covid case for each community level category")
plt.show()
```

    covid-19_community_level
    high      250.427191
    low        84.987487
    medium    191.890469
    Name: covid_cases_per_100k, dtype: float64


    <ipython-input-102-a9de4105af9c>:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      washington_data['covid-19_community_level'] = washington_data.loc[:, 'covid-19_community_level'].str.lower()



    
![png](/assets/img/GroupBy/output_26_2.png)
    



```python
florida_data['covid-19_community_level'] = florida_data.loc[:, 'covid-19_community_level'].str.lower()

#calculating the average covid_cases_per_100k for each of the covid-19_community_level catergories
#Groupby one column and return the mean of only particular column in the group.
averageFlorida = florida_data.groupby('covid-19_community_level')['covid_cases_per_100k'].mean()
print(averageFlorida)

#bar plot
averageFlorida.plot(kind= "bar", title = "Average covid case for each community level category")
plt.show()
"""
check = florida_data['covid-19_community_level']
for i in check:
  if i == "Low":
    print(i)

  elif i == "High":

    print(i)
  elif i == "Medium":

    print(i)
"""
```

    covid-19_community_level
    high      280.663794
    low        59.300614
    medium    136.698816
    Name: covid_cases_per_100k, dtype: float64


    <ipython-input-26-9804c4d3d799>:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      florida_data['covid-19_community_level'] = florida_data.loc[:, 'covid-19_community_level'].str.lower()



    
![png](/assets/img/GroupBy/output_27_2.png)
    





    '\ncheck = florida_data[\'covid-19_community_level\']\nfor i in check:\n  if i == "Low":\n    print(i)\n  \n  elif i == "High":\n\n    print(i)\n  elif i == "Medium":\n\n    print(i)\n'


