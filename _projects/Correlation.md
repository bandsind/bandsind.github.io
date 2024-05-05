---
name: Basic Correlation
tools: [numpy, pandas, pytjon, matplotlib, collab]
image: /assets/img/Correlation/output_5_1.png
description: This project involves a basic correlation analysis using Python's libraries:NumPy, Pandas, and Matplotlib. 
---

The objective is to demonstrate how to generate a scatter plot showing a positive correlation between two datasets.

#Basic Correlation


```python
#Import Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```


```python

```


```python
#1000 random intgers between 0 and 50
x=np.random.randint(0,50,1000)
```


```python
#Positive Correlation with some noise
y_pos = x+ np.random.normal(0,10,1000)
```


```python
plt.scatter(x, y_pos)
```




    <matplotlib.collections.PathCollection at 0x7fed46685550>




    
![png](/assets/img/Correlation/output_5_1.png)
    



```python
#We can calculate r (and the p-value) using the pearsonr method
from scipy.stats import pearsonr
pearsonr(x,y_pos)
```




    (0.8162953408973396, 4.170756073597241e-240)




```python
#You can create a correlation matrix with numpy and pandas
# |x,x|x,y|
# |y,x|y,y|
np.corrcoef(x,y_pos)
```




    array([[1.        , 0.81629534],
           [0.81629534, 1.        ]])




```python
#We can do this with pandas as well
df = pd.DataFrame({'x':x, 'y': y_pos})
df.corr()
```





  <div id="df-86008216-9ad5-4d00-90e8-85296248eed5">
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
      <th>x</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>x</th>
      <td>1.000000</td>
      <td>0.816295</td>
    </tr>
    <tr>
      <th>y</th>
      <td>0.816295</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-86008216-9ad5-4d00-90e8-85296248eed5')"
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
          document.querySelector('#df-86008216-9ad5-4d00-90e8-85296248eed5 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-86008216-9ad5-4d00-90e8-85296248eed5');
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
#We can use a scatter matrix to visualize this
pd.plotting.scatter_matrix(df, alpha=0.2)
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x7fed3ec74700>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fed3ec20b50>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7fed3ebcef70>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fed3ec073a0>]],
          dtype=object)




    
![png](/assets/img/Correlation/output_9_1.png)
    



```python
y_neg = -x+np.random.normal(0,10,1000)
plt.scatter(x,y_neg)
```




    <matplotlib.collections.PathCollection at 0x7fed3eb0ad60>




    
![png](/assets/img/Correlation/output_10_1.png)
    



```python
pearsonr(x,y_neg)
```




    (-0.8371540878283641, 9.21669640172621e-264)




```python
x= np.random.randint(0,50,1000)
y=np.random.normal(0,50,1000)
```


```python
plt.scatter(x,y)
```




    <matplotlib.collections.PathCollection at 0x7fed3ea5ab20>




    
![png](/assets/img/Correlation/output_13_1.png)
    



```python
pearsonr(x,y)
```




    (-0.07794898281589098, 0.013677154592933123)



# P-Value and Statistical Inference

Let use Python to bring in some data and calculate the p-value for a given Null Hypothesis


```python
na_values = [' ','NaN','N/A']
churn = pd.read_csv('https://gitlab.com/CEADS/DrKerby/python/raw/master/churn.txt',na_values=na_values)
churn.head()
```





  <div id="df-88993f4f-f9a5-416c-9412-3af75a5f0124">
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
      <th>State</th>
      <th>Account Length</th>
      <th>Area Code</th>
      <th>Phone</th>
      <th>Int'l Plan</th>
      <th>VMail Plan</th>
      <th>VMail Message</th>
      <th>Day Mins</th>
      <th>Day Calls</th>
      <th>Day Charge</th>
      <th>...</th>
      <th>Eve Calls</th>
      <th>Eve Charge</th>
      <th>Night Mins</th>
      <th>Night Calls</th>
      <th>Night Charge</th>
      <th>Intl Mins</th>
      <th>Intl Calls</th>
      <th>Intl Charge</th>
      <th>CustServ Calls</th>
      <th>Churn?</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>KS</td>
      <td>128</td>
      <td>415</td>
      <td>382-4657</td>
      <td>no</td>
      <td>yes</td>
      <td>25</td>
      <td>265.1</td>
      <td>110</td>
      <td>45.07</td>
      <td>...</td>
      <td>99</td>
      <td>16.78</td>
      <td>244.7</td>
      <td>91</td>
      <td>11.01</td>
      <td>10.0</td>
      <td>3</td>
      <td>2.70</td>
      <td>1</td>
      <td>False.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>OH</td>
      <td>107</td>
      <td>415</td>
      <td>371-7191</td>
      <td>no</td>
      <td>yes</td>
      <td>26</td>
      <td>161.6</td>
      <td>123</td>
      <td>27.47</td>
      <td>...</td>
      <td>103</td>
      <td>16.62</td>
      <td>254.4</td>
      <td>103</td>
      <td>11.45</td>
      <td>13.7</td>
      <td>3</td>
      <td>3.70</td>
      <td>1</td>
      <td>False.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NJ</td>
      <td>137</td>
      <td>415</td>
      <td>358-1921</td>
      <td>no</td>
      <td>no</td>
      <td>0</td>
      <td>243.4</td>
      <td>114</td>
      <td>41.38</td>
      <td>...</td>
      <td>110</td>
      <td>10.30</td>
      <td>162.6</td>
      <td>104</td>
      <td>7.32</td>
      <td>12.2</td>
      <td>5</td>
      <td>3.29</td>
      <td>0</td>
      <td>False.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>OH</td>
      <td>84</td>
      <td>408</td>
      <td>375-9999</td>
      <td>yes</td>
      <td>no</td>
      <td>0</td>
      <td>299.4</td>
      <td>71</td>
      <td>50.90</td>
      <td>...</td>
      <td>88</td>
      <td>5.26</td>
      <td>196.9</td>
      <td>89</td>
      <td>8.86</td>
      <td>6.6</td>
      <td>7</td>
      <td>1.78</td>
      <td>2</td>
      <td>False.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>OK</td>
      <td>75</td>
      <td>415</td>
      <td>330-6626</td>
      <td>yes</td>
      <td>no</td>
      <td>0</td>
      <td>166.7</td>
      <td>113</td>
      <td>28.34</td>
      <td>...</td>
      <td>122</td>
      <td>12.61</td>
      <td>186.9</td>
      <td>121</td>
      <td>8.41</td>
      <td>10.1</td>
      <td>3</td>
      <td>2.73</td>
      <td>3</td>
      <td>False.</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-88993f4f-f9a5-416c-9412-3af75a5f0124')"
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
          document.querySelector('#df-88993f4f-f9a5-416c-9412-3af75a5f0124 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-88993f4f-f9a5-416c-9412-3af75a5f0124');
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
churn['Eve Charge'].hist(bins=20)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fed3ebc6970>




    
![png](/assets/img/Correlation/output_18_1.png)
    



```python
#For this purposes of this lecture this churn dataset will be our population
#Let take a quick look at the average value of this data
mu = churn['Eve Charge'].mean()
mu
```




    17.083540354035403



So based on this mean, our null hypothesis will be that mu will be the average for a sample of the data. Our alternate hypothesis is that mu will not be the average value (17.08)


```python
#Take a sample of 100 points from the dataset
sample = churn.sample(n=100, replace=True)#Ensure this is done randomly
sample.head()
```





  <div id="df-71ab6aec-3928-43d7-b054-2d4ae0c9f132">
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
      <th>State</th>
      <th>Account Length</th>
      <th>Area Code</th>
      <th>Phone</th>
      <th>Int'l Plan</th>
      <th>VMail Plan</th>
      <th>VMail Message</th>
      <th>Day Mins</th>
      <th>Day Calls</th>
      <th>Day Charge</th>
      <th>...</th>
      <th>Eve Calls</th>
      <th>Eve Charge</th>
      <th>Night Mins</th>
      <th>Night Calls</th>
      <th>Night Charge</th>
      <th>Intl Mins</th>
      <th>Intl Calls</th>
      <th>Intl Charge</th>
      <th>CustServ Calls</th>
      <th>Churn?</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2374</th>
      <td>WY</td>
      <td>157</td>
      <td>415</td>
      <td>348-9938</td>
      <td>yes</td>
      <td>no</td>
      <td>0</td>
      <td>180.4</td>
      <td>123</td>
      <td>30.67</td>
      <td>...</td>
      <td>98</td>
      <td>16.49</td>
      <td>227.3</td>
      <td>88</td>
      <td>10.23</td>
      <td>8.4</td>
      <td>5</td>
      <td>2.27</td>
      <td>0</td>
      <td>False.</td>
    </tr>
    <tr>
      <th>3177</th>
      <td>NM</td>
      <td>84</td>
      <td>408</td>
      <td>419-9713</td>
      <td>no</td>
      <td>yes</td>
      <td>41</td>
      <td>153.9</td>
      <td>102</td>
      <td>26.16</td>
      <td>...</td>
      <td>117</td>
      <td>11.96</td>
      <td>217.7</td>
      <td>101</td>
      <td>9.80</td>
      <td>12.8</td>
      <td>5</td>
      <td>3.46</td>
      <td>1</td>
      <td>False.</td>
    </tr>
    <tr>
      <th>1193</th>
      <td>NM</td>
      <td>119</td>
      <td>415</td>
      <td>352-5118</td>
      <td>yes</td>
      <td>yes</td>
      <td>15</td>
      <td>160.0</td>
      <td>95</td>
      <td>27.20</td>
      <td>...</td>
      <td>110</td>
      <td>17.81</td>
      <td>82.3</td>
      <td>107</td>
      <td>3.70</td>
      <td>8.7</td>
      <td>5</td>
      <td>2.35</td>
      <td>5</td>
      <td>True.</td>
    </tr>
    <tr>
      <th>666</th>
      <td>OR</td>
      <td>120</td>
      <td>415</td>
      <td>368-8283</td>
      <td>no</td>
      <td>no</td>
      <td>0</td>
      <td>252.0</td>
      <td>120</td>
      <td>42.84</td>
      <td>...</td>
      <td>106</td>
      <td>12.77</td>
      <td>151.8</td>
      <td>96</td>
      <td>6.83</td>
      <td>9.6</td>
      <td>1</td>
      <td>2.59</td>
      <td>2</td>
      <td>False.</td>
    </tr>
    <tr>
      <th>2005</th>
      <td>NY</td>
      <td>119</td>
      <td>415</td>
      <td>343-1458</td>
      <td>no</td>
      <td>no</td>
      <td>0</td>
      <td>133.4</td>
      <td>102</td>
      <td>22.68</td>
      <td>...</td>
      <td>71</td>
      <td>17.39</td>
      <td>196.9</td>
      <td>103</td>
      <td>8.86</td>
      <td>11.1</td>
      <td>7</td>
      <td>3.00</td>
      <td>1</td>
      <td>False.</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-71ab6aec-3928-43d7-b054-2d4ae0c9f132')"
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
          document.querySelector('#df-71ab6aec-3928-43d7-b054-2d4ae0c9f132 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-71ab6aec-3928-43d7-b054-2d4ae0c9f132');
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
#Check the mean of our sample
x_bar = sample['Eve Charge'].mean()
x_bar

```




    17.1298




```python
#The means are fairly close
#Check the distribution of the data
sample['Eve Charge'].hist(bins=20)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fed3de13df0>




    
![png](/assets/img/Correlation/output_23_1.png)
    

The behavior of this data is a bit a bit different than our entire dataset. Let's take the mean of several samples of our data. We will also calculate the t-value for our dataset.


```python
#A t-test can only be done when comparing two groups of data.
#The formula we will use to calculate our t-value is t = (x_bar-mu)/(s/sqrt(n))
x_bar=[]
t_value=[]
for i in range(3000):
  sample =churn.sample(n=100, replace=True)
  x_bar.append(sample['Eve Charge'].mean())
  t_value.append((sample['Eve Charge'].mean()-mu)/(sample['Eve Charge'].std()/10))
```


```python
#Visualize x_bar
plt.hist(x_bar, bins=20)
```




    (array([  2.,  11.,  27.,  45.,  79., 121., 191., 284., 337., 385., 365.,
            340., 289., 229., 146.,  75.,  47.,  15.,   9.,   3.]),
     array([15.6722  , 15.812005, 15.95181 , 16.091615, 16.23142 , 16.371225,
            16.51103 , 16.650835, 16.79064 , 16.930445, 17.07025 , 17.210055,
            17.34986 , 17.489665, 17.62947 , 17.769275, 17.90908 , 18.048885,
            18.18869 , 18.328495, 18.4683  ]),
     <a list of 20 Patch objects>)




    
![png](/assets/img/Correlation/output_26_1.png)
    



```python
#Visualize the t-value
plt.hist(t_value, bins=20)
```




    (array([  3.,  16.,  25.,  53.,  92., 117., 201., 301., 350., 386., 351.,
            326., 300., 203., 139.,  69.,  42.,  16.,   7.,   3.]),
     array([-3.24403114, -2.91800978, -2.59198842, -2.26596706, -1.93994571,
            -1.61392435, -1.28790299, -0.96188163, -0.63586028, -0.30983892,
             0.01618244,  0.3422038 ,  0.66822516,  0.99424651,  1.32026787,
             1.64628923,  1.97231059,  2.29833194,  2.6243533 ,  2.95037466,
             3.27639602]),
     <a list of 20 Patch objects>)




    
![png](/assets/img/Correlation/output_27_1.png)
    


Now Scipy has Point Probablity Function which we can use to calculate the critical value. This pretty is close to our test statistic formula.


```python
#A 95% Confidence interval (a=5%) is going to take the value of t
#at either the +/- 2.5% tails. We will prove of disprove that 95% of the
#distrubution is between +/- t-critical value
from scipy.stats import t
p=0.975 #One sided => 95% CI
df = 99 #n-1
t_critical = t.ppf(p,df)
t_critical
```




    1.9842169515086827



Let's compare to a t table.

Looking at the table we see that the p-value is 0.05 (two tailed). If it was single tailed it would just be half of this. Since this is not less than our CI we cannot reject the null hypothesis.  

P-Value is very conisideration when performing statistical analysis. The U.S Census uses the following rules for p-values:


1.   P-Values below 0.01 consitutue strong evidence against our null.
2.   P-Values between 0.01 and 0.05 consititute moderate evidence against our null.
3. P-Values between 0.05 and 0.1 consitutue weak evidence against our null.


