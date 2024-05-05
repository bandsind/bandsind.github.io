---
name: Basic Correlation
tools: [numpy, pandas, pytjon, matplotlib, collab]
image: /assets/img/pandas.png
description: Exploring pandas basics 
---



# Pandas DataFrames
Dr. Leslie Kerby </br>
Data Science and Applied Machine Learning </br>
Student's name:


```python
import numpy as np
import pandas as pd
```

## 1. World Population

**Case 1.1**<br/>
Read in the 'world_population.csv' file. Adjust your DataFrame to keep only the 'Year' and 'Population' columns. Show the first 5 elements with the `head` method.


```python
world_pop = pd.read_csv('https://raw.githubusercontent.com/LGKerby/Python/master/world_population.csv',usecols=['Year','Population'])
# using the head mehtod to show the first 5 elements for the Year and Population Column.
world_pop.head()

```





  <div id="df-4c2f66e0-eb70-449b-a7f6-5fe924688f83">
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
      <th>Year</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1950</td>
      <td>2557628654</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1951</td>
      <td>2594939877</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1952</td>
      <td>2636772306</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1953</td>
      <td>2682053389</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1954</td>
      <td>2730228104</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-4c2f66e0-eb70-449b-a7f6-5fe924688f83')"
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
          document.querySelector('#df-4c2f66e0-eb70-449b-a7f6-5fe924688f83 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-4c2f66e0-eb70-449b-a7f6-5fe924688f83');
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




**Case 1.2**<br/>
How many rows (years) are in the DataFrame? Use python code to give the answer.


```python
"""
The shape property returns a tuple containing the shape of the DataFrame.
The shape is the number of rows and columns of the DataFrame
I used the 0 index to show the number of rows in the dataframe
"""
rowCount = world_pop.shape[0]
print("The number of rows is: " + str(rowCount))
```

    The number of rows is: 66


**Case 1.3**<br/>
What was the population of the world in 1976? Use python code to give the answer.


```python
pop = world_pop[world_pop["Year"] == 1976]
print("The population of the world in 1976 is: " + str(pop.iloc[0]["Population"]))

```

    The population of the world in 1976 is: 4160185010


**Case 1.4**<br/>
Use a boolean mask to create a DataFrame called `last_quartile` that contains the world population from years 1975 to 1999. You may do this in two steps (use two boolean masks) or you may do this in one step using the 'and' operator `&`.

*Hint:* For example, to select the years 1960 and 1970 from the DataFrame you would use the 'or' operator `|`:<br/>
`population[ (population['Year'] == 1960) | (population['Year'] == 1970) ]`


```python
last_quartile =world_pop[(world_pop['Year'] >= 1975) & (world_pop['Year'] <= 1999)]
last_quartile
```





  <div id="df-378782b8-5623-47b2-ad26-b2d15473d909">
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
      <th>Year</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25</th>
      <td>1975</td>
      <td>4089083233</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1976</td>
      <td>4160185010</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1977</td>
      <td>4232084578</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1978</td>
      <td>4304105753</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1979</td>
      <td>4379013942</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1980</td>
      <td>4451362735</td>
    </tr>
    <tr>
      <th>31</th>
      <td>1981</td>
      <td>4534410125</td>
    </tr>
    <tr>
      <th>32</th>
      <td>1982</td>
      <td>4614566561</td>
    </tr>
    <tr>
      <th>33</th>
      <td>1983</td>
      <td>4695736743</td>
    </tr>
    <tr>
      <th>34</th>
      <td>1984</td>
      <td>4774569391</td>
    </tr>
    <tr>
      <th>35</th>
      <td>1985</td>
      <td>4856462699</td>
    </tr>
    <tr>
      <th>36</th>
      <td>1986</td>
      <td>4940571232</td>
    </tr>
    <tr>
      <th>37</th>
      <td>1987</td>
      <td>5027200492</td>
    </tr>
    <tr>
      <th>38</th>
      <td>1988</td>
      <td>5114557167</td>
    </tr>
    <tr>
      <th>39</th>
      <td>1989</td>
      <td>5201440110</td>
    </tr>
    <tr>
      <th>40</th>
      <td>1990</td>
      <td>5288955934</td>
    </tr>
    <tr>
      <th>41</th>
      <td>1991</td>
      <td>5371585922</td>
    </tr>
    <tr>
      <th>42</th>
      <td>1992</td>
      <td>5456136278</td>
    </tr>
    <tr>
      <th>43</th>
      <td>1993</td>
      <td>5538268316</td>
    </tr>
    <tr>
      <th>44</th>
      <td>1994</td>
      <td>5618682132</td>
    </tr>
    <tr>
      <th>45</th>
      <td>1995</td>
      <td>5699202985</td>
    </tr>
    <tr>
      <th>46</th>
      <td>1996</td>
      <td>5779440593</td>
    </tr>
    <tr>
      <th>47</th>
      <td>1997</td>
      <td>5857972543</td>
    </tr>
    <tr>
      <th>48</th>
      <td>1998</td>
      <td>5935213248</td>
    </tr>
    <tr>
      <th>49</th>
      <td>1999</td>
      <td>6012074922</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-378782b8-5623-47b2-ad26-b2d15473d909')"
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
          document.querySelector('#df-378782b8-5623-47b2-ad26-b2d15473d909 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-378782b8-5623-47b2-ad26-b2d15473d909');
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




**Question 1.5**<br/>
Now find the average of the world population in `last_quartile`. Use numpy to find the average.


```python
aver = np.average(last_quartile, axis= 0)
aver[1]
```




    5037315305.76



## 2. Top Movies ##

**Case 2.1**<br/>
Read in 'top_movies.csv' into a DataFrame called `top_movies`. Show the first 5 rows with `head`.


```python
movies = pd.read_csv('https://raw.githubusercontent.com/LGKerby/Python/master/top_movies.csv')
movies
```





  <div id="df-7eff8d8f-ea30-4b84-9e9e-bdd8a83faeae">
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
      <th>Title</th>
      <th>Studio</th>
      <th>Gross</th>
      <th>Gross (Adjusted)</th>
      <th>Year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Star Wars: The Force Awakens</td>
      <td>Buena Vista (Disney)</td>
      <td>906723418</td>
      <td>906723400</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Avatar</td>
      <td>Fox</td>
      <td>760507625</td>
      <td>846120800</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Titanic</td>
      <td>Paramount</td>
      <td>658672302</td>
      <td>1178627900</td>
      <td>1997</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jurassic World</td>
      <td>Universal</td>
      <td>652270625</td>
      <td>687728000</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Marvel's The Avengers</td>
      <td>Buena Vista (Disney)</td>
      <td>623357910</td>
      <td>668866600</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>195</th>
      <td>The Caine Mutiny</td>
      <td>Columbia</td>
      <td>21750000</td>
      <td>386173500</td>
      <td>1954</td>
    </tr>
    <tr>
      <th>196</th>
      <td>The Bells of St. Mary's</td>
      <td>RKO</td>
      <td>21333333</td>
      <td>545882400</td>
      <td>1945</td>
    </tr>
    <tr>
      <th>197</th>
      <td>Duel in the Sun</td>
      <td>Selz.</td>
      <td>20408163</td>
      <td>443877500</td>
      <td>1946</td>
    </tr>
    <tr>
      <th>198</th>
      <td>Sergeant York</td>
      <td>Warner Bros.</td>
      <td>16361885</td>
      <td>418671800</td>
      <td>1941</td>
    </tr>
    <tr>
      <th>199</th>
      <td>The Four Horsemen of the Apocalypse</td>
      <td>MPC</td>
      <td>9183673</td>
      <td>399489800</td>
      <td>1921</td>
    </tr>
  </tbody>
</table>
<p>200 rows × 5 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-7eff8d8f-ea30-4b84-9e9e-bdd8a83faeae')"
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
          document.querySelector('#df-7eff8d8f-ea30-4b84-9e9e-bdd8a83faeae button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-7eff8d8f-ea30-4b84-9e9e-bdd8a83faeae');
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




**Case 2.2**<br/>
What movies were released in 2005? Use python to show just the movies titles (column).


```python
movies_realeased = movies[movies['Year'] == 2005].iloc[0:]['Title']
movies_realeased
```




    24         Star Wars: Episode III - Revenge of the Sith
    60    The Chronicles of Narnia: The Lion, the Witch ...
    62                  Harry Potter and the Goblet of Fire
    Name: Title, dtype: object



**Case 2.3**<br/>
Use a boolean mask to create a DataFrame called `paramount` that contains only movies produced by 'Paramount'.


```python
paramount = movies[movies["Studio"] == 'Paramount']
paramount
```





  <div id="df-3bf05e15-9712-4170-90f5-b76ef31a7589">
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
      <th>Title</th>
      <th>Studio</th>
      <th>Gross</th>
      <th>Gross (Adjusted)</th>
      <th>Year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>Titanic</td>
      <td>Paramount</td>
      <td>658672302</td>
      <td>1178627900</td>
      <td>1997</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Forrest Gump</td>
      <td>Paramount</td>
      <td>330252182</td>
      <td>683929300</td>
      <td>1994</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Iron Man</td>
      <td>Paramount</td>
      <td>318412101</td>
      <td>385808100</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Indiana Jones and the Kingdom of the Crystal S...</td>
      <td>Paramount</td>
      <td>317101119</td>
      <td>384231200</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Iron Man 2</td>
      <td>Paramount</td>
      <td>312433331</td>
      <td>341908200</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>77</th>
      <td>Raiders of the Lost Ark</td>
      <td>Paramount</td>
      <td>248159971</td>
      <td>770183000</td>
      <td>1981</td>
    </tr>
    <tr>
      <th>84</th>
      <td>Beverly Hills Cop</td>
      <td>Paramount</td>
      <td>234760478</td>
      <td>584205200</td>
      <td>1984</td>
    </tr>
    <tr>
      <th>92</th>
      <td>Ghost</td>
      <td>Paramount</td>
      <td>217631306</td>
      <td>447747400</td>
      <td>1990</td>
    </tr>
    <tr>
      <th>95</th>
      <td>Mission: Impossible II</td>
      <td>Paramount</td>
      <td>215409889</td>
      <td>347693200</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>101</th>
      <td>Indiana Jones and the Last Crusade</td>
      <td>Paramount</td>
      <td>197171806</td>
      <td>429923500</td>
      <td>1989</td>
    </tr>
    <tr>
      <th>103</th>
      <td>Grease</td>
      <td>Paramount</td>
      <td>188755690</td>
      <td>669632000</td>
      <td>1978</td>
    </tr>
    <tr>
      <th>109</th>
      <td>Mission: Impossible</td>
      <td>Paramount</td>
      <td>180981856</td>
      <td>356231300</td>
      <td>1996</td>
    </tr>
    <tr>
      <th>110</th>
      <td>Indiana Jones and the Temple of Doom</td>
      <td>Paramount</td>
      <td>179870271</td>
      <td>465735500</td>
      <td>1984</td>
    </tr>
    <tr>
      <th>111</th>
      <td>Top Gun</td>
      <td>Paramount</td>
      <td>179800601</td>
      <td>417818200</td>
      <td>1986</td>
    </tr>
    <tr>
      <th>115</th>
      <td>Crocodile Dundee</td>
      <td>Paramount</td>
      <td>174803506</td>
      <td>401961400</td>
      <td>1986</td>
    </tr>
    <tr>
      <th>124</th>
      <td>The Firm</td>
      <td>Paramount</td>
      <td>158348367</td>
      <td>332761100</td>
      <td>1993</td>
    </tr>
    <tr>
      <th>125</th>
      <td>Fatal Attraction</td>
      <td>Paramount</td>
      <td>156645693</td>
      <td>345222500</td>
      <td>1987</td>
    </tr>
    <tr>
      <th>128</th>
      <td>Beverly Hills Cop II</td>
      <td>Paramount</td>
      <td>153665036</td>
      <td>341914500</td>
      <td>1987</td>
    </tr>
    <tr>
      <th>135</th>
      <td>The Godfather</td>
      <td>Paramount</td>
      <td>134966411</td>
      <td>686626300</td>
      <td>1972</td>
    </tr>
    <tr>
      <th>138</th>
      <td>An Officer and a Gentleman</td>
      <td>Paramount</td>
      <td>129795554</td>
      <td>379814600</td>
      <td>1982</td>
    </tr>
    <tr>
      <th>151</th>
      <td>Love Story</td>
      <td>Paramount</td>
      <td>106397186</td>
      <td>608983900</td>
      <td>1970</td>
    </tr>
    <tr>
      <th>160</th>
      <td>Saturday Night Fever</td>
      <td>Paramount</td>
      <td>94213184</td>
      <td>353261200</td>
      <td>1977</td>
    </tr>
    <tr>
      <th>174</th>
      <td>The Ten Commandments</td>
      <td>Paramount</td>
      <td>65500000</td>
      <td>1139700000</td>
      <td>1956</td>
    </tr>
    <tr>
      <th>188</th>
      <td>Rear Window</td>
      <td>Paramount</td>
      <td>36764313</td>
      <td>438086300</td>
      <td>1954</td>
    </tr>
    <tr>
      <th>190</th>
      <td>The Greatest Show on Earth</td>
      <td>Paramount</td>
      <td>36000000</td>
      <td>522000000</td>
      <td>1952</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-3bf05e15-9712-4170-90f5-b76ef31a7589')"
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
          document.querySelector('#df-3bf05e15-9712-4170-90f5-b76ef31a7589 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-3bf05e15-9712-4170-90f5-b76ef31a7589');
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




**Case 2.4** <br/>
How many movies did Paramount produce? Use python to find the answer.


```python
num_movies_produced = paramount.shape[0]
num_movies_produced
```




    25



**Case2.5**<br/>
In what year was the first movie from Paramount (that appears in this top movies list)? Use python to find the answer.


```python
first_movie = paramount["Year"].min()
first_movie
```




    1952



**Case 2.6**<br/>
What was the name of that movie? Use python to show the answer.


```python
movie_name = paramount[paramount['Year'] == first_movie]['Title']
movie_name
```




    190    The Greatest Show on Earth
    Name: Title, dtype: object



**Case 2.7.** <br/>
What was the average Adjusted Gross for all the Paramount movies (in this top movies list)? Use numpy.


```python
ave_ = np.average(paramount['Gross (Adjusted)'])
ave_

```




    520560232.0


