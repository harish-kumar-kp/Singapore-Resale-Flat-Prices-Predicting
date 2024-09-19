# Singapore Resale Flat Prices Predicting


**Introduction**

Project Overview:' Singapore Resale Flat Prices Predicting:A User-Friendly Machine Learning Tool created with Streamlit, This project seeks to develop a machine learning model and deploy it as an easy-to-use online application to accurately predict the resale values of apartments in Singapore . The model will be based on historical resale flat transactions and is designed to assist both prospective buyers and sellers in assessing the value of a flat while resale. Resale prices are affected by numerous factors, such as location, apartment type, total floor area,lease duration and distances from the major landmarks and facilities. By offering an estimated resale price based on these factors , the predictive model aims to help users navigate these complexities.

<br />

**Table of Contents**

1. Tools , Technologies and Skills
2. Installation
3. Utilisation
4. Features
5. Contributing
6. License
7. Contact

<br />

**Key Technologies and Skills**

**Key Technologies**
- Python v3.12
- Pandas 
- Numpy
- Scikit-learn
- xgboost ML model
- Decession Tree ML model
- Streamlit
- Matplotlib

**Key Skills**
Process(Machine Learning Pipeline) : 
- Data Wrangling, 
- Data Preprocessing, 
- Data Visualization through EDA , 
- Feature Engineering, 
- ML Model Parameter Tuning , 
- ML Model Comparision , 
- ML Model Deployment and 
- Application Deployment.


**Technical Discription :**
1. Glob Module : With 'glob' module the various Time Series(Historical) data in csv format(coma seperated values) , is combinead as a single list and a Panda dataframe is created
2. Pandas Library : Pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool , built on top of the Python programming language. DataFrame , A tabular representation of data in Pandas can be used for easy data wrangling ,data preprocessing and feature engineering a process of creating a column from 2 existing columns in the tabular data or applying python function on datasets as a batch process and where even 2 tables can be combined as one on taking a column as pivot. These Tabular data serves as the backbone for the Machine Learning Inputs.
3. GeoPy Library : From 'geopy' Library 'geocoder' and 'geodesic' modules were used , 'geocoder' for geocoding that is for extracting a list of Lattitudes and Longitudes of list of MRT Rail Stations and Property Address ,'geodesic' for Distence calculation between two points like Center of Bussiness Development to Property Distence and property to nearby MRT Rail Station Distence, This was seperatedly done in a different jupiter notebook and that dataframe is expoterd as a 'csv' file and later integrated in main dataframe.
4. NumPy Library : NumPy is the fundamental package for scientific computing in Python , In this Project its use was extended in Lograthmic Calculations and instant array of number generation etc,..
5. Pickle module : Python 'pickle' module is used for serializing and de-serializing a Python object structure. Any object in Python can be pickled so that it can be saved on disk, Especially in this project the catogorical features such as 'Town Name' , 'Street Name' , 'Flat Type' and 'Flat Model' are originally representes as 'String Type values',these were converted as numerical 'Integer Type value' so as get fed as input for ML Modeling the intermediate process a 'Dictionary Type' is created with Strings as dictKeys and corrosponding Integer as dictValue .
6. Exploratory Data Analysis (EDA) : Exploratory Data Analysis or EDA is a crucial step in the data analysis process that involves studying, exploring, and visualizing information to derive important insights. To find patterns, trends, and relationships in the data, it makes use of statistical tools and visualizations . For This Project Primarily 'Matplotlib' was used and in addition 'Plotly' was aslo used.
7. Scikit-Learn : scikit-learn is a Python module for machine learning built on top of SciPy and is distributed under the 3-Clause BSD license , In this Project a lots of ML models were tested from the 'Scikit Learn' modules for Accuracy scores . 'StandardScaler' from 'sklearn.preprocessing' , 'mean_absolute_error, mean_squared_error, r2_score' from 'sklearn.metrics', 'train_test_split , GridSearchCV , GridSearchCV , ShuffleSplit' from 'sklearn.model_selection' 'LinearRegression' from 'sklearn.linear_model' , 'KNeighborsRegressor' from 'sklearn.neighbors' , 'DecisionTreeRegressor' from 'sklearn.tree' , 'RandomForestRegressor , AdaBoostRegressor , HistGradientBoostingRegressor' from 'sklearn.ensemble' , 'XGBRegressor' from 'xgboost.sklearn' .


<br />

**Installation**

To run this project, you need to install the following packages:
```python
pip install streamlit
pip install streamlit-option-menu
pip install pandas
pip install numpy
pip install scikit-learn
pip install xgboost
pip install pickle4
pip install matplotlib
pip install plotly-express
```

<br />

**Utilisation**

To use this project, follow these steps:

1. Clone the repository: ```git clone https://github.com/harish-kumar-kp/Singapore-Resale-Flat-Prices-Predicting.git```
2. Install the required packages and libraries : ```pip install -r requirements.txt```
3. Launch the landingPage with Streamlit and Select the Parameters in dropdon selection boxes for both pages 'Predict' and 'Analyse'
4. Access the app in your browser at URL ```http://localhost:8501``` when you run cloned git repository from the local computer.
5. Access the app in your browser at URL ```https://singapore-resale-flat-prices-predicting-1-4hh9.onrender.com/``` when you run globally.


<br />

**Features**

**Data Fetching:** The data from the source ( https://data.gov.sg/collections/189/view) in the form of multiple from different time series csv files were combined as a single file and pre processed with all data wrangling techniques was again expotred as csv file which is been read by this app.

**Selection of Choices:** The Specifications of the Property has many parameters be selected as the feature for inputing the Machine Learning Model. This has a very advanced filtering feature based the Town and Street selected the the further choices will be narrowed down to the factual existence of those parameters to avoid choosing the inadequate choices. for both 'Predict' and 'Analyse' section of the app ,so as to make the user experience easier and comprehensive ,self understandable.   

**Data Exploratory and Visualization:** The project Feature provides comprehensive data analysis capabilities with Streamlit in the 'Analyse' section of the app to compare the price trends in the past between the 3 different Towns so as to come to a conclusion.This comparison can be done with coice of meger to broader chioce by changing the 'Year Intervals' . 

I🎬 𝗣𝗿𝗼𝗷𝗲𝗰𝘁 𝗗𝗲𝗺𝗼 𝗩𝗶𝗱𝗲𝗼:**I [https://www.youtube.com/watch?v=spn2qX_0ZyY&list=PLvOopPI4rpr8B_EDpqpmuybFApYNbvbIu) 

**Selection of Prediction Models:** This Project Deploys the best chossen model 'Extrean Gradient Boost Regressor' for 'Predict' section which has R2 score of 0.9868480309334249 and 'Decession Tree Regressor' for 'Analyse' section which has R2 score of 0.9802994875778663 .

<br />

**Contributing**

Contributions to this project are welcome! If you encounter any issues or have suggestions for improvements, please feel free to submit a pull request.Your Valuable Sugestions are always invited happily.

<br />

**License**

This project is licensed under the 3-Clause BSD license.

<br />

**Contact**

📧 Email: harishk_kotte@rediffmail.com

🌐 LinkedIn: [https://www.linkedin.com/in/harish-kumar-k-p-67587a262/](https://www.linkedin.com/in/harish-kumar-k-p-67587a262/)

For any further questions or inquiries, feel free to reach out. We are happy to assist you with any queries.

