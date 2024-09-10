# import necessary libraries
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from xgboost.sklearn import XGBRegressor 
from sklearn.tree import DecisionTreeRegressor
import os
import pickle
import matplotlib.pyplot as plt  
import plotly.graph_objects as go   




# importning the pickled Dict of townMapping , streetMapping ,flatTypeMapping
with open(r"saved_Model_Pickel/townMapping.pkl", 'rb') as file:
    townDict = pickle.load(file)        
#st.write(townDict)
with open(r"saved_Model_Pickel/streetMapping.pkl", 'rb') as file:
    streetDict = pickle.load(file)
#st.write(streetDict)
with open(r"saved_Model_Pickel/flatTypeMapping.pkl", 'rb') as file:
    flatModelDict = pickle.load(file)
#st.write(flatTypeDict) 

df_4Sel = pd.read_csv(r'csv_data\\csv_4UI_selFilter.csv')

# -------------------------------This is the configuration page for our Streamlit Application---------------------------
st.set_page_config(
    page_title="Singapore Resale Flat Prices Prediction",
    page_icon="chart/icon_singapore.png",
    layout="wide",
    initial_sidebar_state='expanded'
)

# -------------------------------This is the sidebar in a Streamlit application, helps in navigation--------------------
with st.sidebar:
    selected = option_menu("", ["Home", "Predict","Analyse"],
                           icons=["house", "search","table"],
                           styles={"nav-link": {"font": "sans serif", "font-size": "20px", "text-align": "centre"},
                                   "nav-link-selected": {"font": "sans serif", "background-color": "#0072b1"},
                                   "icon": {"font-size": "20px"},
                                   },
                                   default_index=1
                           )
    st.image("chart/singaStat.png", caption="Singapore Resale Flat Prices Predicting.-Machine  Learning Project By Harish Kumar K P harishk_kotte@rediffmail.com")


# -----------------------------------------------Home Section--------------------------------------------------
if selected == "Home":
    st.markdown("## :grey[Singapore Resale Flat Prices Prediction]")
    st.markdown("###### :grey[A Machine Learning Project in Real Estate Domain ]")
    st.divider() 
    #st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)
     
    st.markdown("##### :blue[Overview :]           This project seeks to develop a machine learning model and deploy it as " 
                " an easy-to-use  online application to accurately  predict the resale values of apartments in Singapore . "
                " The model will be based on historical resale flat transactions and is designed to assist both prospective "
                " buyers and sellers in assessing the value of a flat while resale. Resale prices are affected by numerous " 
                " factors, such as location, apartment type, total floor area,lease duration and distances from the major "
                " landmarks and facilities. By offering an estimated resale price based on these factors , the predictive "
                " model aims to help users navigate these complexities.")
    st.markdown("##### :blue[Domain :] Real Estate")
    st.markdown("##### :blue[Technologies :] Python scripting, Pandas, Numpy, Scikit-Learn , Matplotlib , Plotly , Streamlit")
    st.markdown("##### :blue[Process(Machine Learning Pipeline:) :]  Data Wrangling, Data Preprocessing, Data Visualization through EDA , Feature Engineering,"
                " ML Model Parameter Tuning , ML Model Comparision , ML Model Deployment and Application Deployment.") 
    st.divider()
    st.markdown("##### :blue[Technical Discription :] ") 
    st.markdown("###### :blue[1. Glob Module :] With 'glob' module the various Time Series(Historical) data in csv format(coma seperated values) , is combinead as a single list and a Panda dataframe is created ")
    st.markdown("###### :blue[2. Pandas Library :] Pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool , built on top of the Python programming language. "
                " DataFrame , A tabular representation of data in Pandas can be used for easy data wrangling ,data preprocessing and feature engineering a process of creating a column from 2 existing columns in the tabular data or applying python function on datasets as a batch process and where even 2 tables can be combined as one on taking a column as pivot."
                " These Tabular data serves as the backbone for the Machine Learning Inputs.")             
    st.markdown("###### :blue[3. GeoPy Library :] From 'geopy' Library 'geocoder' and 'geodesic' modules were used , 'geocoder' for geocoding that is for extracting a list of Lattitudes and Longitudes of list of MRT Rail Stations and Property Address ,'geodesic' for Distence calculation between two points like Center of Bussiness Development to Property Distence and property to nearby MRT Rail Station Distence,"
                " This was seperatedly done in a different jupiter notebook and that dataframe is expoterd as a 'csv' file and later integrated in main dataframe.")
    st.markdown("###### :blue[4. NumPy Library :] NumPy is the fundamental package for scientific computing in Python , In this Project its use was extended in Lograthmic Calculations and instant array of number generation etc,..  ")
    st.markdown("###### :blue[5. Pickle module :] Python 'pickle' module is used for serializing and de-serializing a Python object structure. Any object in Python can be pickled so that it can be saved on disk, Especially in this project the catogorical features such as 'Town Name' , 'Street Name' , 'Flat Type' and 'Flat Model' are originally representes as 'String Type values',these were converted as numerical 'Integer Type value'"
                " so as get fed as input for ML Modeling the intermediate process a 'Dictionary Type' is created with Strings as dictKeys and corrosponding Integer as dictValue .")
    st.markdown("###### :blue[6. Exploratory Data Analysis (EDA) :] Exploratory Data Analysis or EDA is a crucial step in the data analysis process that involves studying, exploring, and visualizing information to derive important insights. To find patterns, trends, and relationships in the data, it makes use of statistical tools and visualizations . For This Project Primarily 'Matplotlib' was used and in addition 'Plotly' was aslo used.")
    st.markdown("###### :blue[7. Scikit-Learn :] scikit-learn is a Python module for machine learning built on top of SciPy and is distributed under the 3-Clause BSD license , In this Project a lots of ML models were tested from the 'Scikit Learn' modules for Accuracy scores . 'StandardScaler' from 'sklearn.preprocessing' , 'mean_absolute_error, mean_squared_error, r2_score' from 'sklearn.metrics', 'train_test_split , GridSearchCV , GridSearchCV , ShuffleSplit' from 'sklearn.model_selection' " 
                "'LinearRegression' from 'sklearn.linear_model' , 'KNeighborsRegressor' from 'sklearn.neighbors' , 'DecisionTreeRegressor' from 'sklearn.tree' , 'RandomForestRegressor , AdaBoostRegressor , HistGradientBoostingRegressor' from 'sklearn.ensemble' , 'XGBRegressor' from 'xgboost.sklearn' .")
    st.divider()
    st.markdown("##### :blue[Processed Data for Machine Learning :] ")
    st.markdown("###### :blue[1. DataFrame of House Details :] ")
    df_propData = pd.read_csv(r'csv_data\\ML_data4Prediction.csv')
    st.write(df_propData)
    #st.write("The Datasets are Label Endocoded(Converted From String type data to Numerical type data) for 'Town' , 'Street' , FlayType and 'Flat Model' in a Python way so as to create a ML ready Dataframe")
    st.write("###### :red[ *Note : The Datasets are Label Endocoded(Converted From String to Numerical type data) for 'Town' , 'Street' , FlayType and 'Flat Model' in a Python way so as to create a ML ready Dataframe.*]")
    st.write("###### :green[ *Source :* https://data.gov.sg/collections/189/view]")
    st.divider() 
    st.markdown("###### :blue[2. DataFrame of Distence from MRTS & CBD Details :] ")
    df_distData = pd.read_csv(r'csv_data\\mrts_cbd_Dist_onMainDF_ToML.csv')
    st.write(df_distData)
    st.write("###### :red[ *Note : These datasets were created additionally from external resources as a part of feature engineering aspect.*]")
    st.write("###### :green[ *Source :* https://mrtmapsingapore.com/mrt-stations-singapore/ ]")
    st.divider()
    st.markdown("###### :blue[3. Model Comparison chart :] ")
    st.image('chart\\modelComparison.png') 
    st.divider()
    st.markdown("###### :blue[4. Feature Comparison chart :] ")
    st.image('chart\\FeatureChart.png') 


# ------------------------------------------------Predict Section---------------------------------------------------
predicted_price=0
trainScores=[]
if selected == "Predict":
    st.markdown("## :grey[Singapore Resale Flat Prices Prediction]")
    st.markdown("###### :grey[A Machine Learning Project in Real Estate Domain ]")
    st.divider()
    st.markdown("#### :blue[ Predicting Resale Price ]")
    st.markdown("###### :orange[Prediction Through well trained ML Model Extream Gradiant Boost(XGBoost) with Accuracy: ~ 99%(round of R Square Score 0.986870974342762)]")
    st.markdown("###### :violet[*Select Property Specifications for Price Prediction*]")
    #option = st.radio('**Select your Property to Predict the Price**',('Processed Data', 'Prediction Tab',),horizontal=True)



    # Define a mapping of flat_type to numbers
    flatTypeDict = {
    '1 ROOM': 1,
    '2 ROOM': 2,
    '3 ROOM': 3,
    '4 ROOM': 4,
    '5 ROOM': 5,
    'EXECUTIVE': 6,
    'MULTI GENERATION': 7
    }
    # Define a mapping of FloorRange to numbers
    storeyRangeDict ={'1 TO 3': 2,'2 TO 4': 3,'3 TO 5': 4,'4 TO 6': 5,'5 TO 7': 6,'6 TO 8': 7,'7 TO 9': 8,'8 TO 10': 9,'9 TO 11': 10,'10 TO 12': 11,'11 TO 13': 12,'12 TO 14': 13,'13 TO 15': 14,'14 TO 16': 15,'15 TO 17': 16,'16 TO 18': 17,'17 TO 19': 18,'18 TO 20': 19,'19 TO 21': 20,
                      '20 TO 22': 21,'21 TO 23': 22,'22 TO 24': 23,'23 TO 25': 24,'24 TO 26': 25,'25 TO 27': 26,'26 TO 28': 27,'27 TO 29': 28,'28 TO 30': 29,'29 TO 31': 30,'30 TO 32': 31,'31 TO 33': 32,'32 TO 34': 33,'33 TO 35': 34,'34 TO 36': 35,'35 TO 37': 36,'36 TO 38': 37,'37 TO 39': 38,'38 TO 40': 39,
                      '39 TO 41': 40,'40 TO 42': 41,'41 TO 43': 42,'42 TO 44': 43,'43 TO 45': 44,'44 TO 46': 45 ,'45 TO 47': 46,'46 TO 48': 47}
    

    #creating the function for selection selection filter based on pervioua tab selection
    #creating the columns for selection dropdown menu
    col1 ,col2,col3 ,col4,col5 ,col6  = st.columns(6)
    try:
        with col1:
            town = st.selectbox("**Select Town**", list(townDict.keys()))
            town_val = int(townDict[town])
            town_key = town
            #st.write(town_val)

        with col2:
            
            df_new = df_4Sel[df_4Sel['town']==town_key]
            # list unique street names selscted town
            streets = df_new['street_name'].unique()
            # map the streets to povide input to ML model
            streets_mapping = {street: idx + 1 for idx, street in enumerate(streets)}
            

            #optimalStreetDict = sel_flt(streetDict,town_key)
            street = st.selectbox("**Select Street**", list(streets_mapping.keys()))
            street_val = streetDict[street]
            #st.write(street_val )
            street_key = street

        with col3:
            df_new = df_4Sel[(df_4Sel['town'] == town_key) & (df_new['street_name'] == street_key)] 
            flatRoomTypes = df_new['flat_type'].unique()
            # map the streets to povide input to ML model
            flatType_mapping = {flatRoomType: idx + 1 for idx, flatRoomType in enumerate(flatRoomTypes)}

            flatType = st.selectbox("**Select Flat Type**", list(flatType_mapping.keys()))
            flatType_val = flatTypeDict[flatType]
            #st.write(flatType_val)
            
        with col4:
            df_new = df_4Sel[(df_4Sel['town'] == town_key) & (df_new['street_name'] == street_key)& (df_new['flat_type'] == flatType)]
            flatModelTypes = df_new['flat_model'].unique()
            # map the streets to povide input to ML model
            flatModel_mapping = {flatModelType: idx + 1 for idx, flatModelType in enumerate(flatModelTypes)}

            flatModel = st.selectbox("**Select Flat Model**", list(flatModel_mapping.keys()))
            flatModel_val = flatModelDict[flatModel]
            #st.write(flatModel_val)

        with col5:
            storeyCount = st.selectbox("**Select Storey Range**", list(storeyRangeDict.keys()))
            storeyRange_val = storeyRangeDict[storeyCount]
            #st.write(storeyRange_val)

        with col6:
            AreaInSqMts = st.selectbox("**Select Area In Square Meters**", list(np.arange(34,300)))
            #year_val = storeyRangeDict[storeyCount]
            #st.write(AreaInSqMts)

    except:
        st.write("error")
        #creating the columns for selection dropdown menu
    

    col1 ,col2,col3 ,col4,col5 ,col6  = st.columns(6)
    try:
        with col1:
            year = st.selectbox("**Select Resale Year**", list(np.arange(1990,2051)) )
            #year_val = storeyRangeDict[storeyCount]
            #st.write(year)
        with col2:
            month = st.selectbox("**Select Resale Month**", list(np.arange(1,13)))
            #year_val = storeyRangeDict[storeyCount]
            #st.write(month)
        with col3:
            LeaseRemain = st.selectbox("**Select Lease Remain Years**", list(np.arange(1,100)), )
            #year_val = storeyRangeDict[storeyCount]
            #st.write(LeaseRemain)
            
        with col4:
            st.write("**Average CBD Distence**")
            #df_Street= df_new[df_new['street_name']==street ,df_new['flat_model']==flatModel]
            filtered_df = df_new[(df_new['street_name'] == street) & (df_new['flat_model'] == flatModel) & (df_new['flat_type'] == flatType)]            # creating filters of bool series from isin(

            cbd_Dist_mean = filtered_df['cbd_Dist'].mean()
            st.write("##### "+str( cbd_Dist_mean))
            #st.write(cbd_Dist_mean)

        with col5:
            st.write("**Average MRTS Distence**")
            mrt_minDist_mean = filtered_df['mrt_minDist'].mean()
            st.write("##### "+str( mrt_minDist_mean))
            #st.write(mrt_minDist_mean)

        with col6:

            features = {'town':town_val,
                'street_name': street_val,
                'storey_AvgCount': storeyRange_val,
                'flat_model': flatModel_val,
                'flatRoom_type': flatType_val,
                'lease_remain_years': LeaseRemain,
                'resale_year': year,
                'resale_month': month,                
                'floor_area_sqm': AreaInSqMts,
                'cbd_Dist':cbd_Dist_mean,
                'mrt_minDist':mrt_minDist_mean
                }
            
            features_df = pd.DataFrame(features, index=[0])

            st.markdown("###### :small_red_triangle_down: :red[ *click here to predict*]")
            if st.button('Predict Price' ):
                def get_data():
                    # Load data
                    data = pd.read_csv(r'csv_data\\ML_data4Prediction.csv')
                    sample_df = data.sample(n=25000, random_state=42)
                    return sample_df
                
                csv_df = get_data()


                def train_model(data):
                    # Train model here
                    X = data.drop(['resale_price'], axis=1)
                    y = data['resale_price']

                    trained_model = XGBRegressor(n_estimators=2820 ,booster='gbtree', max_depth= 6 ,max_leaves=33, subsample = 1.0 ,colsample_bytree =0.8, random_state=1 , n_jobs=-1 )
                    trained_model.fit(X, y)
                    return trained_model
                
                xgb_regressor = train_model(csv_df)

                # Use the trained model to make predictions
                prediction = xgb_regressor.predict(features_df)  # Replace X_test with your test data

                predicted_price = str(prediction)[1:-1]
                #predicted_price=prediction.round(2)

                # Display the prediction
                st.success(f"Resale Price: :red[S$] :red[{predicted_price}]", icon="✅") 

                #st.header(mean)
        with st.container(border=3):
            st.markdown("### :green[ *Predicted Price*] " +str(predicted_price)+ " :red[ *S$ (Singapore Dollers)*] ")
                
            
    except Exception as e:
        st.write(e)

    #trainScores = modelTrainScores(csv_df)

    #st.write(df_new)
    #st.write(filtered_df)
# ------------------------------------------------Analyse Section---------------------------------------------------
try:
    if selected == "Analyse":
        st.markdown("## :grey[Singapore Resale Flat Prices Prediction]")
        st.markdown("###### :grey[A Machine Learning Project in Real Estate Domain ]")
        st.divider()
        col1 ,col2,col3 = st.columns(3,gap="small")
        with col1:
            st.markdown("##### :violet[Analyzing Results of Future from Past by]")
        with col2:
            st.markdown("##### :grey[Prices on Years Trend Pattern Analysis]")
        with col3:
            st.write(":violet[5Rooms,ImprovedModel,8 Storey & 80 SqM as Standards]")
        
        col1 ,col2,col3,col4 = st.columns(4,gap="small",)      
        with col1:
            with st.container(border=3):
                town1 = st.selectbox(":blue[ **Select Town 1** ]", list(townDict.keys()))
                town1_val  = int(townDict[town1])
                town_key = town1
                #st.write(town1_val)
                
                df_new = df_4Sel[df_4Sel['town']==town_key]
                # list unique street names selscted town
                streets = df_new['street_name'].unique()
                # map the streets to povide input to ML model
                streets_mapping = {street: idx + 1 for idx, street in enumerate(streets)}

                #optimalStreetDict = sel_flt(streetDict,town_key)
                street1 = st.selectbox(":blue[ **Select Town 1 Street** ]", list(streets_mapping.keys()))
                street1_val = streetDict[street1]
                #st.write(street1_val )


                filtered_df = df_new[(df_new['street_name'] == street1) & (df_new['town'] == town1)]            # creating filters of bool series from isin(

                town1cbd_Dist_mean = filtered_df['cbd_Dist'].mean()
                #st.write("##### "+str( town1cbd_Dist_mean))

                #st.write("**Average MRTS Distence**")
                town1mrt_minDist_mean = filtered_df['mrt_minDist'].mean()
                #st.write("##### "+str( town1mrt_minDist_mean))

        with col2:
            with st.container(border=3):
                town2 = st.selectbox(":green[ **Select Town 2**] ", list(townDict.keys()))
                town2_val  = int(townDict[town2])
                town_key = town2
                #st.write(town2_val)
                
                df_new = df_4Sel[df_4Sel['town']==town_key]
                # list unique street names selscted town
                streets = df_new['street_name'].unique()
                # map the streets to povide input to ML model
                streets_mapping = {street: idx + 1 for idx, street in enumerate(streets)}

                #optimalStreetDict = sel_flt(streetDict,town_key)
                street2 = st.selectbox(":green[**Select Town 2 Street**]", list(streets_mapping.keys()))
                street2_val = streetDict[street2]
                #st.write(street2_val )

                filtered_df = df_new[(df_new['street_name'] == street2) & (df_new['town'] == town2)]            # creating filters of bool series from isin(

                town2cbd_Dist_mean = filtered_df['cbd_Dist'].mean()
                #st.write("##### "+str( town1cbd_Dist_mean))

                #st.write("**Average MRTS Distence**")
                town2mrt_minDist_mean = filtered_df['mrt_minDist'].mean()
                #st.write("##### "+str( town1mrt_minDist_mean))

        with col3:
            with st.container(border=3):
                town3 = st.selectbox(":red[**Select Town 3**]", list(townDict.keys()))
                town3_val  = int(townDict[town3])
                town_key = town3
                #st.write(town3_val)
                
                df_new = df_4Sel[df_4Sel['town']==town_key]
                # list unique street names selscted town
                streets = df_new['street_name'].unique()
                # map the streets to povide input to ML model
                streets_mapping = {street: idx + 1 for idx, street in enumerate(streets)}

                #optimalStreetDict = sel_flt(streetDict,town_key)
                street3 = st.selectbox(":red[ **Select Town 3 Street** ]", list(streets_mapping.keys()))
                street3_val = streetDict[street3]
                #st.write(street3_val )

                filtered_df = df_new[(df_new['street_name'] == street3) & (df_new['town'] == town3)]            # creating filters of bool series from isin(

                town3cbd_Dist_mean = filtered_df['cbd_Dist'].mean()
                #st.write("##### "+str( town1cbd_Dist_mean))

                #st.write("**Average MRTS Distence**")
                town3mrt_minDist_mean = filtered_df['mrt_minDist'].mean()
                #st.write("##### "+str( town1mrt_minDist_mean))

        with col4:
            yearsLst=[]
            yearSeriesLst=[]
            priceLst=[]
            townLst = []
            streetLst = []
            townValLst=[town1_val,town2_val,town3_val]
            townNameLst=[town1,town2,town3]
            streetValLst=[street1_val,street2_val,street3_val]
            streetNameLst=[street1,street2,street3]
            mrt_minDistLst =[town1mrt_minDist_mean,town2mrt_minDist_mean,town3mrt_minDist_mean]
            cbd_DistLst=[town1cbd_Dist_mean,town2cbd_Dist_mean,town3cbd_Dist_mean]
            yearStart=1990
            interval_year = st.selectbox("**Select Intervals Year** :orange[**lesserValue => MoreComputing**]", list([6,5,4,3,2,1]) )
            #year_val = storeyRangeDict[storeyCount]
            #st.write(year)
            for i in range (1990,2024,interval_year):
                yearsLst.append(i)
            yearsCount = len(yearsLst)
            st.write( str(yearsCount *3)+":violet[*Decession Tree Regressor calculations*]" )

            st.markdown("###### :small_red_triangle_down: :orange[ *Click here to Analyse by Comparing 3 selected Towns*]")

            if st.button( 'Analyse Trendz' ):
                for i in range(3):
                    town_val=townValLst[i]
                    town = townNameLst[i]
                    street_val = streetValLst[i]
                    street = streetNameLst[i]
                    mrt = mrt_minDistLst[i]
                    cbd = cbd_DistLst[i]

                    for year in yearsLst:
                        features = {'town':town_val,
                            'street_name': street_val,
                            'storey_AvgCount': 8,
                            'flat_model': 1,
                            'flatRoom_type': 5,
                            'lease_remain_years': 50,
                            'resale_year': year,
                            'resale_month': 1,                
                            'floor_area_sqm': 80,
                            'cbd_Dist':town1cbd_Dist_mean,
                            'mrt_minDist':town1mrt_minDist_mean
                            }

                        features_df = pd.DataFrame(features, index=[0])

                        def get_data():
                            # Load data
                            data = pd.read_csv(r'csv_data\\ML_data4Prediction.csv')
                            sample_df = data.sample(n=45000, random_state=47)
                            return sample_df
                        
                        csv_df = get_data()


                        def train_model(data):
                            # Train model here
                            X = data.drop(['resale_price'], axis=1)
                            y = data['resale_price']

                            trained_model = DecisionTreeRegressor(max_depth=25, max_features = 1.0 ,min_samples_leaf = 10, min_samples_split = 4 ,random_state=12)
                            trained_model.fit(X, y)
                            return trained_model
                        
                        dt_regressor = train_model(csv_df)

                        # Use the trained model to make predictions
                        prediction = dt_regressor.predict(features_df)  # Replace X_test with your test data

                        predicted_price = str(prediction)[1:-1]
                        priceLst.append(predicted_price)
                        townLst.append(town)
                        streetLst.append(street)
                        yearSeriesLst.append(year)
                        #predicted_price=prediction.round(2)

                        # Display the prediction
                        #st.success(f"Resale Price: :red[S$] :red[{predicted_price}]", icon="✅") 

                            #st.header(mean)

                        # get the list of tuples from 4 lists.
                        # and merge them by using zip().
                        list_of_tuples = list(zip(yearSeriesLst,townLst,streetLst,priceLst))
                        #list_of_tuples
                        # Converting lists of tuples into
                        # pandas Dataframe.
        df_analyse = pd.DataFrame(list_of_tuples,columns=['Year', 'Town','Street','Price'])        
                                        
        df_town1 =df_analyse[df_analyse['Town']==town1]
        df_town2 =df_analyse[df_analyse['Town']==town2]
        df_town3 =df_analyse[df_analyse['Town']==town3]

        town1Location = town1 + " - " +street1
        df_town1.rename(columns={'Price':town1Location }, inplace=True)
        town2Location = town2 + " - " +street2
        df_town2.rename(columns={'Price':town2Location }, inplace=True)
        town3Location = town3 + " - " +street3
        df_town3.rename(columns={'Price':town3Location }, inplace=True)
        
        df_town1 = df_town1.drop(['Town', 'Street'], axis=1)
        df_town2 = df_town2.drop(['Town', 'Street'], axis=1)
        df_town3 = df_town3.drop(['Town', 'Street'], axis=1)

        df_town_a= df_town1.merge(df_town2, on="Year", how='outer')
        df_town_final = df_town_a.merge(df_town3, on="Year", how='outer')
    
        # Create line chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_town_final['Year'], y=df_town_final[town1Location],
                            mode='lines+markers',name=town1Location,line_color='blue'))
                            
        fig.add_trace(go.Scatter(x=df_town_final['Year'], y=df_town_final[town2Location],
                            mode='lines+markers',name=town2Location,line_color='green'))
                            
        fig.add_trace(go.Scatter(x=df_town_final['Year'], y=df_town_final[town3Location],
                            mode='lines+markers', name=town3Location,line_color='red'))

        st.plotly_chart(fig, use_container_width=False, theme="streamlit", key=None, on_select="ignore", selection_mode=('points', 'box', 'lasso'))

        st.write(df_town_final)
        st.write(":green[*Decession Tree Regressor with ~98% RSquare Score used for this Computation*]")
        
except:
    st.write("")
    st.write(":green[*Decession Tree Regressor with ~98% RSquare Score used for this Computation*]")
    st.write(":red[Note: Please Avoid 'LIM CHU KANG' & 'MARINE PARADE' for best results as these Town Datasets do not match on standardsall other Towns.]")

    


    
