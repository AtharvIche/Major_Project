# Major_Project
Final Year Project :- Location Predication of Wildlife Species Using ML
<br>
📌 Project Title: Wildlife Species Location Prediction System
Made By: A10

1️⃣ Overview of the Project
The goal of this project is to analyze wildlife movement patterns and predict future locations of a species (e.g., vultures) based on historical tracking data. It takes a CSV file with GPS coordinates, time stamps, weather conditions, and food availability, and then uses Machine Learning (ML) to predict the next probable location of the species.

2️⃣ Technologies Used & Their Purpose
Technology	Purpose
Python	Programming language used to implement the project.
Streamlit	Python framework used to build an interactive web app.
Pandas	Used for reading, processing, and analyzing CSV data.
Matplotlib & Seaborn	Used for data visualization (histograms, distributions).
Folium	Used to visualize GPS locations on an interactive map.
Scikit-learn	Used for data preprocessing, model training, and evaluation.
RandomForestRegressor	Machine Learning model used to predict future locations.
3️⃣ Step-by-Step Explanation
📂 Step 1: Uploading the CSV File
The user uploads a CSV file containing data about wildlife movement, including GPS locations, date, weather, and food availability.
The st.file_uploader() function in Streamlit allows users to upload the CSV file.
📊 Step 2: Data Preprocessing
Read the CSV file using pd.read_csv().
Clean column names to remove unnecessary spaces.
Handle missing values:
Numeric columns → Replace missing values with the column mean.
Categorical columns → Replace missing values with the most frequent value (mode).
Extract Latitude & Longitude:
If the CSV contains a "GPS Location" column, we split it into "latitude" and "longitude".
If "latitude" and "longitude" are separate, we convert them to numeric values.
📆 Step 3: Feature Engineering
If a "Date" column exists, we convert it to datetime format and extract useful features:
Day
Month
Year
Additional columns like "Weather" and "Food offered" are used as features.
📈 Step 4: Data Visualization
We use Matplotlib & Seaborn to plot feature distributions:

Histograms help analyze the spread of numerical features (e.g., latitude, longitude, and weather conditions).
Helps in understanding how data is distributed.
🧠 Step 5: Encoding Categorical Data
If there are non-numeric (categorical) features like "Weather", we convert them into numbers using Label Encoding (from sklearn).
📌 Step 6: Defining Features & Target Variables
Features (X):
Date-related features (Day, Month, Year)
Weather (if available)
Food availability (if available)
Target Variables (y):
Latitude
Longitude
These are the values we want to predict!

🤖 Step 7: Training the Machine Learning Model
Model Used: RandomForestRegressor
Why Random Forest?
It is a powerful ensemble learning algorithm.
Works well with numerical data.
Handles missing and unbalanced data better than many models.
Steps to Train the Model:
Split Data:
80% training, 20% testing using train_test_split().
Train Two Random Forest Models:
One for predicting latitude.
One for predicting longitude.
Fit the models on training data using .fit().
python
Copy
Edit
lat_model = RandomForestRegressor(n_estimators=200, random_state=42)
lon_model = RandomForestRegressor(n_estimators=200, random_state=42)

lat_model.fit(X_train, y_lat_train)
lon_model.fit(X_train, y_lon_train)
📊 Step 8: Evaluating the Model
We test the trained model on unseen data and calculate:

Mean Absolute Error (MAE): Measures average prediction error.
R² Score: Measures how well the model explains the variance in data.
python
Copy
Edit
lat_mae = mean_absolute_error(y_lat_test, y_lat_pred)
lon_mae = mean_absolute_error(y_lon_test, y_lon_pred)
lat_r2 = r2_score(y_lat_test, y_lat_pred)
lon_r2 = r2_score(y_lon_test, y_lon_pred)
Higher R² and lower MAE mean better predictions!

📍 Step 9: Predicting the Next Location
We take the latest available data from the dataset.
Use the trained RandomForestRegressor to predict latitude & longitude.
python
Copy
Edit
latest_data = X.iloc[-1:].copy()
future_lat = lat_model.predict(latest_data)[0]
future_lon = lon_model.predict(latest_data)[0]
🗺️ Step 10: Visualizing Past Movements & Predictions
We use Folium to plot:
Past movement path (Blue line).
Starting point (Green marker).
Last known location (Orange marker).
Predicted next location (Red marker).
python
Copy
Edit
folium.Marker(
    [future_lat, future_lon],
    popup=f"Predicted Location: ({future_lat:.6f}, {future_lon:.6f})",
    icon=folium.Icon(color="red", icon="map-marker")
).add_to(m)
Finally, we display the map inside Streamlit using folium_static(m).
💡 Final Output & Working
1️⃣ The user uploads a CSV file with wildlife tracking data.
2️⃣ The system cleans the data, extracts features, and trains a Random Forest Model.
3️⃣ The model predicts the next location of the species.
4️⃣ A map is displayed showing past movement and the predicted location.

📌 Summary of Key Concepts Used
Concept	Description
Data Cleaning	Handling missing values, correcting formats
Feature Engineering	Extracting useful information from dates
Machine Learning Model	RandomForestRegressor
Evaluation Metrics	MAE, R² Score
Prediction	Using trained model to predict wildlife's next location
Data Visualization	Seaborn (histograms), Folium (interactive map)
🔮 Future Enhancements
Improve accuracy by using Deep Learning models like LSTMs.
Include more features (e.g., temperature, terrain data).
Allow real-time data updates from GPS tracking devices.
📢 Conclusion
This project successfully predicts the next movement of wildlife species using machine learning based on past data. The combination of data preprocessing, feature extraction, ML modeling, and visualization makes it an effective tool for conservation and tracking.
