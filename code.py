import pandas as pd
df_houseindia = pd.read_csv("HouseIndia (1).csv")
import seaborn as sns
df_houseindia['Price_numeric'] = pd.to_numeric(df_houseindia['Price (INR)'].str.replace(',', ''))
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Create an instance of LabelEncoder
label_encoder = LabelEncoder()


# Encode the 'City' column
df_houseindia['City_encoded'] = label_encoder.fit_transform(df_houseindia['City'])

# Encode the 'State' column
df_houseindia['State_encoded'] = label_encoder.fit_transform(df_houseindia['State'])

df_houseindia_encoded = pd.get_dummies(df_houseindia, columns=['City', 'State'], drop_first=True)
df_houseindia_encoded[['City_Ahmedabad', 'City_Aizawl', 'City_Bangalore', 'City_Berhampur', 'City_Bhubaneswar', 'City_Chandigarh', 'City_Chennai', 'City_Coimbatore', 'City_Cuttack', 'City_Dehradun', 'City_Delhi', 'City_Dharmanagar', 'City_Dibrugarh', 'City_Dimapur', 'City_Dispur', 'City_Durgapur', 'City_Gangtok', 'City_Gaya', 'City_Ghaziabad', 'City_Guwahati', 'City_Haridwar', 'City_Hyderabad', 'City_Imphal', 'City_Itanagar', 'City_Jaipur', 'City_Jamshedpur', 'City_Jodhpur', 'City_Kochi', 'City_Kohima', 'City_Kolkata', 'City_Kozhikode', 'City_Lucknow', 'City_Madurai', 'City_Mangalore', 'City_Mumbai', 'City_Mysore', 'City_Nagpur', 'City_Naharlagun', 'City_Nainital', 'City_Noida', 'City_Patna', 'City_Pune', 'City_Puri', 'City_Ranchi', 'City_Rourkela', 'City_Sambalpur', 'City_Shillong', 'City_Siliguri', 'City_Thiruvananthapuram', 'City_Tura', 'City_Udaipur', 'State_Assam', 'State_Bihar', 'State_Delhi', 'State_Gujarat', 'State_Jharkhand', 'State_Karnataka', 'State_Kerala', 'State_Maharashtra', 'State_Manipur', 'State_Meghalaya', 'State_Mizoram', 'State_Nagaland', 'State_Odisha', 'State_Punjab', 'State_Rajasthan', 'State_Sikkim', 'State_Tamil Nadu', 'State_Telangana', 'State_Tripura', 'State_Uttar Pradesh', 'State_Uttarakhand', 'State_West Bengal']] = df_houseindia_encoded[['City_Ahmedabad', 'City_Aizawl', 'City_Bangalore', 'City_Berhampur', 'City_Bhubaneswar', 'City_Chandigarh', 'City_Chennai', 'City_Coimbatore', 'City_Cuttack', 'City_Dehradun', 'City_Delhi', 'City_Dharmanagar', 'City_Dibrugarh', 'City_Dimapur', 'City_Dispur', 'City_Durgapur', 'City_Gangtok', 'City_Gaya', 'City_Ghaziabad', 'City_Guwahati', 'City_Haridwar', 'City_Hyderabad', 'City_Imphal', 'City_Itanagar', 'City_Jaipur', 'City_Jamshedpur', 'City_Jodhpur', 'City_Kochi', 'City_Kohima', 'City_Kolkata', 'City_Kozhikode', 'City_Lucknow', 'City_Madurai', 'City_Mangalore', 'City_Mumbai', 'City_Mysore', 'City_Nagpur', 'City_Naharlagun', 'City_Nainital', 'City_Noida', 'City_Patna', 'City_Pune', 'City_Puri', 'City_Ranchi', 'City_Rourkela', 'City_Sambalpur', 'City_Shillong', 'City_Siliguri', 'City_Thiruvananthapuram', 'City_Tura', 'City_Udaipur', 'State_Assam', 'State_Bihar', 'State_Delhi', 'State_Gujarat', 'State_Jharkhand', 'State_Karnataka', 'State_Kerala', 'State_Maharashtra', 'State_Manipur', 'State_Meghalaya', 'State_Mizoram', 'State_Nagaland', 'State_Odisha', 'State_Punjab', 'State_Rajasthan', 'State_Sikkim', 'State_Tamil Nadu', 'State_Telangana', 'State_Tripura', 'State_Uttar Pradesh', 'State_Uttarakhand', 'State_West Bengal']].astype(int)
x_data_encoded_1 = df_houseindia_encoded.drop(['Price (INR)','Price_numeric', 'City_encoded', 'State_encoded'], axis=1)
y_data_1 = df_houseindia_encoded['Price_numeric']
x_train, x_test, y_train, y_test = train_test_split(x_data_encoded_1, y_data_1, test_size=0.2, random_state=70)
from sklearn.linear_model import Ridge

# Create an instance of the Ridge model
model_ridge = Ridge()

# Fit the model to the training data
model_ridge.fit(x_train, y_train)