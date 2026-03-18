import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

print("Starting Task 2: Financial Market Prediction (Step 7)...")

# 1. Load the data
ds2 = pd.read_csv('data/DS2.csv')
ds3 = pd.read_csv('data/DS3.csv')

# 2. Merge the datasets
print("Merging financial and material signals...")
# We join them where both the Date and the Commodity name match
merged_df = pd.merge(ds2, ds3, on=['date', 'commodity'], how='inner')

# Drop any blank rows so our AI doesn't get confused
merged_df = merged_df.dropna()

# Save this merged data so the frontend team can build charts with it!
merged_df.to_csv('data/merged_financials.csv', index=False)
print("Merged data saved to 'data/merged_financials.csv'")

# 3. Pick Features (Inputs) and Target (Output)
print("\nSelecting features to predict Daily Returns...")

# Here is the "Cross-Domain" magic: Mixing Finance with Material Science
features = ['rsi_14', 'macd', 'bollinger_z', 'mqi', 'supply_disruption_prob', 'substitution_elasticity']
X = merged_df[features]

# Target: We want to predict the daily price movement
y = merged_df['daily_return'] 

# 4. Split the data
print("Splitting data for training and testing...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the Model
print("Training the Financial AI Model... (This might take a minute depending on data size)")
model_fin = RandomForestRegressor(n_estimators=100, random_state=42)
model_fin.fit(X_train, y_train)

# 6. Test the Model
print("Testing the model...")
predictions = model_fin.predict(X_test)

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n--- TASK 2 MODEL PERFORMANCE ---")
print(f"Mean Squared Error: {mse:.6f}")
print(f"Accuracy Score (R2): {r2:.2f}")
print("--------------------------------\n")

# 7. Save the trained model
model_filename = 'data/task2_rf_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model_fin, file)

print(f"Success! Financial model trained and saved to: {model_filename}")
print("Your Backend work is officially complete!")