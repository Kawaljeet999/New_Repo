import streamlit as st
import pandas as pd
from joblib import load, dump
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import BytesIO 


# Load necessary files (assuming these paths are correct)
model = load('Churn.pkl')
imputer = load('impute.joblib')
scaler = load('scaler.joblib')
df = load('pca_df_transformed.joblib')
Prediction = load('df_labels.joblib')
Labels = load('df_labels.joblib')

# Streamlit UI setup
st.title("Churn Prediction Model")
st.markdown("This model predicts whether a customer will churn or not.")
st.divider()

# Define file to store the data
data_file = 'data_frame.joblib'

# Initialize or load the existing dataFrame
if os.path.exists(data_file):
    st.session_state.data = load(data_file)
else:
    st.session_state.data = pd.DataFrame(columns=["PC1", "PC2", "PC3"])


# Feature inputs for prediction
PC1 = st.number_input("Enter How many Times Customer service calls", min_value=-100000, max_value=100000, value=0, step=1)
PC2 = st.number_input("Enterthere is a Voice mail plan pr not ", min_value=-100000, max_value=100000, value=0, step=1)
PC3 = st.number_input("Enter how many Number vmail messages you get ", min_value=-100000, max_value=100000, value=0, step=1)

if PC1 and PC2 and PC3:  # Ensure user has provided input values
    new_row = {"PC1": PC1, "PC2": PC2, "PC3": PC3}
    st.session_state.data = pd.concat([st.session_state.data, pd.DataFrame([new_row])], ignore_index=True)

    dump(st.session_state.data, data_file)  # Save the dataFrame after adding data

    # Step 2: Scale the data using the scaler
    if not st.session_state.data.empty:
        scaled_data = scaler.transform(st.session_state.data)
        st.session_state.data.loc[:, :] = scaled_data

    # Step 3: Predict using the trained model
    if not st.session_state.data.empty:
        predictions = model.predict(st.session_state.data)
        st.session_state.data['Prediction'] = predictions

        # Apply imputer to handle NaN values in 'Prediction'
        
        df['Prediction'] = Labels
        
        # df['Prediction'] = imputer.fit_transform(df[['Prediction']])
        df.to_csv('New_df', index=False)

        # imputer = SimpleImputer(strategy='median')

        # Step 5: Combine both dataFrames
        st.session_state.data = pd.concat([st.session_state.data, df], axis=0)
        st.session_state.data[['PC1', 'PC2', 'PC3']] = imputer.fit_transform(st.session_state.data[['PC1', 'PC2', 'PC3']])


        

        # if st.session_state.data.isnull().any().any():  # Check if any NaN values exist in the DataFrame
        # Fit and transform the entire DataFrame (all columns including 'Prediction')
        st.session_state.data = pd.DataFrame(imputer.fit_transform(st.session_state.data), columns=st.session_state.data.columns)

        # Save the updated dataFrame to CSV
        st.session_state.data.to_csv('New_df.csv', index=False)

        # st.write(st.session_state.data)



if st.button("Submit"):   
    # Display prediction for the most recent row
    prediction_value = st.session_state.data['Prediction'].iloc[-1]
    st.text_area("Prediction", f"Churn Prediction: {prediction_value}", height=70)
    # Create a function to download data as CSV
    def download_data():
        # Convert the DataFrame to CSV
        csv = st.session_state.data.to_csv(index=False)
        return csv

    # Add the download button to the sidebar
    st.sidebar.download_button(
        label="Download Data as CSV",
        data=download_data(),
        file_name="churn_predictions.csv",
        mime="text/csv"
    )

    # Create a visualization button
    if 'visualization_shown' not in st.session_state:
        st.session_state.visualization_shown = False

# Define button click behavior
if st.button("Show Visualization"):
    st.session_state.visualization_shown = True

    st.header("Visualization")
    st.divider()
    st.title("Churn Prediction Model Visualization")
    st.markdown("This model visualizes whether a customer will churn or not.")
    st.header("Churn Prediction Visualizations")
    st.divider()

   # Calculate churned and active customers
    churned_customers = st.session_state.data[st.session_state.data['Prediction'] == 1].shape[0]
    active_customers = st.session_state.data[st.session_state.data['Prediction'] == 0].shape[0]

    # Create subplots (2 rows, 2 columns)
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    # Pie Chart (Churn vs Active)
    ax[0, 0].pie([active_customers, churned_customers], labels=['Active', 'Churned'], autopct='%1.1f%%', 
                startangle=90, colors=['green', 'red'])
    ax[0, 0].set_title('Churn Distribution')

    # Donut Chart (Churn vs Active)
    wedges, texts, autotexts = ax[0, 1].pie([active_customers, churned_customers], labels=['Active', 'Churned'], 
                                            autopct='%1.1f%%', startangle=90, colors=['green', 'red'], 
                                            wedgeprops={'width': 0.3})
    ax[0, 1].set_title('Churn Distribution (Donut)')

    # Bar Chart (Active vs Churned Customers)
    ax[1, 0].bar(['Active', 'Churned'], [active_customers, churned_customers], color=['green', 'red'])
    ax[1, 0].set_title('Active vs Churned Customers')
    ax[1, 0].set_ylabel('Number of Customers')

    # Histogram for feature distributions (PC1, PC2, and PC3)
    ax[1, 1].hist(st.session_state.data['PC1'], bins=20, alpha=0.7, label='PC1', color='green')
    ax[1, 1].hist(st.session_state.data['PC2'], bins=20, alpha=0.7, label='PC2', color='red')
    ax[1, 1].hist(st.session_state.data['PC3'], bins=20, alpha=0.7, label='PC3', color='blue')
    ax[1, 1].set_title('Feature Distribution')
    ax[1, 1].set_xlabel('Value')
    ax[1, 1].set_ylabel('Frequency')
    ax[1, 1].legend()

    # Display the plots
    st.pyplot(fig)

    
    # Save the plot to a BytesIO object to allow download
    def save_figure(fig):
        img_io = BytesIO()
        fig.savefig(img_io, format='png')
        img_io.seek(0)
        return img_io

    # Save the plot to a BytesIO object
    img_io = save_figure(fig)

    # Create a download button for the figure in the sidebar
    st.sidebar.download_button(
        label="Download All Visualizations as PNG",
        data=img_io,
        file_name="churn_visualizations.png",
        mime="image/png"
    )
        
# # Ensure dataset is initialized in session state
# if "data" not in st.session_state:
#     st.session_state.data = df

# Initialize session states for analysis
if "analysis_shown" not in st.session_state:
    st.session_state.analysis_shown = False
if "sort_column" not in st.session_state:
    st.session_state.sort_column = df.columns[0]  # Default to first column
if "sort_order" not in st.session_state:
    st.session_state.sort_order = "Ascending"
if "page_size" not in st.session_state:
    st.session_state.page_size = 5
if "current_page" not in st.session_state:
    st.session_state.current_page = 1

# Sidebar Analysis Section
with st.sidebar:
    st.header("Analysis Options")
    if st.button("Show Analysis"):
        st.session_state.analysis_shown = True

    if st.session_state.analysis_shown:
        # Sorting controls
        st.subheader("Sorting")
        sort_column = st.selectbox(
            "Select column to sort by:", options=st.session_state.data.columns, index=st.session_state.data.columns.get_loc(st.session_state.sort_column)
        )
        sort_order = st.radio("Sort order", ["Ascending", "Descending"])

        # Update sort state
        st.session_state.sort_column = sort_column
        st.session_state.sort_order = sort_order

        # Pagination controls
        st.subheader("Pagination")
        page_size = st.selectbox("Page size", options=[5, 10, 20], index=[5, 10, 20].index(st.session_state.page_size))
        st.session_state.page_size = page_size
        total_pages = (len(st.session_state.data) // page_size) + (1 if len(st.session_state.data) % page_size else 0)
        current_page = st.number_input("Page number", min_value=1, max_value=total_pages, value=st.session_state.current_page)

        # Update current page
        st.session_state.current_page = current_page

# Main Content Area
if st.session_state.analysis_shown:
    st.header("Customer Data Analysis")

    # Show stats for customer data
    total_customers = st.session_state.data.shape[0]
    churned_customers = st.session_state.data[st.session_state.data['Prediction'] == 1].shape[0]
    active_customers = st.session_state.data[st.session_state.data['Prediction'] == 0].shape[0]
    st.write(f"Total number of customers: {total_customers}")
    st.write(f"Churned customers (Prediction = 1): {churned_customers}")
    st.write(f"Active customers (Prediction = 0): {active_customers}")

    # Display descriptive statistics (mean, median, mode, 75%, etc.)
    st.subheader("Descriptive Statistics")
    stats = st.session_state.data.describe()  # Gets mean, std, min, 25%, 50%, 75%, max for numeric columns
    st.write(stats)

    # Calculate and display mode, median, and mean for each numeric column
    st.subheader("Additional Statistics")
    for column in st.session_state.data.select_dtypes(include=["number"]).columns:
        mean_val = st.session_state.data[column].mean()
        median_val = st.session_state.data[column].median()
        mode_val = st.session_state.data[column].mode().values  # Mode might return multiple values

        st.write(f"**{column}:**")
        st.write(f"  - Mean: {mean_val:.2f}")
        st.write(f"  - Median: {median_val:.2f}")
        st.write(f"  - Mode: {mode_val if len(mode_val) < 3 else mode_val[:3]}")  # Show top 3 modes if there are many
        st.write("")

    # Apply sorting
    sorted_data = st.session_state.data.sort_values(
        by=st.session_state.sort_column, ascending=(st.session_state.sort_order == "Ascending")
    )

    # Apply pagination
    start_row = (st.session_state.current_page - 1) * st.session_state.page_size
    end_row = start_row + st.session_state.page_size
    paginated_data = sorted_data.iloc[start_row:end_row]

    # Display the data
    st.dataframe(paginated_data)

    # Display navigation info
    st.markdown(
        f"Showing rows {start_row + 1} to {min(end_row, len(st.session_state.data))} of {len(st.session_state.data)} "
        f"(Page {st.session_state.current_page}/{total_pages})"
    )
    # Create a CSV file for download
    def create_downloadable_csv(df):
        # Convert dataframe to CSV
        csv = df.to_csv(index=False)
        return csv

    # Add the download button for CSV file in the sidebar
    with st.sidebar:
        csv_data = create_downloadable_csv(st.session_state.data)
        st.download_button(
            label="Download Data Analysis (CSV)",
            data=csv_data,
            file_name="customer_data_analysis.csv",
            mime="text/csv",
        )