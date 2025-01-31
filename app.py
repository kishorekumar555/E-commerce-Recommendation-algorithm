import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Creating a Sample dataset for training K-Nearest Neighbour Model
users_data = {
    'username': ['Kishore', 'John', 'Karthik', 'Tilak', 'Dhanush'],
    'age': [25, 30, 22, 28, 35],
    'gender': ['M', 'F', 'M', 'F', 'M'],
    'location': ['Chennai', 'Mumbai', 'Kolkata', 'Hyderabad', 'Bangalore'],
    'preferences': [
        "electronics,clothing",
        "books,toys",
        "sports,electronics",
        "clothing,accessories",
        "books,electronics",
    ],
    'purchase_history': [
        ["Smartphone", "Laptop"],
        ["Book: Python Programming"],
        ["Basketball"],
        ["T-shirt", "Jeans"],
        ["Toy Car", "Watch"]
    ]
}

products_data = {
    'product_id': [101, 102, 103, 104, 105, 106, 107, 108],
    'name': [
        "Smartphone", "Laptop", "T-shirt", "Jeans", 
        "Book: Python Programming", "Toy Car", 
        "Basketball", "Watch"],
    'category': [
        "Electronics", "Electronics", "Clothing", 
        "Clothing", "Books", "Toys", 
        "Sports", "Accessories"],
    'tags': [
        "electronics,mobile",
        "electronics,laptop",
        "clothing,t-shirt",
        "clothing,pants",
        "books,programming",
        "toys,kids",
        "sports,basketball",
        "accessories,watches"
    ],
    'price': [699.99, 999.99, 19.99, 49.99, 29.99, 14.99, 29.99, 199.99],
    'rating': [4.5, 4.7, 4.0, 4.2, 4.8, 3.9, 4.5, 4.1]
}

# Creating pandas Dataframes using the sample data 
users_df = pd.DataFrame(users_data)
products_df = pd.DataFrame(products_data)

# Create a CountVectorizer to convert tags into a matrix of token counts
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(products_df['tags'])

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(count_matrix)

# Function to get recommendations based on user preferences
def get_recommendations(username):
    # Get the user's preferences and purchase history
    user_preferences = users_df[users_df['username'] == username]['preferences'].values[0]
    
    # Convert user preferences to a list of tags
    user_tags = [tag.strip() for tag in user_preferences.split(',')]
    
    # Calculate similarity scores for each product based on user tags
    product_scores = []
    
    for idx, row in products_df.iterrows():
        product_tags = row['tags'].split(',')
        score = len(set(user_tags) & set(product_tags))  # Count common tags
        product_scores.append((row['product_id'], score))
    
    # Sort products based on scores in descending order
    product_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Get top recommendations (excluding products already interacted with)
    recommended_products = [product_id for product_id, score in product_scores if score > 0]
    
    return recommended_products

# Streamlit UI
st.title("E-Commerce Recommendation System")
st.header("User Interaction Form")

# User input fields
username = st.text_input("Username")
age = st.text_input("Age")
gender = st.selectbox("Gender", ["Male", "Female", "Others"])

# Location as radio buttons
location_options = ['Chennai', 'Mumbai', 'Kolkata', 'Hyderabad','Bangalore']
location = st.radio("Location", location_options)

# Preferences as checkboxes
preference_options = ["Electronics", "Clothing", "Books", "Toys", "Sports", "Accessories"]
preferences_selected = []
for option in preference_options:
    if st.checkbox(option):
        preferences_selected.append(option)

#Show all products and allow users to express interest in categories
st.header("Explore Products")
for index, row in products_df.iterrows():
    if st.checkbox(f"Interested in {row['name']}?"):
        st.write(f"You selected: {row['name']} (ID: {row['product_id']}) - Price: ${row['price']}, Rating: {row['rating']}")

# Button to fetch recommendations
if st.button("Fetch Recommendations"):
    if username:
        # Check if username exists in the dataset
        if username not in users_df['username'].values:
            st.error("Username not found. Please enter a valid username.")
        else:
            recommended_products = get_recommendations(username)

            # Display recommendations with product names and additional info
            if recommended_products:
                st.subheader("Recommended Products:")
                for product_id in recommended_products:
                    product_name = products_df[products_df['product_id'] == product_id]['name'].values[0]
                    product_price = products_df[products_df['product_id'] == product_id]['price'].values[0]
                    product_rating = products_df[products_df['product_id'] == product_id]['rating'].values[0]
                    st.write(f"Product Name: {product_name} (Product ID: {product_id}) - Price: ${product_price}, Rating: {product_rating}")
            else:
                st.write("No new recommendations available.")
    
    else:
        st.error("Please enter a valid Username.")
