import streamlit as st
import requests
from PIL import Image
import io

API_URL = "http://localhost:8000"

st.title("🔍 Face Recognition System")

menu = st.sidebar.selectbox("Choose Option", ["Predict", "Add Person", "Delete Person"])

def get_people():
    try:
        response = requests.get(f"{API_URL}/list-people")
        return response.json().get("people", [])
    except:
        return []

# ----------------------------
# Predict Section
# ----------------------------
if menu == "Predict":
    st.header("Upload Image for Recognition")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(f"{API_URL}/predict", files=files)

            result = response.json()

            st.subheader("Results:")
            st.json(result)

# ----------------------------
# Add Person Section
# ----------------------------
elif menu == "Add Person":
    st.header("Add New Person")

    name = st.text_input("Enter Name")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if st.button("Add Person"):
        if name and uploaded_file:
            files = {"file": uploaded_file.getvalue()}
            data = {"name": name}

            response = requests.post(
                f"{API_URL}/add-person",
                files=files,
                data=data
            )

            st.success(response.json())
        else:
            st.error("Please provide name and image")

            
elif menu == "Delete Person":
    st.header("Delete Person")

    people = get_people()

    if people:
        selected_person = st.selectbox("Select Person", people)

        if st.button("Delete"):
            response = requests.delete(
                f"{API_URL}/delete-person",
                data={"name": selected_person}
            )

            st.success(response.json())
    else:
        st.warning("No people found")