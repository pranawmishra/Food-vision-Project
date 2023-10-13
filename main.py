import cv2
from PIL import Image
import numpy as np
import os
import streamlit as st
import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
import preprocess_file

model = tf.keras.models.load_model("models/101_food_Vision_model")
### load file
uploaded_image = preprocess_file.load_image()

class_names = ['apple_pie',
 'baby_back_ribs',
 'baklava',
 'beef_carpaccio',
 'beef_tartare',
 'beet_salad',
 'beignets',
 'bibimbap',
 'bread_pudding',
 'breakfast_burrito',
 'bruschetta',
 'caesar_salad',
 'cannoli',
 'caprese_salad',
 'carrot_cake',
 'ceviche',
 'cheese_plate',
 'cheesecake',
 'chicken_curry',
 'chicken_quesadilla',
 'chicken_wings',
 'chocolate_cake',
 'chocolate_mousse',
 'churros',
 'clam_chowder',
 'club_sandwich',
 'crab_cakes',
 'creme_brulee',
 'croque_madame',
 'cup_cakes',
 'deviled_eggs',
 'donuts',
 'dumplings',
 'edamame',
 'eggs_benedict',
 'escargots',
 'falafel',
 'filet_mignon',
 'fish_and_chips',
 'foie_gras',
 'french_fries',
 'french_onion_soup',
 'french_toast',
 'fried_calamari',
 'fried_rice',
 'frozen_yogurt',
 'garlic_bread',
 'gnocchi',
 'greek_salad',
 'grilled_cheese_sandwich',
 'grilled_salmon',
 'guacamole',
 'gyoza',
 'hamburger',
 'hot_and_sour_soup',
 'hot_dog',
 'huevos_rancheros',
 'hummus',
 'ice_cream',
 'lasagna',
 'lobster_bisque',
 'lobster_roll_sandwich',
 'macaroni_and_cheese',
 'macarons',
 'miso_soup',
 'mussels',
 'nachos',
 'omelette',
 'onion_rings',
 'oysters',
 'pad_thai',
 'paella',
 'pancakes',
 'panna_cotta',
 'peking_duck',
 'pho',
 'pizza',
 'pork_chop',
 'poutine',
 'prime_rib',
 'pulled_pork_sandwich',
 'ramen',
 'ravioli',
 'red_velvet_cake',
 'risotto',
 'samosa',
 'sashimi',
 'scallops',
 'seaweed_salad',
 'shrimp_and_grits',
 'spaghetti_bolognese',
 'spaghetti_carbonara',
 'spring_rolls',
 'steak',
 'strawberry_shortcake',
 'sushi',
 'tacos',
 'takoyaki',
 'tiramisu',
 'tuna_tartare',
 'waffles']
if uploaded_image:
    # Display the uploaded image
    with st.spinner("Processing..."):
        # Create a temporary directory if it doesn't exist
        os.makedirs("temp", exist_ok=True)

        # Define the path for saving the uploaded image
        image_path = os.path.join("temp", uploaded_image.name)

        # Save the image to the specified path
        with open(image_path, "wb") as f:
            f.write(uploaded_image.read())

    img = preprocess_file.load_and_prep_image(image_path)
    # Display the image
    st.image(image_path, caption=f"Uploaded Image: {image_path}", use_column_width=True)

    Genrate_pred = st.button("Generate Prediction")
    if Genrate_pred:
     pred_prob = model.predict(tf.expand_dims(img, axis=0))  # get prediction probablities array
     pred_class = class_names[pred_prob.argmax()]  # get highest prediction probablity index
     st.write(pred_class)
     st.write(pred_prob)
     st.write(pred_prob.argmax())
     st.write(class_names[pred_prob.argmax()])
    #     prediction = tf.argmax(model.predict(image),axis=1)
    #     prediction=np.array(prediction)
    #     st.write(prediction)
    #     # st.write(type(prediction))
    #     st.title("Predicted Label for the image is {}".format(map_dict [prediction[0]]))
