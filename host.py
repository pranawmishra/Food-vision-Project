import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input

model = tf.keras.models.load_model("models/07_efficientnetb0_fine_tuned_101_classes_mixed_precision/")
### load file
uploaded_file = st.file_uploader("Choose a image file", type="jpg")

map_dict={  0 : " apple_pie " ,
            1 : " baby_back_ribs " ,
            2 : " baklava " ,
            3 : " beef_carpaccio " ,
            4 : " beef_tartare " ,
            5 : " beet_salad " ,
            6 : " beignets " ,
            7 : " bibimbap " ,
            8 : " bread_pudding " ,
            9 : " breakfast_burrito " ,
            10 : " bruschetta " ,
            11 : " caesar_salad " ,
            12 : " cannoli " ,
            13 : " caprese_salad " ,
            14 : " carrot_cake " ,
            15 : " ceviche " ,
            16 : " cheesecake " ,
            17 : " cheese_plate " ,
            18 : " chicken_curry " ,
            19 : " chicken_quesadilla " ,
            20 : " chicken_wings " ,
            21 : " chocolate_cake " ,
            22 : " chocolate_mousse " ,
            23 : " churros " ,
            24 : " clam_chowder " ,
            25 : " club_sandwich " ,
            26 : " crab_cakes " ,
            27 : " creme_brulee " ,
            28 : " croque_madame " ,
            29 : " cup_cakes " ,
            30 : " deviled_eggs " ,
            31 : " donuts " ,
            32 : " dumplings " ,
            33 : " edamame " ,
            34 : " eggs_benedict " ,
            35 : " escargots " ,
            36 : " falafel " ,
            37 : " filet_mignon " ,
            38 : " fish_and_chips " ,
            39 : " foie_gras " ,
            40 : " french_fries " ,
            41 : " french_onion_soup " ,
            42 : " french_toast " ,
            43 : " fried_calamari " ,
            44 : " fried_rice " ,
            45 : " frozen_yogurt " ,
            46 : " garlic_bread " ,
            47 : " gnocchi " ,
            48 : " greek_salad " ,
            49 : " grilled_cheese_sandwich " ,
            50 : " grilled_salmon " ,
            51 : " guacamole " ,
            52 : " gyoza " ,
            53 : " hamburger " ,
            54 : " hot_and_sour_soup " ,
            55 : " hot_dog " ,
            56 : " huevos_rancheros " ,
            57 : " hummus " ,
            58 : " ice_cream " ,
            59 : " lasagna " ,
            60 : " lobster_bisque " ,
            61 : " lobster_roll_sandwich " ,
            62 : " macaroni_and_cheese " ,
            63 : " macarons " ,
            64 : " miso_soup " ,
            65 : " mussels " ,
            66 : " nachos " ,
            67 : " omelette " ,
            68 : " onion_rings " ,
            69 : " oysters " ,
            70 : " pad_thai " ,
            71 : " paella " ,
            72 : " pancakes " ,
            73 : " panna_cotta " ,
            74 : " peking_duck " ,
            75 : " pho " ,
            76 : " pizza " ,
            77 : " pork_chop " ,
            78 : " poutine " ,
            79 : " prime_rib " ,
            80 : " pulled_pork_sandwich " ,
            81 : " ramen " ,
            82 : " ravioli " ,
            83 : " red_velvet_cake " ,
            84 : " risotto " ,
            85 : " samosa " ,
            86 : " sashimi " ,
            87 : " scallops " ,
            88 : " seaweed_salad " ,
            89 : " shrimp_and_grits " ,
            90 : " spaghetti_bolognese " ,
            91 : " spaghetti_carbonara " ,
            92 : " spring_rolls " ,
            93 : " steak " ,
            94 : " strawberry_shortcake " ,
            95 : " sushi " ,
            96 : " tacos " ,
            97 : " takoyaki " ,
            98 : " tiramisu " ,
            99 : " tuna_tartare " ,
            100 : " waffles "}
if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(224,224))
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="RGB")

    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]

    Genrate_pred = st.button("Generate Prediction")
    if Genrate_pred:
        prediction = model.predict(img_reshape).argmax()
        st.title("Predicted Label for the image is {}".format(map_dict [prediction]))