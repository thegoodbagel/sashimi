import streamlit as st

def show_info_page():
    st.header("🍱 About This App")
    st.markdown("""
    Hi! This website doesn't just classify sushi—it includes several types of **Japanese seafood dishes**, including **sashimi**!

    ---
    ### 🍚 What is Sushi, Really?
    The word **sushi** actually refers to **rice seasoned with sweetened vinegar**, usually combined with various ingredients. While raw seafood is most common, sushi doesn't always include fish!

    ---
    ### 🍽️ Dish Categories""")
    
    for i, dish in enumerate(dish_categories):
        with st.container():

            col_img, col_text = st.columns([1, 3])

            with col_img:
                st.image(dish["image"], width=250)

            with col_text:
                st.markdown(f'<h5 class="dish-name">{dish["name"]}</h5>', unsafe_allow_html=True)
                st.markdown(f'<p class="dish-desc">{dish["desc"]}</p>', unsafe_allow_html=True)
                if "subtypes" in dish:
                    st.markdown('<p class="dish-subtypes"><b>Subtypes:</b></p>', unsafe_allow_html=True)
                    for subtype in dish["subtypes"]:
                        st.markdown(f'<p class="dish-subtypes">- {subtype}</p>', unsafe_allow_html=True)



# Data structure for dishes and subtypes
dish_categories = [
    {
        "name": "Sashimi",
        "desc": "Thinly sliced raw seafood, served without rice.",
        "image": "components/assets/sashimi.jpeg"
    },
    {
        "name": "Nigiri",
        "desc": "Hand-formed rice topped with a slice of seafood.",
        "image": "components/assets/nigiri.jpeg"
    },
    {
        "name": "Maki Sushi",
        "desc": "Rolled sushi with rice and fillings wrapped in seaweed.",
        "subtypes": [
            "Hosomaki: Thin rolls with one filling.",
            "Futomaki: Thick rolls with multiple fillings.",
            "Uramaki: Inside-out rolls with rice on the outside.",
            "Temaki: Hand-rolled cone-shaped sushi.",
            "Gunkan Maki: Elliptic, boat-shaped sushi wrapped in seaweed with toppings."
        ],
        "image": "components/assets/maki.jpeg"
    },
    {
        "name": "Chirashi",
        "desc": "A bowl of sushi rice topped with assorted ingredients.",
        "image": "components/assets/chirashi.jpeg"
    },
    {
        "name": "Temari Sushi",
        "desc": "Ball-shaped sushi made by hand.",
        "image": "components/assets/temari.jpeg"
    },
    {
        "name": "Inarizushi",
        "desc": "Rice stuffed into sweet fried tofu pouches.",
        "image": "components/assets/inarizushi.jpeg"
    },
    {
        "name": "Oshizushi",
        "desc": "Pressed sushi with layers of fish and rice into a box-like mold.",
        "image": "components/assets/oshizushi.jpeg"
    }
]