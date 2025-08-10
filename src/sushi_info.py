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
            st.markdown(
                """
                <style>
                .dish-card {
                    border: 1px solid #ccc;
                    border-radius: 12px;
                    padding: 16px;
                    margin-bottom: 20px;
                    display: flex;
                    align-items: center;
                    background-color: #fff;
                }
                .dish-text {
                    flex: 1;
                    padding: 0 16px;
                }
                .dish-image {
                    width: 200px;
                    height: 200px;
                    object-fit: cover;
                    border-radius: 8px;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            # Alternate image position (left/right)
            if i % 2 == 0:
                col_img, col_text = st.columns([1, 2])
            else:
                col_text, col_img = st.columns([2, 1])

            with col_img:
                st.image(dish["image"], use_column_width=True)

            with col_text:
                st.subheader(dish["name"])
                st.write(dish["desc"])
                if "subtypes" in dish:
                    st.markdown("**Subtypes:**")
                    for subtype in dish["subtypes"]:
                        st.write(f"- {subtype}")



# Data structure for dishes and subtypes
dish_categories = [
    {
        "name": "Sashimi",
        "desc": "Thinly sliced raw seafood, served without rice.",
        "image": "/assets/sashimi.jpg"
    },
    {
        "name": "Nigiri",
        "desc": "Hand-formed rice topped with a slice of seafood.",
        "image": "/assets/nigiri.jpg"
    },
    {
        "name": "Maki Sushi",
        "desc": "Rolled sushi with rice and fillings wrapped in seaweed.",
        "subtypes": [
            "Hosomaki – Thin rolls with one filling.",
            "Futomaki – Thick rolls with multiple fillings.",
            "Uramaki – Inside-out rolls with rice on the outside.",
            "Temaki – Hand-rolled cone-shaped sushi.",
            "Gunkan Maki – Oval-shaped sushi wrapped in seaweed with toppings."
        ],
        "image": "/assets/maki.jpg"
    },
    {
        "name": "Chirashi",
        "desc": "A bowl of sushi rice topped with assorted ingredients.",
        "image": "/assets/chirashi.jpg"
    },
    {
        "name": "Temari Sushi",
        "desc": "Ball-shaped sushi made by hand.",
        "image": "/assets/temari.jpg"
    },
    {
        "name": "Inarizushi",
        "desc": "Rice stuffed into sweet fried tofu pouches.",
        "image": "/assets/inarizushi.jpg"
    },
    {
        "name": "Oshizushi",
        "desc": "Pressed sushi with layers of fish and rice into a box-like mold.",
        "image": "/assets/oshizushi.jpg"
    }
]