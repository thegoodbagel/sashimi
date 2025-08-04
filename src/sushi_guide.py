import streamlit as st

def show_sushi_guide():
    sashimi_info = {
        "🍣 Salmon (Sake)": """
- **Toro** – Fatty part  
- **Harasu** – Belly cut  
- **Ikura** – Salmon roe
""",
        "🐟 Tuna (Maguro)": """
**Parts**:
- **Kashiraniku** – Top of head  
- **Kama/Kamatoro** – Gill flesh  
- **Otoro** – Fattiest part (harakami to haranaka)  
- **Chutoro** – Medium fatty  
    - **Sekami** – Upper back  
    - **Senaka** – Mid back  
    - **Seshimo** – Lower back (near tail)  
    - **Harashimo** – Lower belly (near tail)  
- **Akami** – Lean red meat

**Types**:
- **Honmaguro** – Northern bluefin  
- **Minamimaguro** – Southern bluefin  
- **Mebachimaguro** – Bigeye  
- **Kihadamaguro** – Yellowfin  
- **Binnagamaguro** – Albacore  
➡️ *Ahi* = Yellowfin & Bigeye
""",
        "🔥 Katsuo (Skipjack Tuna)": "**Tataki** – Lightly grilled outside, raw inside",
        "🟡 Yellowtail Family": """
- **Buri / Hamachi** – Japanese amberjack  
- **Hiramasa** – Yellowtail amberjack  
- **Kanpachi** – Greater amberjack
""",
        "🌊 Other Seafood": """
- **Engawa** – Flounder fin  
- **Hotate** – Scallop  
- **Ebi** – Shrimp / Prawn  
    - Amaebi, Botan ebi – Sweet shrimp  
    - Aka ebi – Red shrimp  
    - Kurama ebi – Japanese tiger prawn  
- **Shime Saba** – Cured mackerel  
- **Aji** – Horse mackerel  
- **Suzuki** – Sea bass
""",
        "🦪 Shellfish": """
- **Hokkigai** – Surf clam  
- **Akagai** – Red clam  
- **Tsubagai** – Whelk  
- **Mirugai** – Geoduck clam  
- **Hiougi (Noble Scallops)**
""",
        "🦑 Others": """
- **Ika** – Squid  
- **Tako** – Octopus  
- **Ika Somen** – Squid “noodles”  
- **Kujira** – Whale
"""
    }

    st.sidebar.header("📖 Sushi Guide")

    # Search input
    query = st.sidebar.text_input("Search sashimi")

    if query:
        query_lower = query.lower()
        found = False
        for name, desc in sashimi_info.items():
            if query_lower in name.lower() or query_lower in desc.lower():
                with st.sidebar.expander(name, expanded=True):
                    st.markdown(desc)
                found = True
        if not found:
            st.sidebar.write("No results found.")
    else:
        # Show full guide if no search
        for name, desc in sashimi_info.items():
            with st.sidebar.expander(name):
                st.markdown(desc)

import streamlit as st

def show_info_page():
    st.title("🍱 About This App")
    st.markdown("""
    Hi! This website doesn't just classify sushi—it includes several types of **Japanese seafood dishes**, including **sashimi**!

    ---
    ### 🍚 What is Sushi, Really?
    The word **sushi** actually refers to **rice seasoned with sweetened vinegar**, usually combined with various ingredients. While raw seafood is most common, sushi doesn't always include fish!

    ---
    ### 🍽️ Dish Categories

    - **Sashimi**: Thinly sliced raw seafood, served without rice.
    - **Nigiri**: Hand-formed rice topped with a slice of seafood.
    - **Maki Sushi**: Rolled sushi with rice and fillings wrapped in seaweed.
    - **Hako Sushi**: Pressed sushi made in a box mold.
    - **Chirashi**: A bowl of sushi rice topped with assorted ingredients.
    - **Temari Sushi**: Ball-shaped sushi made by hand.
    - **Inarizushi**: Rice stuffed into sweet fried tofu pouches.
    - **Oshizushi**: Box-pressed sushi with layers of fish and rice.

    ---
    ### 🐟 Fish & Seafood Categories

    - **Sake (Salmon)**: Popular fatty fish, orange in color.
    - **Maguro (Tuna) Otoro**: Fatty belly portion of bluefin tuna.
    - **Maguro (Tuna) Chutoro**: Medium-fatty part between otoro and akami.
    - **Maguro (Tuna) Akami**: Lean red tuna, classic sushi topping.
    - **Katsuo (Skipjack Tuna)**: Often served seared (tataki style), strong flavor.
    - **Buri / Hamachi (Amberjack / Yellowtail)**: Rich, buttery white fish.
    - **Hiramasa (Yellowtail Amberjack)**: Mild, slightly sweet fish.
    - **Kanpachi (Greater Amberjack)**: Firmer and cleaner taste than hamachi.
    - **Hirame (Flounder)**: Mild white fish, often with a delicate texture.
    - **Karei (Flatfish)**: Similar to hirame but slightly more fibrous.
    - **Engawa (Flounder Fin)**: Fatty, chewy strip from the fin muscle.
    - **Hotate (Scallop)**: Sweet, soft, and creamy shellfish.
    - **Ebi (Shrimp)**: Typically boiled or grilled before serving.
    - **Tai Kurodai (Black Sea Bream)**: Mild and slightly firm flesh.
    - **Tai Kinmedai (Golden Eye Snapper)**: Rich and delicate, slightly oily.
    - **Tai Himedai (Rosy Seabass)**: Buttery, high-fat white fish.
    - **Shime Saba (Cured Mackerel)**: Cured in vinegar, oily and strong.
    - **Aji (Horse Mackerel)**: Light and zesty, often served with garnish.
    - **Suzuki (Sea Bass)**: Delicate white fish with subtle flavor.
    - **Hokkigai (Surf Clam)**: Red-tipped clam, chewy and sweet.
    - **Akagai (Red Clam)**: Firm and slightly briny.
    - **Tsubagai (Whelk)**: Crunchy sea snail meat.
    - **Mirugai (Geoduck Clam)**: Very large clam, crunchy and sweet.
    - **Ika (Squid)**: Chewy and subtly sweet.
    - **Tako (Octopus)**: Firm texture, usually cooked before serving.
    """)

