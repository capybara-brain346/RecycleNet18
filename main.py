import streamlit as st
from typing import Any, List, Dict, Tuple
import requests
import llm
import llm.inference
import llm.prompts

infered_recyclable_item_class_name = ""

st.title("RecycleNet18")

left_column, right_column = st.columns(2)

with left_column:
    st.header("Upload Your Image 📷!")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        files: Dict[str, Tuple[Any]] = {
            "file": (
                uploaded_image.name,
                uploaded_image.read(),
                "image/jpeg",
            )
        }

        resp: Dict[str, Any] = dict(
            requests.post(
                "https://recyclenet18.onrender.com/upload", files=files
            ).json()
        )

        infered_recyclable_item_class_name: str = resp["prediction"]["class_name"]
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        st.write(f"Class: {infered_recyclable_item_class_name}")
        st.write(f"Confidence: {resp["prediction"]["confidence"]}")


with right_column:
    st.header("Chat with the LLM")

    chat_history: List[Dict[str, str]] = [
        {
            "System": "You are RecycleBot, an expert assistant dedicated to helping users recycle items effectively. You understand the rules and best practices for recycling in various regions and provide clear, actionable advice. Always offer tips that are environmentally friendly and practical. If you're unsure, suggest consulting local recycling guidelines."
        }
    ]

    user_input = st.text_input("Your message:", key="chat_input")

    if st.button("Send"):
        if user_input:
            chat_history.append({"user": user_input})
            response = llm.inference.chat(user_input, chat_history=chat_history)
            response = response.split("RecycleBot:")[-1].strip()
            # print(response)
            chat_history.append({"bot": response})

    for chat in chat_history:
        if "user" in chat:
            st.markdown(f"**You:** {chat['user']}")
        elif "bot" in chat:
            st.markdown(f"**Bot:** {chat['bot']}")
