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
        files: Dict[
            str, Tuple[Any]
        ] = {  # the server takes in form-data with the key name "file", "file" contains image byte data
            "file": (
                uploaded_image.name,
                uploaded_image.read(),
                "image/jpeg",
            )
        }

        # json string data is returned from the server, converted to json content then type casted to dict
        resp: Dict[str, Any] = dict(
            requests.post(
                "https://recyclenet18.onrender.com/upload", files=files
            ).json()
        )

        infered_recyclable_item_class_name: str = resp["prediction"][
            "class_name"
        ]  # access class name
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        st.write(f"Class: {infered_recyclable_item_class_name}")
        st.write(f"Confidence: {resp["prediction"]["confidence"]}")  # access confidence


with right_column:
    st.header("Chat with the RecycleBot")

    # store history of chats
    chat_history: List[Dict[str, str]] = [
        {
            "System": "You are RecycleBot, an expert assistant dedicated to helping users recycle items effectively. You provide clear, actionable advice on recycling and suggest best practices aligned with environmental standards. If the user’s query is unclear or incomplete, expand on it with educated guesses and offer helpful suggestions. Always recommend consulting local recycling guidelines when in doubt. Answer in minimum 200 words when asked a proper question."
        }
    ]

    user_input = st.text_input("Your message:", key="chat_input")

    if st.button("Send"):
        if user_input:
            chat_history.append({"user": user_input})  # append user input
            response = llm.inference.chat(
                user_input, chat_history=chat_history
            )  # perform inference
            response = response.split("RecycleBot:")[-1].strip()
            # print(response)
            chat_history.append(
                {"bot": response}
            )  # append generated text to chat history

    for chat in chat_history:
        if "user" in chat:
            st.markdown(f"**You:** {chat['user']}")
        elif "bot" in chat:
            st.markdown(f"**Bot:** {chat['bot']}")
