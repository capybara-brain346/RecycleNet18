import streamlit as st
import requests
import llm
import llm.inference
import llm.prompts

infered_recyclable_item_class_name = ""


def chat_with_llm(user_input):
    responses = {
        "hello": "Hi there! How can I help you?",
        "who are you?": "I'm an AI chatbot powered by an LLM.",
        "bye": "Goodbye! Have a great day!",
    }
    return responses.get(user_input.lower(), "Sorry, I don't understand that.")


st.title("Image Upload & LLM Chat App")

left_column, right_column = st.columns(2)

with left_column:
    st.header("Image Upload")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        files = {
            "file": (
                uploaded_image.name,
                uploaded_image.read(),
                "image/jpeg",
            )
        }
        resp = dict(
            requests.post(
                "https://recyclenet18.onrender.com/upload", files=files
            ).json()
        )
        infered_recyclable_item_class_name = resp["prediction"]["class_name"]
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        st.write(f"Class: {infered_recyclable_item_class_name}")
        st.write(f"Confidence: {resp["prediction"]["confidence"]}")


with right_column:
    st.header("Chat with the LLM")

    chat_history = [
        {"System": llm.prompts.system_prompt.format(infered_recyclable_item_class_name)}
    ]
    user_input = st.text_input("Your message:", key="chat_input")
    if st.button("Send"):
        if user_input:
            chat_history.append({"user": user_input})
            response = llm.inference.chat(user_input, chat_history=chat_history)
            print(response)
            chat_history.append({"bot": response})

    for chat in chat_history:
        if "user" in chat:
            st.markdown(f"**You:** {chat['user']}")
        elif "bot" in chat:
            st.markdown(f"**Bot:** {chat['bot']}")
