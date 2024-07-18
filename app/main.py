import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from typing import OrderedDict
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from PIL import Image
from io import BytesIO
import streamlit as st


def get_classification(image_bytes: bytes) -> tuple[int, float] | None:
    """
    Classify an image and return the class index and confidence score.

    Args:
        image_bytes (bytes): The image data in bytes format.

    Returns:
        tuple[int, float] | None: A tuple containing the class index and confidence score,
        or None if classification fails.
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on GPU..." if torch.cuda.is_available() else "Running on CPU...")

    BATCH_NORM_MEAN: list[float] = [0.485, 0.456, 0.406]
    BATCH_NORM_STD: list[float] = [0.229, 0.224, 0.225]
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=BATCH_NORM_MEAN, std=BATCH_NORM_STD),
        ]
    )
    RecycleNet = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
    num_features = RecycleNet.fc.in_features
    RecycleNet.fc = nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(num_features, 256)),
                ("relu1", nn.ReLU()),
                ("fc2", nn.Linear(256, 30)),
            ]
        )
    )
    RecycleNet.load_state_dict(torch.load("./RecycleNet18.pth"))
    RecycleNet.to(device)
    RecycleNet.eval()

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    st.image(image)
    input_tensor = transform(image).unsqueeze(0)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = RecycleNet(input_tensor)

    logits_to_probablities = torch.nn.functional.softmax(output[0], dim=0)
    class_idx = torch.argmax(logits_to_probablities).item()
    return class_idx, logits_to_probablities[class_idx].item()


def chat_response(object: str) -> None:
    """
    Generate and display a chat response regarding the sustainability and recyclability of an object.

    Args:
        object (str): The object to inquire about its recyclability.

    Returns:
        None
    """

    st.subheader(
        "Hi 👋🏼, I am here to help you with you're questions on sustainability ♻️"
    )
    st.divider()

    model = ChatOllama(model="llama2")

    # output_schema = {
    #     "title": "Recyclable object",
    #     "description": "Identify information about the recyclability of the object.",
    #     "properties": "A step-by-step guide to recycle the object in sustainable way.",
    # }

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a highly knowledgeable ecologist with extensive expertise in sustainability and recycling practices.",
            ),
            (
                "human",
                """
                Provide an explanation on how to recycle or reuse {object}. 
                Include detailed information about the recyclability of {object} and provide a step-by-step guide on how to recycle {object} in the most sustainable way.
                """,
            ),
        ]
    )

    run = template | model | StrOutputParser()
    response = run.stream({"object": object})
    st.write(response)


st.title("RecycleNet 18 Demo ⚒️")
uploaded_image = st.file_uploader("Choose a file")

if uploaded_image is not None:
    bytes_data = uploaded_image.read()

    class_map = {
        0: "aerosol cans",
        1: "aluminum food_cans",
        2: "aluminum soda cans",
        3: "cardboard boxes",
        4: "cardboard packaging",
        5: "clothing",
        6: "coffee grounds",
        7: "disposable plastic cutlery",
        8: "eggshells",
        9: "food waste",
        10: "glass beverage bottles",
        11: "glass cosmetic containers",
        12: "glass food jars",
        13: "magazines",
        14: "newspaper",
        15: "office paper",
        16: "paper cups",
        17: "plastic cup lids",
        18: "plastic detergent bottles",
        19: "plastic food containers",
        20: "plastic shopping bags",
        21: "plastic soda bottles",
        22: "plastic straws",
        23: "plastic trash bags",
        24: "plastic water bottles",
        25: "shoes",
        26: "steel food cans",
        27: "styrofoam cups",
        28: "styrofoam food containers",
        29: "tea bags",
    }

    predicted_class, class_probability = get_classification(bytes_data)
    predicted_class = class_map[predicted_class]
    st.subheader(f"Predicted class: {predicted_class.title()}")
    st.divider()
    chat_response(predicted_class)
