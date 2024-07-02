from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

model = ChatOllama(model="llama2")

output_schema = {
    "title": "Recyclable object",
    "description": "Identify information about the recyclability of the object.",
    "properties": "A step-by-step guide to recycle the object in sustainable way.",
}

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "you are an expert ecologist with vast experience in the field of sustainability and house deep knowledge about recycling.",
        ),
        (
            "ai",
            "Hi 👋🏼, I am here to help you with you're questions on sustainability ♻️",
        )(
            "human",
            "Tell me how can I recycle or reuse {object}. Return your response in the following format: {output_schema}",
        ),
    ]
)

run = prompt | model | StrOutputParser()

for chunk in run.stream({"object": "plastic cup", "output_schema": output_schema}):
    print(chunk, end="", flush=True)
