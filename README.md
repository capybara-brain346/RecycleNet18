<h1 align="center" id="title">RecycleNet18</h1>

# RecycleNet: Recyclable Items Classification and Chatbot Guide

## Overview

RecycleNet is a deep learning-based project designed to classify images of recyclable items into 30 distinct categories. Alongside the classification model, a chatbot is integrated to assist users in understanding how to recycle the identified items properly. The goal is to promote effective recycling practices and reduce environmental waste.

## Features

### Image Classification

**30 Classes of Recyclable Items**: The model is trained to recognize a variety of common recyclable materials, including:

| **Recyclable Items**            | **Recyclable Items**            |
|----------------------------------|---------------------------------|
| Aerosol Cans                     | Plastic Detergent Bottles       |
| Aluminum Food Cans               | Plastic Food Containers         |
| Aluminum Soda Cans               | Plastic Shopping Bags           |
| Cardboard Boxes                  | Plastic Soda Bottles            |
| Cardboard Packaging              | Plastic Straws                 |
| Clothing                         | Plastic Trash Bags             |
| Coffee Grounds                   | Plastic Water Bottles          |
| Disposable Plastic Cutlery        | Shoes                          |
| Eggshells                        | Steel Food Cans                |
| Food Waste                       | Styrofoam Cups                 |
| Glass Beverage Bottles           | Styrofoam Food Containers       |
| Glass Cosmetic Containers        | Tea Bags                       |
| Glass Food Jars                  | Magazines                      |
| Newspaper                        | Office Paper                   |
| Paper Cups                       | Plastic Cup Lids               |


### Chatbot Guide

- **Interactive Assistance**: The integrated chatbot provides guidance on how to properly recycle the items identified by the classification model.
- **Educational Content**: Learn about different types of recyclable materials and best practices for recycling.
- **User-Friendly Interface**: The chatbot is designed to be intuitive and easy to interact with, making recycling information accessible to everyone.

## Usage

1. **Upload an Image**: Users can upload an image of an item they want to recycle.
2. **Classification**: The model processes the image and classifies it into one of the 30 recyclable categories.
3. **Recycling Guidance**: The chatbot provides detailed information on how to recycle the identified item, including any special instructions or local recycling regulations.

## Future Enhancements

- **Expand Classification Categories**: Include more types of recyclable items to cover a broader range of materials.
- **Localization**: Provide region-specific recycling guidelines and regulations.
- **Improved Chatbot Interaction**: Enhance the chatbot’s capabilities to offer more personalized and detailed assistance.

## Setup Instructions

Follow these steps to set up the project locally on your machine.

### Prerequisites

Ensure you have the following installed:
- >=Python 3.10
- `pip` (Python package manager)
- `anaconda` environment manager 
- `git`

### Clone the Repository

1. Open your terminal and run the following command to clone the repository:

   ```bash
   git clone https://github.com/capybara-brain346/RecycleNet18.git
   ```
2. cd into directory
   ```bash
   cd RecycleNet18
   ```
3. Setup virtual environment
   ```bash
   conda create -p venv python=3.10 -y
   ```
   For windows
   ```bash
   conda activate venv/
   ```
   For UNIX system
   ```bash
   source venv/
   ```

   Install requirements
   ```bash
   pip install -r requirements.txt
   ```

### Installing Ollama

Ollama is a tool that allows you to run large language models locally. Follow the steps below to install it.

### Step 1: Download Ollama

You can download the latest version of Ollama from the official website:

- [Download Ollama](https://ollama.com/download)

### Step 2: Install Ollama by running .exe file

### Step 3: Verify Installation

After installation, open Terminal and verify Ollama is installed by running the following command:

```bash
ollama run llama3.2:3b
```

### Train the model
```bash
python train.py
```

### Run the app
```bash
streamlit run app/main.py
```





