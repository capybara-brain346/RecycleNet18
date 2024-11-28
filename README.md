# RecycleNet: Recyclable Items Classification and Chatbot Guide  

## Demo: [Watch on YouTube](https://www.youtube.com/watch?v=K3wz3cSf9is)  

---

## Overview  
RecycleNet is a deep learning-based solution for promoting effective recycling practices. The project combines an image classification model, capable of categorizing recyclable items into 30 distinct classes, with an interactive chatbot that guides users on proper recycling methods.  

## Features  

### **Image Classification**  
RecycleNet identifies items as one of 30 recyclable categories, including:  

| **Recyclable Items**         | **Recyclable Items**            |  
|-------------------------------|---------------------------------|  
| Aerosol Cans                 | Plastic Detergent Bottles       |  
| Aluminum Food Cans           | Plastic Food Containers         |  
| Aluminum Soda Cans           | Plastic Shopping Bags           |  
| Cardboard Boxes              | Plastic Soda Bottles            |  
| Cardboard Packaging          | Plastic Straws                 |  
| Clothing                     | Plastic Trash Bags              |  
| Coffee Grounds               | Plastic Water Bottles           |  
| Disposable Plastic Cutlery   | Shoes                           |  
| Eggshells                    | Steel Food Cans                 |  
| Food Waste                   | Styrofoam Cups                  |  
| Glass Beverage Bottles       | Styrofoam Food Containers       |  
| Glass Cosmetic Containers    | Tea Bags                        |  
| Glass Food Jars              | Magazines                       |  
| Newspaper                    | Office Paper                    |  
| Paper Cups                   | Plastic Cup Lids                |  

---

### **Chatbot Guide**  
- **Interactive Assistance**: Provides clear instructions on how to recycle items.  
- **Educational Content**: Shares best practices and explains the recycling process for different materials.  
- **User-Friendly Interface**: Ensures accessibility with intuitive interaction.  

---

## Usage  

1. **Upload an Image**: Upload an image of the item you wish to recycle.  
2. **Get Classification**: The system identifies the item category and provides a confidence score.  
3. **Receive Guidance**: The chatbot delivers step-by-step instructions for recycling, including any region-specific considerations.  

---

## Technical Details  

### **Model Training**  
- The classification model is based on **ResNet18** and fine-tuned on a dataset of recyclable items.  
- It achieves high accuracy with normalized ImageNet weights and a custom prediction head.  

### **Inference**  
The `classify()` function:  
- Takes image bytes as input.  
- Processes the image using transformations (resize, crop, normalize).  
- Returns the predicted class, its index, and confidence score.  

### **Chatbot**  
- Built using **Google's GEMMA-1.1-2b** and fine-tuned with LoRA for better context understanding in recycling-related queries.  
- Supports interactive chat sessions to deliver detailed recycling instructions.  

### **API**  
Developed using **FastAPI** to provide endpoints for:  
- **Health Check**: Verify API availability (`/health`).  
- **Upload Endpoint**: Upload an image and receive classification details along with metadata (`/upload`).  

### **Tech Stack**  
- **Backend**: FastAPI  
- **Modeling**: PyTorch, HuggingFace Transformers  
- **Preprocessing**: torchvision  
- **Deployment**: Docker-ready for easy scaling  

---

## Installation  

1. **Clone Repository**  
   ```bash  
   git clone https://github.com/yourusername/recyclenet.git  
   cd recyclenet  
   ```  

2. **Install Dependencies**  
   ```bash  
   pip install -r requirements.txt  
   ```  

3. **Run Locally**  
   ```bash  
   uvicorn app.main:app --reload  
   ```  

4. **Access API**  
   Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for interactive API documentation.  

---

## Future Enhancements  
- **Expand Classification Categories**: Incorporate more recyclable materials.  
- **Localization**: Support region-specific recycling rules.  
- **Advanced Chatbot Features**: Improve interaction for more personalized guidance.  

---

## License  
This project is licensed under the [MIT License](LICENSE).  

![image](https://github.com/user-attachments/assets/e773799f-f44d-4362-8695-acc4e7229eda)
![image](https://github.com/user-attachments/assets/25fdbe08-193f-4074-b312-e2d4652236d7)
![image](https://github.com/user-attachments/assets/8e1adcc9-fa9d-4a57-a459-0e9d7227d5d6)

