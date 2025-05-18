from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
from PIL import Image
import os
from config import Config
from werkzeug.utils import secure_filename


class AssistantService:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        self.model = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16
        )
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

    def process_query(self, question, image_file=None):
        image = None
        if image_file and image_file.filename != "":
            filename = secure_filename(image_file.filename)
            temp_path = os.path.join(Config.UPLOAD_FOLDER, filename)
            os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
            image_file.save(temp_path)

            image = Image.open(temp_path)

            os.remove(temp_path)

        if image:
            inputs = self.processor(images=image, text=question, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
        else:
            inputs = self.processor(text=question, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

        outputs = self.model.generate(**inputs, max_length=200, num_beams=4)
        response = self.processor.decode(outputs[0], skip_special_tokens=True)

        return {
            "question": question,
            "response": response,
            "has_image": image is not None,
        }
