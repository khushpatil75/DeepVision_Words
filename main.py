from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import base64
import io
import json

app = FastAPI(title="DeepVision AI - Sequence Recognition")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. LOAD CTC VOCABULARY ---
print("Loading CTC vocabulary...")
try:
    with open('ctc_vocab.json', 'r', encoding='utf-8') as f:
        idx_to_char = json.load(f)
    # Ensure keys are integers
    idx_to_char = {int(k): v for k, v in idx_to_char.items()}
    num_classes = len(idx_to_char)
    print(f"Successfully loaded {num_classes} classes.")
except Exception as e:
    print(f"Error loading ctc_vocab.json: {e}")
    num_classes = 109 # Fallback based on your recent training output

# --- 2. REBUILD CTC HYBRID ARCHITECTURE ---
class HybridCTCModel(nn.Module):
    def __init__(self, num_classes):
        super(HybridCTCModel, self).__init__()
        resnet = timm.create_model('resnet18', pretrained=False)
        self.cnn = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.act1, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3
        )
        self.sequence_builder = nn.Conv2d(256, 256, kernel_size=(4, 1)) 
        
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.cnn(x)                
        x = self.sequence_builder(x)   
        x = x.squeeze(2).permute(0, 2, 1) 
        x = self.transformer(x)
        x = self.fc(x)                 
        return x.log_softmax(2).permute(1, 0, 2)

# Load Weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridCTCModel(num_classes).to(device)
try:
    model.load_state_dict(torch.load('hybrid_ctc_model.pth', map_location=device))
    model.eval() 
    print("Model weights loaded successfully.")
except Exception as e:
    print(f"Error loading weights: {e}")

# --- 3. DATA TRANSFORMS (Must match training!) ---
transform = transforms.Compose([
    transforms.Resize((64, 384)), # Resizing to the exact dimensions the model expects
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 4. CTC GREEDY DECODER ---
def decode_predictions(predictions):
    """
    Translates the model's output sequence back to text.
    CTC Rule 1: Remove consecutive duplicate predictions.
    CTC Rule 2: Remove the <BLANK> token (Index 0).
    """
    predicted_text = []
    previous_char_idx = -1
    
    for idx in predictions:
        idx = idx.item()
        # If the index is not a blank (0) and is not a duplicate of the immediate previous step
        if idx != 0 and idx != previous_char_idx:
            predicted_text.append(idx_to_char[idx])
        previous_char_idx = idx
        
    return "".join(predicted_text)

# --- 5. API ENDPOINT ---
class ImageData(BaseModel):
    image: str 

@app.post("/predict_sentence")
async def predict_sentence(data: ImageData):
    try:
        # Decode Base64 string directly into a PIL Image
        image_bytes = base64.b64decode(data.image.split(",")[1])
        pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Apply transforms and add batch dimension [1, 3, 64, 384]
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        # Run Inference
        with torch.no_grad():
            output = model(input_tensor) # Shape: [24, 1, Num_Classes]
            
            # Get the index with the highest probability at each of the 24 time steps
            _, max_indices = torch.max(output, 2) # Shape: [24, 1]
            
            # Squeeze to a 1D list of 24 numbers
            predictions = max_indices.squeeze(1) 
            
        # Decode the numbers into Devanagari text
        final_word = decode_predictions(predictions)
        
        return {
            "status": "success", 
            "prediction": final_word
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))