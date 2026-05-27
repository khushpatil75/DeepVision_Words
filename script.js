let base64String = "";

// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const previewSection = document.getElementById('previewSection');
const imagePreview = document.getElementById('imagePreview');
const fileNameDisplay = document.getElementById('fileName');
const predictBtn = document.getElementById('predictBtn');
const resultBox = document.getElementById('resultBox');
const predictionText = document.getElementById('predictionText');

// --- Drag and Drop Logic ---
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        handleFileSelect(fileInput.files[0]);
    }
});

fileInput.addEventListener('change', function(event) {
    if (event.target.files.length) {
        handleFileSelect(event.target.files[0]);
    }
});

// --- File Processing Logic ---
function handleFileSelect(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload a valid image file.');
        return;
    }

    fileNameDisplay.textContent = file.name;
    const reader = new FileReader();
    
    reader.onload = function(e) {
        base64String = e.target.result;
        imagePreview.src = base64String;
        
        // Update UI state
        previewSection.style.display = 'block';
        predictBtn.disabled = false;
        resultBox.style.display = 'none'; 
        dropZone.style.border = '2px dashed #cbd5e1';
    };
    reader.readAsDataURL(file);
}

// Bind the button to the prediction function
predictBtn.addEventListener('click', predictSentence);

// --- API Request Logic ---
async function predictSentence() {
    if (!base64String) return;
    
    // Set loading state
    predictBtn.innerHTML = `<span>Analyzing Sequence...</span>`;
    predictBtn.disabled = true;

    try {
        const response = await fetch('http://localhost:8000/predict_sentence', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: base64String })
        });

        const data = await response.json();

        if (response.ok) {
            resultBox.style.display = 'block';
            predictionText.innerText = data.prediction;
            
            // Scroll to result slightly for better UX on mobile
            resultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        } else {
            alert("Model Error: " + data.detail);
        }
    } catch (error) {
        console.error("Error:", error);
        alert("Connection failed. Ensure the FastAPI backend is running on port 8000.");
    } finally {
        // Reset button state
        predictBtn.innerHTML = `<span>Upload & Predict Sequence</span>`;
        predictBtn.disabled = false;
    }
}