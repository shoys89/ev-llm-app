# 🚗🔋 EV Charging Assistant — LLM + ANN Prediction

This project provides an intelligent assistant capable of analyzing EV charging sessions using **LLMs (Groq + Qwen3)** and a **neural network model** trained on EV charging patterns.  
It extracts structured information from natural language, identifies the vehicle from the EV-DB dataset, computes physical properties of the charge, and predicts the **energy consumed (kWh)** using a model hosted on Hugging Face.

The full application runs locally and exposes a **Streamlit chatbot UI** where users can interact naturally.

---

# 🚀 Getting Started

## ✔ Requirements

- **Python 3.11**
- macOS users **must** install a version of Python that supports TensorFlow  
  (ANN was originally trained with TF)
- HuggingFace token + Groq API key
- Recommended: virtual environment

---

# ⚙ 1. Create and Activate Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

If using Windows PowerShell:

```powershell
venv\Scripts\activate
```

---

# 📦 2. Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

---

# 🍏 Practical Hack for macOS (TensorFlow + Keras Compatibility)

TensorFlow versions on macOS (especially ARM/M1/M2 chips) often conflict with modern Keras.  
Use this **minimal hack** so your ANN model loads successfully.

### 🔧 Step 1 — Remove any preinstalled Keras

```bash
pip uninstall -y keras
```

### 🔧 Step 2 — Install Keras 3 manually (no dependency resolution)

```bash
pip install keras==3.3.3 --no-deps
```

### 🔧 Step 3 — Install missing TensorFlow dependency (`optree`)

```bash
pip install optree
```

These steps allow your HuggingFace snapshot's ANN model to load without errors.

---

# 🔑 3. Environment Variables

Create a simple `env.sh` file:

```bash
export HF_TOKEN="your-hf-token-here"
export GROQ_API_KEY="your-groq-key-here"
```

Load it before running the app:

```bash
source env.sh
```

---

# 🖥 4. Run the Application (Streamlit UI)

```bash
streamlit run src/ui/streamlit_llm_chat.py
```

This launches the EV Assistant at:

➡ **http://localhost:8501**

---

# 💬 Example Questions

Try natural language queries such as:

> “Tengo un Abarth 500e Hatchback de 2023, lo cargué de 20% a 60% y tardó 1.5 horas.”

The system will:

1. Extract vehicle + session info  
2. Match the car inside the EV-DB dataset  
3. Compute SoC difference, charging rate, energy estimate  
4. Ask follow-up questions if needed  
5. Run the ANN model to predict **energy consumed**  
6. Generate an LLM explanation in Spanish  

---

# 📁 Project Structure (Simplified)

```
project/
 ├── src/
 │   ├── model/               # ANN model loading + HuggingFace snapshot
 │   ├── pipeline/            # LLM extraction + EV-DB matching
 │   └── ui/
 │       ├── streamlit_llm_chat.py   # Chatbot UI
 │       └── ...
 ├── data/
 │   └── EV-DB.csv            # Vehicle specifications database
 ├── requirements.txt
 ├── env.sh
 └── README.md
```

---

# 🛠 Troubleshooting

### ❌ The vehicle is not found in EV-DB  
Use a fuzzy-matching implementation in `find_vehicle_row()`.

### ❌ Import errors involving TF/Keras  
Use the macOS fix above — Keras/TensorFlow compatibility is strict on Apple Silicon.

### ❌ LLM does not respond  
Ensure your environment variables are loaded correctly:

```bash
echo $HF_TOKEN
echo $GROQ_API_KEY
```

---

# 🎉 You're Ready to Go!
