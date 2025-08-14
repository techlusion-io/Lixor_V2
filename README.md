# Lixor_V2

## About
Lixor_V2 is an updated version of the Lixor (Detecting liquor in the bottle) analysis pipeline.  
It uses a **Fully Convolutional Network (FCN)** to segment and evaluate chemical images, identifying features such as liquid levels, vessels, corks, and the thick base of the bottle and segements the area.  
We then use the segements created by the FCN model for Cork, Vessel and the Liquid and use those as the factors to calculate the % of liquor in the bottle.
Used Yolo 11m-seg model to detect the thick bases for the bottles and eliminate them form the bottle's total area.

---

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/techlusion-io/Lixor_V2.git
cd Lixor_V2
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the APP.PY file (that's all)
```bash
streamlit run app.py
```

---

## Deployed on Streamlit for demo

URL : https://lixornewv2.streamlit.app/
