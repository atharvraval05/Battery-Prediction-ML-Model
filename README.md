# VoltAI - Interactive Mobile Battery Life Predictor

VoltAI is a clean, interactive, and modern web application that uses simple Linear Regression to predict remaining mobile battery life based on Screen-On Time (SOT).

> [!NOTE]
> 💡 **Project Background**:
> - **This is my first Machine Learning project!**
> - The core machine learning regression logic (fitting the model, custom data retraining, and predicting values using scikit-learn) is **mine (written by me)**.
> - The interactive frontend layout, pastel visual styling, wave simulation physics, and synthesized Web Audio sound effects were built by my AI pair-programming partner, **Antigravity** (created by the Google DeepMind team).

---

## 🎨 Key Features

1. **Active Linear Regression Engine**:
   - Built on Python's `scikit-learn` library.
   - Retrains the model dynamically in the backend when you inject custom SOT and battery coordinates.
   - Outputs statistical metrics like $R^2$, slope, intercept, and sample sizes live.

2. **Refreshing Pastel Interface**:
   - Symmetrical, rounded two-column dashboard using the clean `Outfit` and `Space Grotesk` fonts.
   - Light, pastel lilac-mint-peach aurora gradient background.
   - Fully interactive graphs detailing SOT vs Battery Remaining, SOT vs Battery Consumed, and SOT vs Model Residuals/Simulation Variance.

3. **High-Fidelity Mobile Mockup**:
   - Floating 3D phone chassis that responds to your cursor's hover direction.
   - SVG battery wave liquid fill that dynamically shifts colors based on charge:
     - **80% - 100%**: Soft Sage Green
     - **50% - 79%**: Calm Pastel Blue
     - **21% - 49%**: Warm Pastel Peach-Orange
     - **0% - 20%**: Soft Pastel Crimson-Red (Triggers low battery notifications)
   - Clamped fluid wave rendering ($8\%$ to $78\%$) ensuring the liquid animation remains visible at all charge levels.

4. **Synthesized Sound Effects (SFX)**:
   - Uses the browser's native **Web Audio API** (zero-dependency synthesizer) to play:
     - Slider ticks as you drag SOT.
     - Toggle keyclicks when modifiers are checked.
     - Success chords when custom coordinates retrain the model.
     - Dual-tone low-battery warnings.

---

## 🚀 Setup & Installation

### 1. Prerequisites
Make sure you have Python 3.10+ installed on your system.

### 2. Install Dependencies
Install all the required python libraries using pip:
```bash
pip install -r requirements.txt
```

### 3. Run the Server
Launch the FastAPI development server:
```bash
python app.py
```

### 4. Open in Browser
Open your browser and navigate to the local host address:
👉 **[http://localhost:8000](http://localhost:8000)**

---

## 📂 Repository Structure

- `app.py`: FastAPI server handling dataset storage, sklearn regression pipelines, and API services.
- `index.html`: Unified frontend holding all semantic markup, pastel style variables, SVG animations, and JS audio controllers.
- `Mobile_Battery_Life_Prediction_Dataset.csv`: Original 80-coordinate training dataset.
- `requirements.txt`: Python package requirements list.
- `.gitignore`: Excludes Python cache, system files, and local persistence datasets.
"# Battery-Prediction-ML-Model" 
