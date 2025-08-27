```markdown
# 📚 Real-Time Learning Recommendation System (KNN vs Deep Learning)

This project implements a **real-time personalized learning recommendation system** using both **K-Nearest Neighbors (KNN)** and **Deep Learning models**.  

The system predicts whether a learner should be recommended **practice content** or **advanced material** based on activity data such as time of activity, platform, user answers, and actions. It also includes an interactive **Streamlit web application** for real-time recommendations.

---

## Note

This project was originally developed in April 2025 for one of my freelance clients.
I am now pushing it to my GitHub repository at a later stage for portfolio and reference purposes.

All code and implementation were done by me from scratch for the client, and this public repo is meant to showcase my work.


---
## Features
- **Data Preprocessing**: Merging multiple CSV datasets, handling missing values, encoding categorical features, and feature engineering.  
- **Model Training**:
  - **KNN Classifier** – Traditional ML approach.
  - **Deep Learning (TensorFlow/Keras)** – Neural network for capturing non-linear patterns.
- **Model Evaluation**:
  - Accuracy and F1 Score comparison.
  - Saved results in a `model_comparison.csv`.
- **Streamlit App**:
  - Real-time personalized recommendations.
  - Interactive sidebar for input (platform, action, user response, etc.).
  - Comparison table and visualization charts.
- **Visualizations**:
  - Activity over time.
  - Activity by hour.
  - Activity by day of the week.

---

## Project Structure
```

Learning-Recommendation-System/
│── data/                        # Raw dataset files (CSV)
│── models/                      # Trained models (KNN, Deep Learning)
│── app.py                       # Streamlit application
│── model\_training.py             # Model training and preprocessing script
│── model\_comparison.csv         # Accuracy/F1 score comparison table
│── hourly\_activity.png          # Activity by hour visualization
│── weekly\_activity.png          # Activity by day visualization
│── activity\_over\_time.png       # Activity timeline
│── README.md                    # Project documentation

````

---

## Installation & Requirements
Make sure you have Python 3.x installed. Install dependencies with:

```bash
pip install -r requirements.txt
````


---

## Usage

### 1. Train Models

Run the training script to preprocess data, train models, and generate comparison files:

```bash
python model_training.py
```

### 2. Launch Streamlit App

Start the web app:

```bash
streamlit run app.py
```

* Select model (**KNN** or **Deep Learning**)
* Choose user activity features from the sidebar
* Get **real-time recommendations** instantly

---

## Results & Visualizations

### Model Comparison

| Model         | Accuracy | F1 Score |
| ------------- | -------- | -------- |
| KNN           | xx%      | xx%      |
| Deep Learning | xx%      | xx%      |

### Example Graphs

* Activity Over Time
* Activity by Hour
* Activity by Day of Week

---

## Applications

* Personalized e-learning platforms.
* Recommendation engines for ed-tech apps.
* Understanding student activity patterns.

---

## 🤝 Contributing

Pull requests are welcome! Feel free to open issues for suggestions or improvements.

---

## License

This project is licensed under the **MIT License**.

