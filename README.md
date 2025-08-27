
# 📚 Real-Time Learning Recommendation System (KNN vs Deep Learning)

---

This project implements a **real-time personalized learning recommendation system** using both **K-Nearest Neighbors (KNN)** and **Deep Learning models**.  

The system predicts whether a learner should be recommended **practice content** or **advanced material** based on activity data such as time of activity, platform, user answers, and actions. It also includes an interactive **Streamlit web application** for real-time recommendations.

---
## Background

This project was originally developed in **April 2025** as a freelance project for a client.  
I am now publishing it on GitHub for **portfolio and reference purposes**.  

All code, training, and implementation were done **independently by me from scratch**.

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

## Demo Video

You can download and watch the demo here:  
[Recommendation System Demo](recommendation_system_app_demo_with_KNN+NN.mkv)


## Project Structure
```

Learning-Recommendation-System/
│── data/ # Dataset files (CSV)
│── models/ # Trained models (KNN, Deep Learning)
│── visualization/ # Saved plots and charts
│ │── activity_over_time.png
│ │── hourly_activity.png
│ │── weekly_activity.png
│── app.py # Streamlit application
│── model_training.py # Model training and preprocessing script
│── model_comparison.csv # Accuracy/F1 score comparison table
│── final_output_with_recommendations.csv # Final recommendations output
│── requirements.txt # Python dependencies
│── Project Summary for EDNet K3.docx # Project summary document
│── student activity rec system.docx # Detailed documentation
│── lecture.mkv # Demo/recorded lecture file
│── README.md # Project documentation

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

