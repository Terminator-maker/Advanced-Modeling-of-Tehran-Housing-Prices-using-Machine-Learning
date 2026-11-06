# 🏠 Advanced Modeling of Tehran Housing Prices using Machine Learning

A comprehensive machine learning project and critical review of housing price prediction models in Tehran, leveraging both regression and classification techniques.

## 📊 Project Overview
This project analyzes a real-world dataset of residential properties across Tehran to predict housing values. It implements and compares multiple machine learning algorithms, evaluates their limitations (such as linear assumptions and non-standard metrics), and proposes methodological enhancements like **Polynomial Regression** and **Price Tier Classification**.

## 🛠️ Tech Stack & Libraries
* **Python**
* **Scikit-Learn** (Lasso, Elastic Net, Kernel Ridge, Random Forest, Logistic Regression)
* **XGBoost** (Extreme Gradient Boosting)
* **Pandas & NumPy** (Data Preprocessing & Manipulation)
* **Matplotlib & Seaborn** (Data Visualization & Heatmaps)

## 🔍 Key Models & Findings
1. **Lasso Regression:** Chosen in the original study for its interpretability and formula simplicity, though it suffers from linearity bias in luxury markets[cite: 29, 31, 38].
2. **Kernel Ridge & Gradient Boosting:** Captured non-linear price patterns more effectively[cite: 34, 35, 38].
3. **XGBoost:** Achieved the highest predictive accuracy by capturing complex feature-neighborhood interactions[cite: 36, 38].
4. **Proposed Improvements:**
   * **Polynomial Regression:** To better model the exponential price surge in luxury homes[cite: 40].
   * **Price Tier Classification:** Converting exact price prediction into robust categories (*Affordable, Moderate, Luxury*) to handle economic volatility[cite: 42].

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/Terminator-maker/Advanced-Modeling-of-Tehran-Housing-Prices-using-Machine-Learning.git](https://github.com/Terminator-maker/Advanced-Modeling-of-Tehran-Housing-Prices-using-Machine-Learning.git)
   cd Advanced-Modeling-of-Tehran-Housing-Prices-using-Machine-Learning
   Install dependencies:

Bash
pip install -r requirements.txt 
Run the script: 

Bash
python Project_Code.py
