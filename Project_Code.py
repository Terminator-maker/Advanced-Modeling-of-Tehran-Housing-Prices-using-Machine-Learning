import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, ElasticNet, LinearRegression, LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import r2_score, confusion_matrix, accuracy_score
import xgboost as xgb

# 1. Load Data
df = pd.read_csv('41599_2024_3244_MOESM1_ESM.csv')

# Basic Preprocessing
df_clean = df[df['total_value'] > 0].copy()
df_clean = df_clean[df_clean['area'] <= 200]
df_clean = df_clean[df_clean['year'] >= 1374]

# Log-transform target for stability
df_clean['log_price'] = np.log1p(df_clean['total_value'])

X = df_clean[['area', 'year', 'neighborhood', 'elavator', 'parking']]
y = df_clean['log_price']

# 2. Pipeline Setup
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['area', 'year']),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['neighborhood']),
        ('passthrough', 'passthrough', ['elavator', 'parking'])
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- SECTION 2: REGRESSION ALGORITHMS FIGURES ---

models = {
    'Lasso': Lasso(alpha=0.001), 
    'Elastic Net': ElasticNet(alpha=0.001, l1_ratio=0.5),
    'Kernel Ridge': KernelRidge(alpha=1.0, kernel='linear'), 
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, objective='reg:squarederror', random_state=42)
}

for name, model in models.items():
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    
    if name == 'Kernel Ridge': 
        clf.fit(X_train[:2000], y_train[:2000])
    else: 
        clf.fit(X_train, y_train)
    
    y_pred_log = clf.predict(X_test)
    y_pred_actual = np.expm1(y_pred_log)
    y_test_actual = np.expm1(y_test)
    
    # Plotting
    plt.figure(figsize=(7, 6))
    indices = np.random.choice(len(y_test), size=min(500, len(y_test)), replace=False)
    
    plt.scatter(y_test_actual.iloc[indices], y_pred_actual[indices], alpha=0.6, edgecolors='w', s=40, color='royalblue')
    
    min_val = min(y_test_actual.iloc[indices].min(), y_pred_actual[indices].min())
    max_val = max(y_test_actual.iloc[indices].max(), y_pred_actual[indices].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Fit')
    
    plt.title(name, fontsize=14)
    plt.xlabel("Actual Price (Tomans)")
    plt.ylabel("Predicted Price (Tomans)")
    
    r2 = r2_score(y_test_actual, y_pred_actual)
    plt.text(0.05, 0.9, f"$R^2$ = {r2:.2f}", transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'Fig_{name.replace(" ", "_")}.png')
    plt.close()

# --- SECTION 4: IMPROVEMENT FIGURES ---

# 4.1 Polynomial Fit Visualization
df_poly = df[df['total_value'] > 0]
df_viz = df_poly[df_poly['total_value'] < df_poly['total_value'].quantile(0.95)]
df_viz = df_viz[df_viz['area'] <= 300] 

X_area = df_viz[['area']].values
y_price = df_viz['total_value'].values

lin_reg = LinearRegression()
lin_reg.fit(X_area, y_price)

poly_reg = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly_reg.fit(X_area, y_price)

X_seq = np.linspace(X_area.min(), X_area.max(), 300).reshape(-1, 1)
y_lin_pred = lin_reg.predict(X_seq)
y_poly_pred = poly_reg.predict(X_seq)

plt.figure(figsize=(10, 6))
plt.scatter(X_area, y_price, color='lightgray', alpha=0.6, label='Actual Data (95% percentile)')
plt.plot(X_seq, y_lin_pred, color='red', linewidth=3, label='Linear Model (Lasso style)')
plt.plot(X_seq, y_poly_pred, color='blue', linestyle='--', linewidth=3, label='Polynomial Model (Proposed)')

plt.xlabel('Area ($m^2$)')
plt.ylabel('Price (Tomans)')
plt.title('Corrected: Linear vs. Polynomial Fit on Housing Data')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Fig7_Polynomial_Fit.png')
plt.close()

# 4.3 Classification Heatmap
df['price_tier'] = pd.qcut(df['total_value'], q=3, labels=['Affordable', 'Moderate', 'Luxury'])
X_cls = df[['area', 'year']] 
y_cls = df['price_tier']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
clf_log = make_pipeline(StandardScaler(), LogisticRegression())
clf_log.fit(X_train_c, y_train_c)
y_pred_cls = clf_log.predict(X_test_c)

cm = confusion_matrix(y_test_c, y_pred_cls, labels=['Affordable', 'Moderate', 'Luxury'])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Affordable', 'Moderate', 'Luxury'],
            yticklabels=['Affordable', 'Moderate', 'Luxury'])
plt.xlabel('Predicted Tier')
plt.ylabel('Actual Tier')
plt.title(f'Classification Performance (Accuracy: {accuracy_score(y_test_c, y_pred_cls):.2f})')
plt.savefig('Fig8_Classification_Matrix.png')
plt.close()

