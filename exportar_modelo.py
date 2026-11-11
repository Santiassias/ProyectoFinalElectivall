import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Módulos de Scikit-Learn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    roc_curve,
    roc_auc_score
)

# Conector personalizado a MySQL
from src.database.connection2 import MySQLConnector


def main():
    print("=================================================================")
    print("  ACTIVIDAD FINAL: ENTRENAMIENTO, OPTIMIZACIÓN Y SERIALIZACIÓN")
    print("=================================================================\n")

    # -----------------------------------------------------------------------
    # FASE 1: INGESTA Y PREPROCESAMIENTO
    # -----------------------------------------------------------------------
    db = MySQLConnector()

    query = """
        SELECT
            p.player_id,
            p.name AS jugador,
            p.position AS posicion,
            p.height_in_cm AS estatura_cm,
            TIMESTAMPDIFF(YEAR, p.date_of_birth, CURDATE()) AS edad,
            p.market_value_in_eur AS valor_mercado
        FROM players p
        WHERE p.market_value_in_eur IS NOT NULL
          AND p.date_of_birth IS NOT NULL;
    """

    print("[FASE 1] Extrayendo datos desde la base MySQL...")
    df = db.extract_data(query)

    df['estatura_cm'] = pd.to_numeric(df['estatura_cm'], errors='coerce')
    df['edad'] = pd.to_numeric(df['edad'], errors='coerce')
    df['valor_mercado'] = pd.to_numeric(df['valor_mercado'], errors='coerce')
    df['posicion'] = df['posicion'].fillna('Sin Posicion')

    # Guardar las medianas de imputación para reutilizar en producción
    medianas_estatura = df.groupby('posicion')['estatura_cm'].median().to_dict()
    mediana_global_estatura = df['estatura_cm'].median()
    mediana_global_edad = df['edad'].median()

    # Imputación por mediana agrupada (posición)
    df['estatura_cm'] = df.groupby('posicion')['estatura_cm'].transform(lambda x: x.fillna(x.median()))
    df['estatura_cm'] = df['estatura_cm'].fillna(mediana_global_estatura)
    df['edad'] = df['edad'].fillna(mediana_global_edad)

    # Target: Élite (Top 25% Valor de Mercado)
    umbral_elite = df['valor_mercado'].quantile(0.75)
    df['es_elite'] = (df['valor_mercado'] >= umbral_elite).astype(int)

    # Matriz X (One-Hot Encoding) y Vector y
    df_dummies = pd.get_dummies(df[['posicion']], drop_first=True, dtype=int)
    X = pd.concat([df[['edad', 'estatura_cm']], df_dummies], axis=1)
    y = df['es_elite']

    # Depuración Anti-NaN
    filas_con_nan = X.isna().any(axis=1)
    if filas_con_nan.sum() > 0:
        X = X[~filas_con_nan]
        y = y[~filas_con_nan]

    # División 80/20 Estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[OK] Preprocesamiento completado. Registros procesados: {X.shape[0]}.")
    print(f"     Umbral Élite (P75): {umbral_elite:,.0f} EUR")
    print(f"     Clase 0 (Estándar): {np.sum(y==0)} | Clase 1 (Élite): {np.sum(y==1)}\n")

    # -----------------------------------------------------------------------
    # FASE 2: DIAGNÓSTICO DE OVERFITTING (ÁRBOL SIN CONTROL)
    # -----------------------------------------------------------------------
    print("[FASE 2] Evaluando Árbol Sin Control (Diagnóstico de Overfitting)...")
    tree_libre = DecisionTreeClassifier(max_depth=None, random_state=42)
    tree_libre.fit(X_train, y_train)

    acc_train_libre = accuracy_score(y_train, tree_libre.predict(X_train))
    acc_test_libre = accuracy_score(y_test, tree_libre.predict(X_test))
    brecha_libre = acc_train_libre - acc_test_libre

    print(f"  • Accuracy Train: {acc_train_libre:.4f} | Accuracy Test: {acc_test_libre:.4f}")
    print(f"  • Brecha Train - Test: {brecha_libre:.4f} ({brecha_libre*100:.2f}%)\n")

    # -----------------------------------------------------------------------
    # FASE 3: PODA PREVIA (PRE-PRUNING) Y PARADOJA DE LA EXACTITUD
    # -----------------------------------------------------------------------
    print("[FASE 3] Aplicando Poda Previa en Árbol de Decisión...")
    tree_podado = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42)
    tree_podado.fit(X_train, y_train)

    acc_train_podado = accuracy_score(y_train, tree_podado.predict(X_train))
    acc_test_podado = accuracy_score(y_test, tree_podado.predict(X_test))
    f1_podado = f1_score(y_test, tree_podado.predict(X_test), pos_label=1)

    print(f"  • Accuracy Train: {acc_train_podado:.4f} | Accuracy Test: {acc_test_podado:.4f}")
    print(f"  • F1-Score Clase Élite: {f1_podado:.4f}  <- Paradoja de la Exactitud\n")

    # Árbol podado con penalización por pesos de costo (Guía 6)
    tree_model = DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=42)
    tree_model.fit(X_train, y_train)
    f1_tree = f1_score(y_test, tree_model.predict(X_test), pos_label=1)
    print(f"  • Árbol podado + class_weight='balanced' -> F1-Score Clase Élite: {f1_tree:.4f}\n")

    # -----------------------------------------------------------------------
    # FASE 4: RANDOM FOREST OPTIMIZADO CON GRIDSEARCHCV
    # -----------------------------------------------------------------------
    print("[FASE 4] Optimización del Random Forest con GridSearchCV...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6, 8],
        'min_samples_leaf': [5, 10]
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
        param_grid=param_grid,
        cv=5,
        scoring='f1'
    )
    grid_search.fit(X_train, y_train)

    rf_model = grid_search.best_estimator_
    y_pred_rf = rf_model.predict(X_test)
    acc_train_rf = accuracy_score(y_train, rf_model.predict(X_train))
    acc_test_rf = accuracy_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf, pos_label=1)

    print(f"  • Mejores Hiperparámetros: {grid_search.best_params_}")
    print(f"  • Accuracy Train: {acc_train_rf:.4f} | Accuracy Test: {acc_test_rf:.4f}")
    print(f"  • F1-Score Optimizado en Test: {f1_rf:.4f}\n")
    print(classification_report(y_test, y_pred_rf, target_names=['Estándar', 'Élite']))

    # -----------------------------------------------------------------------
    # FASE 5: CURVA ROC-AUC E IMPORTANCIA DE VARIABLES
    # -----------------------------------------------------------------------
    y_prob_tree = tree_model.predict_proba(X_test)[:, 1]
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

    auc_tree = roc_auc_score(y_test, y_prob_tree)
    auc_rf = roc_auc_score(y_test, y_prob_rf)

    print("=================================================")
    print("  COMPARATIVA ROC-AUC (CAPACIDAD DE DISCRIMINACIÓN)")
    print("=================================================")
    print(f"  • Árbol de Decisión Único ROC-AUC: {auc_tree:.4f}")
    print(f"  • Random Forest Classifier ROC-AUC: {auc_rf:.4f}")
    print("=================================================\n")

    importancias = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("Importancia de Variables según Random Forest:")
    print(importancias)
    print("\n")

    # -----------------------------------------------------------------------
    # FASE 6: SERIALIZACIÓN DEL PAQUETE DE DESPLIEGUE (JOBLIB)
    # -----------------------------------------------------------------------
    print("[FASE 6] Entrenando modelo final y exportando paquete...")
    rf_final = RandomForestClassifier(
        **grid_search.best_params_, class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf_final.fit(X, y)

    paquete_despliegue = {
        'modelo': rf_final,
        'columnas_X': X.columns.tolist(),
        'medianas_estatura': medianas_estatura,
        'mediana_global_estatura': mediana_global_estatura,
        'mediana_global_edad': mediana_global_edad,
        'umbral_elite_eur': umbral_elite
    }

    os.makedirs('models', exist_ok=True)
    ruta_modelo = 'models/modelo_scouting_rf.joblib'
    joblib.dump(paquete_despliegue, ruta_modelo)
    print(f"[OK] Paquete de modelo exportado exitosamente en: '{ruta_modelo}'\n")

    # -----------------------------------------------------------------------
    # FASE VISUAL: PANEL 1 - DIAGNÓSTICO (Overfitting, desbalance, matrices)
    # -----------------------------------------------------------------------
    print("[FASE 7] Generando paneles de gráficas...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Gráfico 1: Diagnóstico de Overfitting
    etapas = ['Árbol Sin Límite', 'Árbol Podado', 'Random Forest']
    train_accs = [acc_train_libre, acc_train_podado, acc_train_rf]
    test_accs = [acc_test_libre, acc_test_podado, acc_test_rf]
    x = np.arange(len(etapas))
    width = 0.35
    axes[0, 0].bar(x - width/2, train_accs, width, label='Train Accuracy', color='#e74c3c', edgecolor='black')
    axes[0, 0].bar(x + width/2, test_accs, width, label='Test Accuracy', color='#3498db', edgecolor='black')
    axes[0, 0].set_title('1. Diagnóstico de Overfitting y Efecto de Poda', fontsize=11, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(etapas)
    axes[0, 0].set_ylim(0, 1.1)
    axes[0, 0].legend(loc='lower right')
    axes[0, 0].grid(True, linestyle='--', alpha=0.5, axis='y')

    # Gráfico 2: Desbalance de Clases
    conteo_clases = y.value_counts().sort_index()
    axes[0, 1].bar(['Estándar (0)', 'Élite (1)'], conteo_clases.values, color=['#2980b9', '#e67e22'], edgecolor='black')
    axes[0, 1].set_title('2. Inspección del Desbalance de Clases', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Cantidad de Jugadores', fontsize=10)
    for i, v in enumerate(conteo_clases.values):
        axes[0, 1].text(i, v + 1, str(v), ha='center', fontweight='bold')

    # Gráfico 3: Matriz Confusión Árbol Sin Límite
    cm_libre = confusion_matrix(y_test, tree_libre.predict(X_test))
    ConfusionMatrixDisplay(confusion_matrix=cm_libre, display_labels=['Estándar', 'Élite']).plot(ax=axes[1, 0], cmap='Reds', colorbar=False)
    axes[1, 0].set_title('3. Matriz Confusión: Árbol Sin Control', fontsize=11, fontweight='bold')

    # Gráfico 4: Matriz Confusión Random Forest Optimizado
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=['Estándar', 'Élite']).plot(ax=axes[1, 1], cmap='Greens', colorbar=False)
    axes[1, 1].set_title('4. Matriz Confusión: Random Forest Optimizado', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('resultado_actividad_final.png', dpi=300)
    print("[OK] Panel guardado como 'resultado_actividad_final.png'.")

    # -----------------------------------------------------------------------
    # FASE VISUAL: PANEL 2 - CURVAS ROC E IMPORTANCIA (GUÍA 6)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    fpr_tree, tpr_tree, _ = roc_curve(y_test, y_prob_tree)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    axes[0].plot(fpr_tree, tpr_tree, color='#e67e22', linestyle='--', linewidth=2,
                 label=f'Árbol Único (AUC = {auc_tree:.3f})')
    axes[0].plot(fpr_rf, tpr_rf, color='#27ae60', linewidth=2.5,
                 label=f'Random Forest (AUC = {auc_rf:.3f})')
    axes[0].plot([0, 1], [0, 1], color='gray', linestyle=':', label='Clasificador Aleatorio (AUC = 0.500)')
    axes[0].set_title('Comparativa de Curvas ROC', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Tasa de Falsos Positivos (FPR - Falsas Alarmas)', fontsize=10)
    axes[0].set_ylabel('Tasa de Verdaderos Positivos (TPR - Recall)', fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(loc='lower right', frameon=True)

    top_importancias = importancias.head(6).sort_values(ascending=True)
    axes[1].barh(top_importancias.index, top_importancias.values, color='#2980b9', edgecolor='black', alpha=0.85)
    axes[1].set_title('Top Importancia de Variables (Random Forest)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Importancia Relativa (Gini Impurity)', fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.6, axis='x')

    plt.tight_layout()
    plt.savefig('evaluacion_rf_vs_tree.png', dpi=300)
    print("[OK] Panel guardado como 'evaluacion_rf_vs_tree.png'.")
    plt.show()


if __name__ == "__main__":
    main()
