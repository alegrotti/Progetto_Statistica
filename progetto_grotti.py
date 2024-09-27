"""
Progetto Grotti Alessandro - Statistica numerica

Dataset : Economic Freedom of the World

Variabile target di previsione : Economic freedom index

Varibili input : size_government, property_rights, sound_money, trade, regulation
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection
#from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
import seaborn as sns
from sklearn import svm
from scipy import stats

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

'''
Fase 1 : Caricamento dataset
'''

# Carica il dataset
data = pd.read_csv('data.csv')

'''
Fase 2 : Pre-Processing
'''

# Rimuovi le colonne inutili
data = data.drop(columns=[
    "ISO_code", "countries", "year", "rank", "quartile",
    "1a_government_consumption", "1b_transfers", "1c_gov_enterprises",
    "1d_top_marg_tax_rate", "2a_judicial_independence", "2b_impartial_courts",
    "2c_protection_property_rights", "2d_military_interference",
    "2e_integrity_legal_system", "2f_legal_enforcement_contracts",
    "2g_restrictions_sale_real_property", "2h_reliability_police", "2i_business_costs_crime",
    "2j_gender_adjustment", "3a_money_growth", "3b_std_inflation", "3c_inflation",
    "3d_freedom_own_foreign_currency", "4a_tariffs", "4b_regulatory_trade_barriers",
    "4c_black_market", "4d_control_movement_capital_ppl", "5a_credit_market_reg",
    "5b_labor_market_reg", "5c_business_reg"
])

# Rimuovi righe con valori NaN
data = data.dropna()

'''
# Visualizzazione boxplot (opzionale, utile per identificare gli outliers)
sns.boxplot(data=data)
plt.title("Boxplot delle feature")
plt.show()
'''

# Discretizzare il target 'ECONOMIC FREEDOM' in classi
est = KBinsDiscretizer(n_bins=50, encode='ordinal', strategy='uniform')
data['ECONOMIC FREEDOM'] = est.fit_transform(data[['ECONOMIC FREEDOM']])

'''
Fase 3 : EDA
'''

# Creo la matrice con i valori che mi interessano escluso il target
data_num = data[[
    "1_size_government",
    "2_property_rights",
    "3_sound_money",
    "4_trade",
    "5_regulation"
]]

# Creo la matrice di correlazione
C = data_num.corr()
#print(C)

'''
# Stampo la matrice di correlazione
plt.matshow(C, vmin=-1, vmax=1)
plt.xticks(np.arange(0, data_num.shape[1]), data_num.columns, rotation=45)
plt.yticks(np.arange(0, data_num.shape[1]), data_num.columns)
plt.title("Visualisation of Correlation Matrix")
plt.colorbar()
plt.show()
'''

plt.matshow(C, vmin=-1, vmax=1)
plt.xticks(np.arange(0, C.shape[1]), C.columns, rotation=45)
plt.yticks(np.arange(0, C.shape[1]), C.columns)
plt.title("Matrice di correlazione")
plt.colorbar()
plt.show()

# Informazioni dataset
#print(data.describe())

'''
Analisi Univariata
'''
# Istogrammi delle variabili numeriche
data_num.hist(figsize=(10, 8))
plt.suptitle("Istogrammi delle variabili numeriche")
plt.show()

'''
Analisi Bivariata
'''
# Scatter plot tra variabili fortemente correlate
sns.pairplot(data_num)
plt.suptitle("Scatter plot delle variabili numeriche")
plt.show()

'''
Fase 4 : Splitting
'''

# Splitting del dataset
data_final = data[[
    "1_size_government",
    "2_property_rights",
    "3_sound_money",
    "4_trade",
    "5_regulation",
    "ECONOMIC FREEDOM"
]]

# Scelta del seed
np.random.seed(24)

#0.2 - 0.2 - 0.6

#0.8 * x = 0.2 -> 0.8 * 0.25 = 0.2 

train_val, data_test = model_selection.train_test_split(data_final, test_size=0.2)
data_train, data_val = model_selection.train_test_split(train_val, test_size=0.25)

'''
Fase 5 : Regressione Lineare
'''

# Regressione tra `3_sound_money` e `4_trade`
x = data[['3_sound_money']]
y = data[['4_trade']]
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

plt.scatter(x_test, y_test, color='black')
plt.plot(x_test, y_pred, color='blue', linewidth=3)
plt.xlabel('Sound Money')
plt.ylabel('Trade')
plt.title('Linea regressione - Sound Money vs Trade')
plt.show()

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print("\nSound Money - Trade")
print('R^2:', r2)
print('MSE:', mse)

# Coefficienti della regressione
coefficients = model.coef_
intercept = model.intercept_
print("Coefficiente:", coefficients)
print("Intercetta:", intercept)

residuals = y_test - y_pred
media = np.mean(residuals)
mediana = np.median(residuals)
print('Mediana : ' , mediana)
print('Media : ', media)
sns.histplot(residuals, kde=True, palette='Blues', stat="density", linewidth=1)
plt.xlabel('Residui')
plt.ylabel('Densità')
plt.title('Istogramma residui - Trade vs Sound Money')
plt.legend(['y_test', 'y_pred'])
plt.show()

import statsmodels.api as sm

residuals_sorted = np.sort(residuals, axis=0)

fig = sm.qqplot(residuals_sorted, line='r')
plt.title('QQ Plot dei Residui - Trade vs Sound Money')
plt.show()

# Regressione tra `4_trade` e `5_regulation`
x = data[['4_trade']]
y = data[['5_regulation']]
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

plt.scatter(x_test, y_test, color='black')
plt.plot(x_test, y_pred, color='blue', linewidth=3)
plt.xlabel('Trade')
plt.ylabel('Regulation')
plt.title('Linea regressione - Trade vs Regulation')
plt.show()

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print("\nTrade - Regulation")
print('R^2:', r2)
print('MSE:', mse)

# Coefficienti della regressione
coefficients = model.coef_
intercept = model.intercept_
print("Coefficiente:", coefficients)
print("Intercetta:", intercept)

residuals = y_test - y_pred
media = np.mean(residuals)
mediana = np.median(residuals)
print('Mediana : ' , mediana)
print('Media : ', media)
sns.histplot(residuals, kde=True, palette='Blues', stat="density", linewidth=1)
plt.xlabel('Residui')
plt.ylabel('Densità')
plt.title('Istogramma residui - Trade vs Sound Money')
plt.legend(['y_test', 'y_pred'])
plt.show()

residuals_sorted = np.sort(residuals, axis=0)

fig = sm.qqplot(residuals_sorted, line='r')
plt.title('QQ Plot dei Residui - Trade vs Sound Money')
plt.show()



'''
Parte 6 : Addestramento del modello
'''

# Addestramento del modello SVM
x_train = data_train[[
    "1_size_government",
    "2_property_rights",
    "3_sound_money",
    "4_trade",
    "5_regulation"
]]
y_train = data_train[["ECONOMIC FREEDOM"]]

#model = linear_model.LogisticRegression()
#model = svm.SVC(kernel="poly", degree=3)
model = svm.SVC(kernel="linear", C = 7)
model.fit(x_train, y_train["ECONOMIC FREEDOM"])

'''
Parte 7 : Hyperparameter Tuning
'''

x_val = data_val[[
    "1_size_government",
    "2_property_rights",
    "3_sound_money",
    "4_trade",
    "5_regulation"
]]
y_val = data_val[["ECONOMIC FREEDOM"]]

y_pred = model.predict(x_val)

'''
cm=confusion_matrix(y_val , y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm,annot=True, fmt = 'd', cmap='Blues', cbar=False)
plt.title('Confusione')
plt.xlabel('Predetto')
plt.ylabel('Effettivo')
plt.show
'''

print("\nMisclassification Error")
ME = np.sum(y_pred != y_val["ECONOMIC FREEDOM"])
MR = ME / len(y_pred)
Acc = 1 - MR

print(f"ME : {ME}.")
print(f"MR : {MR}.")
print(f"Acc : {Acc}.")

# Ciclo il grado
g = 100
accuracies = []

print("\nMetriche di ogni grado [ ME, MR, acc] ")
for d in range(1, g+1):
    #model = linear_model.LogisticRegression()
    #model = svm.SVC(kernel="poly", degree=d)
    model = svm.SVC(kernel="linear", C=d)
    model.fit(x_train, y_train["ECONOMIC FREEDOM"])
    y_pred = model.predict(x_val)

    ME = np.sum(y_pred != y_val["ECONOMIC FREEDOM"])
    MR = ME / len(y_pred)
    acc = 1 - MR
    accuracies.append(acc)
    print(f"{d} : {ME} , {MR} , {acc}")

plt.figure(figsize=(14, 7))
plt.plot(range(1, g+1), accuracies, marker='o')
plt.title('Accuratezza in funzione del grado del kernel lineare')
plt.xlabel('Grado del kernel lineare')
plt.ylabel('Accuratezza')
plt.grid(True)
plt.show()

'''
Punto 8 : Valutazione della performance
'''

# Valutazione della performance sul test set
#model = svm.SVC(kernel="poly", degree=3)
model = svm.SVC(kernel="linear", C=61)
model.fit(x_train, y_train.values.ravel())

x_test = data_test[[
    "1_size_government",
    "2_property_rights",
    "3_sound_money",
    "4_trade",
    "5_regulation"
]]
y_test = data_test[["ECONOMIC FREEDOM"]]

y_pred_test = model.predict(x_test)

ME_test = np.sum(y_pred_test != y_test["ECONOMIC FREEDOM"])
MR_test = ME_test / len(y_pred_test)
Acc_test = 1 - MR_test

print("\nAccuratezza test")
print(f"ME : {ME_test}.")
print(f"MR : {MR_test}.")
print(f"Acc : {Acc_test}.")

'''
Punto 9 : Studio statistico sui risultati della valutazione
'''

# Rimozione seed di casualità costante
np.random.seed()

# Ripetizioni per la valutazione del modello
k = 20
accuracies = []

for r in range(k):
    
    train_val, data_test = model_selection.train_test_split(data_final, test_size=0.2)
    data_train, data_val = model_selection.train_test_split(train_val, test_size=0.25)  # 0.25 * 0.8 = 0.2
    
    # Addestramento del modello SVM
    x_train = data_train[[
        "1_size_government",
        "2_property_rights",
        "3_sound_money",
        "4_trade",
        "5_regulation"
    ]]
    y_train = data_train[["ECONOMIC FREEDOM"]]
    
    x_test = data_test[[
        "1_size_government",
        "2_property_rights",
        "3_sound_money",
        "4_trade",
        "5_regulation"
    ]]
    y_test = data_test[["ECONOMIC FREEDOM"]]
    
    #model = svm.SVC(kernel="poly", degree=3)
    model = svm.SVC(kernel="linear", C=61)
    model.fit(x_train, y_train.values.ravel())
    y_pred_test = model.predict(x_test)

    ME = np.sum(y_pred_test != y_test["ECONOMIC FREEDOM"])
    MR = ME / len(y_pred_test)
    acc = 1 - MR
    accuracies.append(acc)
    

# Statistica descrittiva
mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)
median_acc = np.median(accuracies)
var_acc = np.var(accuracies)
min_acc = np.min(accuracies)
max_acc = np.max(accuracies)
q1 = np.percentile(accuracies, 25)
q3 = np.percentile(accuracies, 75)

print(f"\nAccuratezza media: {mean_acc}")
print(f"Deviazione standard dell'accuratezza: {std_acc}")
print(f"Mediana: {median_acc}")
print(f"Varianza: {var_acc}")
print(f"Min: {min_acc}")
print(f"Max: {max_acc}")
print(f"1° Quartile: {q1}")
print(f"3° Quartile: {q3}")

# Istogramma delle accuratezze
plt.hist(accuracies, bins=np.linspace(0.85, 0.97, 1000), edgecolor='black', alpha=1)
plt.title('Distribuzione delle Accuratezze')
plt.xlabel('Accuratezza')
plt.ylabel('Frequenza')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Boxplot delle accuratezze
plt.boxplot(accuracies)
plt.title('Boxplot delle Accuratezze')
plt.ylabel('Accuratezza')
plt.grid();
plt.show()

# Statistica inferenziale - Intervallo di confidenza
confidence_level = 0.95
degrees_freedom = k - 1
sample_mean = np.mean(accuracies)
sample_standard_error = stats.sem(accuracies)
confidence_interval = stats.t.interval(confidence_level, degrees_freedom, sample_mean, sample_standard_error)

print(f"Intervallo di confidenza al 95%: {confidence_interval}")

