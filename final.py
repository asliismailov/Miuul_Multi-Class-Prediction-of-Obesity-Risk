#genel gerekli kütüphaneler
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
#standartlaştırma için kütüphaneler
from sklearn.preprocessing import StandardScaler
#encode işlemleri için kütüphaneler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
#model oluşturmak için kütüphaneler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
#modeli iyileştirmek için kütüphaneler
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve, train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import shap

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

warnings.simplefilter(action='ignore', category=Warning)

############################################################ GÖREV 1: KEŞİFCİ VERİ ANALİZİ ############################################################

df_train = pd.read_csv("datasets/train.csv")
df_train.head()
df_test = pd.read_csv("datasets/test.csv")
##################################
# GENEL RESİM
##################################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df_train, head=2)



##################################
# NUMERİK VE KATEGORİK DEĞİŞKENLERİN YAKALANMASI
##################################

# Sayısal değişkenlerin listesi
numerical_columns = df_train.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Kategorik değişkenlerin listesi
categorical_columns = df_train.select_dtypes(include=['object']).columns.tolist()


numerical_analysis = df_train[numerical_columns].describe()
categorical_analysis = df_train[categorical_columns].describe()

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df_train)

cat_cols
num_cols
cat_but_car

df_train.head()

##################################
# KATEGORİK DEĞİŞKENLERİN ANALİZİ
##################################
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df_train,col)

# Kategorik değişkenler için çubuk grafikler
plt.figure(figsize=(15, 10))
for i, column in enumerate(categorical_columns, 1):
    plt.subplot(3, 3, i)
    sns.countplot(x=column, data=df_train)
    plt.title(column)
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

#Gender ve family_history_with_overweight gibi değişkenlerde belirgin bir dağılım görülmekte. Örneğin, ailede obezite öyküsü olanların sayısı, olmayanlardan daha fazla.
#MTRANS (ulaşım şekli) değişkeninde, insanların çoğunlukla halka açık taşıma kullanımı göze çarpıyor.

##################################
# NUMERİK DEĞİŞKENLERİN ANALİZİ
##################################
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df_train, col, plot=True)

# Numerik değişkenler için histogramlar
# Veri seti boyutunu küçültmek için sayısal değişkenlerden bazılarını seç
selected_numerical = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

plt.figure(figsize=(15, 10))
for i, column in enumerate(selected_numerical, 1):
    plt.subplot(2, 4, i)
    sns.histplot(df_train[column], kde=True)
    plt.title(column)

plt.tight_layout()
plt.show()
#Age, Height, ve Weight değişkenleri için dağılımlar, çeşitli şekillerde yoğunlaşmış veri noktalarını gösteriyor. Bu, katılımcıların yaş, boy ve kilo çeşitliliğini ortaya koyuyor.
#FCVC (sebze tüketimi), NCP (günlük ana yemek sayısı), CH2O (günlük su tüketimi), FAF (fiziksel aktivite frekansı) ve TUE (teknoloji kullanımı) gibi diğer değişkenler de benzersiz dağılımlara sahip.


##################################
# NUMERİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ
##################################

## Kategorik ve sayısal değişkenler arasındaki ilişkileri incelemek için kutu grafikleri
plt.figure(figsize=(15, 20))
for i, column in enumerate(selected_numerical, 1):
    plt.subplot(4, 2, i)
    sns.boxplot(x='NObeyesdad', y=column, data=df_train)
    plt.title(f'NObeyesdad vs {column}')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

## Kategorik değişkenlerin hedefe göre grafikleri
plt.figure(figsize=(20, 15))
for i, column in enumerate(categorical_columns, 1):
    plt.subplot(3, 3, i)
    sns.countplot(x=column, hue='NObeyesdad', data=df_train)
    plt.title(f'{column} by NObeyesdad')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

## Numerik değişkenlerin hedefe göre grafikleri
plt.figure(figsize=(20, 15))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(3, 4, i)
    sns.boxplot(x='NObeyesdad', y=column, data=df_train)
    plt.title(f'{column} by NObeyesdad')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

##################################
# KORELASYON
##################################

korelasyon_matrisi = df_train[numerical_columns].corr()
korelasyon_matrisi
plt.figure(figsize=(10, 8))
sns.heatmap(df_train[selected_numerical].corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Korelasyon Matrisi')
plt.savefig('korelasyon_matrisi.png')
plt.show()

######################################################################################################################################################
############################################################ GÖREV 2: FEATURE ENGINEERING ############################################################


##################################
# EKSİK DEĞER ANALİZİ
##################################

missing_values = df_train.isnull().sum()


##################################
# AYKIRI DEĞER ANALİZİ
##################################
# Eksik değerlerin sayısını kontrol edelim
missing_values_train = df_train.isnull().sum()
missing_values_test = df_test.isnull().sum()

missing_values_train, missing_values_test


# Aykırı değerleri sınırlar içine çekme fonksiyonu
def cap_values(data, column, min_val, max_val):
    data[column] = data[column].clip(lower=min_val, upper=max_val)
    return data

# Age, Height ve Weight için sınırların uygulanması
df_train = cap_values(df_train, 'Age', 0, 100)
df_train = cap_values(df_train, 'Height', 1.2, 2.5)
df_train = cap_values(df_train, 'Weight', 30, 200)

df_test = cap_values(df_test, 'Age', 0, 100)
df_test = cap_values(df_test, 'Height', 1.2, 2.5)
df_test = cap_values(df_test, 'Weight', 30, 200)

# Aykırı değer işlemi sonrası verileri tekrar kontrol edelim
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

sns.boxplot(y=df_train["Age"], ax=ax[0])
ax[0].set_title('Adjusted Age Distribution')

sns.boxplot(y=df_train["Height"], ax=ax[1])
ax[1].set_title('Adjusted Height Distribution')

sns.boxplot(y=df_train["Weight"], ax=ax[2])
ax[2].set_title('Adjusted Weight Distribution')

plt.tight_layout()
plt.show()



##################################
# DEĞİŞKEN EKLEME
##################################

# BMI hesaplama fonksiyonu
def calculate_bmi(weight, height):
    return weight / (height ** 2)

# Eğitim ve test veri setlerine BMI değişkeni ekleme
df_train['BMI'] = calculate_bmi(df_train['Weight'], df_train['Height'])
df_test['BMI'] = calculate_bmi(df_test['Weight'], df_test['Height'])

# BMI dağılımını kontrol etmek için ilk beş satırı gösterelim
df_train.head(), df_test.head()



##################################
# ENCODING
##################################

# Kategorik değişkenlerin benzersiz değer sayılarına bakalım
categorical_unique_counts = {column: df_train[column].nunique() for column in df_train.select_dtypes(include=['object']).columns}
categorical_unique_counts


def model_ready(df):
    bin_cat_columns = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC', 'Gender']
    multi_cat_columns = ['CAEC', 'CALC', 'MTRANS']

    # Binary categorical columns
    for col in bin_cat_columns:
        df[col] = df[col].apply(lambda x: 1 if x == 'yes' or x == 'Male' else 0)

    # One hot encoding for multi categorical columns
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    for col in multi_cat_columns:
        encoded_data = encoder.fit_transform(df[[col]])
        df_encoded = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([col]))
        df.drop([col], axis=1, inplace=True)
        df = pd.concat([df, df_encoded], axis=1)

    return df


# Apply the function to both train and test datasets
df_train_prepared = model_ready(df_train.copy())
df_test_prepared = model_ready(df_test.copy())

df_train_prepared.head(), df_test_prepared.head()

# BMI sütununu tekrar hesaplayıp ekleyelim
df_train_prepared['BMI'] = calculate_bmi(df_train['Weight'], df_train['Height'])
df_test_prepared['BMI'] = calculate_bmi(df_test['Weight'], df_test['Height'])

# Eğitim veri setindeki NObeyesdad sütununu çıkartmıştık, tekrar ekleyelim
df_train_prepared['NObeyesdad'] = df_train['NObeyesdad']

# Sütun sıralarını kontrol edelim
df_train_prepared.head(), df_test_prepared.head(), df_train_prepared.columns, df_test_prepared.columns

##################################
# STANDARTLAŞTIRMA
##################################

# Diğer numerik sütunları (id hariç) standartlaştıralım
numeric_columns = df_train_prepared.select_dtypes(include=['float64', 'int64']).columns.tolist()
numeric_columns.remove('id')  # ID sütununu çıkaralım

# Tekrar standartlaştırma işlemi yapalım
scaler = StandardScaler()
df_train_prepared[numeric_columns] = scaler.fit_transform(df_train_prepared[numeric_columns])
df_test_prepared[numeric_columns] = scaler.transform(df_test_prepared[numeric_columns])

df_train_prepared.head(), df_test_prepared.head()

columns_train = [col for col in df_train_prepared.columns if col not in ['BMI', 'NObeyesdad']] + ['BMI', 'NObeyesdad']
columns_test = [col for col in df_test_prepared.columns if col not in ['BMI', 'NObeyesdad']] + ['BMI']
train_data_encoded = df_train_prepared[columns_train]
test_data_encoded_CALC = df_test_prepared[columns_test]
test_data_encoded = test_data_encoded_CALC.drop(["CAEC_Always"], axis=1)


##################################
# MODELLEME
##################################
# Veri ve etiketlerin yüklenmesi
X = train_data_encoded.drop(columns=['NObeyesdad', 'id'])
y = train_data_encoded['NObeyesdad']

# Hedef değişkenin sayısallaştırılması
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Eğitim ve test setlerine ayrılması
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2)

# Model listesi
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True),
    'GBM': GradientBoostingClassifier(),
   # 'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    'LightGBM': LGBMClassifier()
}

# Model sonuçlarını saklamak için bir sözlük
results = {}

# Modelleri eğitme ve performans değerlendirme
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = 'N/A'  # ROC AUC başlangıçta 'N/A' olarak ayarlanır

    # Modelin predict_proba desteği varsa ROC AUC hesapla
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovo', average='macro')

    results[name] = {'Accuracy': accuracy, 'F1 Score': f1, 'ROC AUC': roc_auc}

# Sonuçların yazdırılması
for model_name, metrics in results.items():
    print(
        f"{model_name} - Accuracy: {metrics['Accuracy']:.3f}, F1 Score: {metrics['F1 Score']:.3f}, ROC AUC: {metrics['ROC AUC']}")

#Logistic Regression - Accuracy: 0.870, F1 Score: 0.869, ROC AUC: 0.9790860135579265
#Decision Tree - Accuracy: 0.840, F1 Score: 0.841, ROC AUC: 0.8974122828799521
#Random Forest - Accuracy: 0.898, F1 Score: 0.898, ROC AUC: 0.9853846765178457
#SVM - Accuracy: 0.861, F1 Score: 0.861, ROC AUC: 0.9789156192145787
#GBM - Accuracy: 0.901, F1 Score: 0.901, ROC AUC: 0.9871334932653371
#XGBoost - Accuracy: 0.907, F1 Score: 0.907, ROC AUC: 0.9875590085763084
#LightGBM - Accuracy: 0.904, F1 Score: 0.904, ROC AUC: 0.9876912210512012

##Lightgbm ile ilermeye karar verdik.Modeli kuruyoruz.
lgbm_model_train = LGBMClassifier()
lgbm_model_train.get_params()

##hiperparametre ayarları

# LightGBM parametre aralığı
param_grid = {
    'num_leaves': [31, 50, 70],
    'max_depth': [10, 20, -1],
    'learning_rate': [0.1, 0.01],
    'n_estimators': [100, 200]
}

lgbm = lgb.LGBMClassifier()
grid_search_lgbm = GridSearchCV(estimator=lgbm, param_grid=param_grid, cv=3, scoring='accuracy')
grid_search_lgbm.fit(X_train, y_train)

print(f"LightGBM en iyi parametreler: {grid_search_lgbm.best_params_}")

#LightGBM en iyi parametreler: {'learning_rate': 0.1, 'max_depth': 10, 'n_estimators': 100, 'num_leaves': 31}
final_train_model = lgbm_model_train.set_params(**grid_search_lgbm.best_params_).fit(X, y)


##Direk crossvalidate dedigimizde f1 ve roc_auc nan geldigi icin stratifiedkfold kullandik
# Katman sayısı
k_folds= 10  # Örneğin 10 katmanlı çapraz doğrulama yapacağız

# StratifiedKFold'u kullanarak dengeli katmanlar oluşturun
stratified_kfold = StratifiedKFold(n_splits=k_folds, shuffle=True)

# Her bir katman için eğitim ve doğrulama verilerini alın
for train_index, val_index in stratified_kfold.split(X, y):
    X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
    y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

    lgbm_model_train.fit(X_train_fold, y_train_fold)
    y_pred = lgbm_model_train.predict(X_val_fold)

    # Modelin tahminlerine göre performans metriklerini hesaplayın
    accuracy = accuracy_score(y_val_fold, y_pred)
    f1 = f1_score(y_val_fold, y_pred, average='weighted')
    y_proba = final_train_model.predict_proba(X_val_fold)
    roc_auc = roc_auc_score(y_val_fold, y_proba, multi_class='ovr')

    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("ROC AUC Score:", roc_auc)
    print("")

#Accuracy: 0.912289156626506
#F1 Score: 0.9117074426314465
#ROC AUC Score: 0.9907029673127952



##################################
# FEATURE IMPORTANCE
##################################
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    print(feature_imp.sort_values("Value",ascending=False))
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(final_train_model, X)

import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve



import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
train_sizes, train_scores, test_scores = learning_curve(final_train_model, X, y_encoded, cv=5, scoring='roc_auc_ovr', train_sizes=np.linspace(0.1, 1.0, 5))

# Ortalama ve standart sapma hesapla
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Öğrenme eğrilerini çizdir
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='r', alpha=0.1)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color='g', alpha=0.1)
plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")

plt.title("Learning Curve")
plt.xlabel("Training Set Size")
plt.ylabel("ROC AUC Score")
plt.legend(loc="best")
plt.show()

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################## TEST Veri setine geçiyoruz ##################################################
# Bağımlı ve bağımsız değişkenler
X_test = test_data_encoded.drop(["id"], axis=1)  # 'id' ve hedef değişken hariç tüm değişkenler

# LightGBM modelinizi kullanarak tahmin yapın
predicted_labels = final_train_model.predict(X_test)

# Tahmin edilen değerleri "NObeyesdad" sütununa ekleyin
test_data_encoded['NObeyesdad'] = predicted_labels


#test ve train datasinin sutunlarinin ayni sayi ve sirada olmasi gerekiyor
train_data_encoded.head(5)
train_data_encoded.shape

test_data_encoded.shape
test_data_encoded.head(5)


##Train ve Test datasi ile son modeli kuruyoruz

# Bağımlı ve bağımsız değişkenler
X_train = train_data_encoded.drop(['NObeyesdad', 'id'], axis=1)  # 'id' ve hedef değişken hariç tüm değişkenler
X_test = test_data_encoded.drop(['NObeyesdad', 'id'], axis=1)
y_train = train_data_encoded['NObeyesdad']
y_test = test_data_encoded['NObeyesdad']

final_train_model = LGBMClassifier()

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Katman sayısı
k_folds_final = 10  # Örneğin 10 katmanlı çapraz doğrulama yapacağız

# StratifiedKFold'u kullanarak dengeli katmanlar oluşturun
stratified_kfold_final = StratifiedKFold(n_splits=k_folds_final, shuffle=True)

# Her bir katman için eğitim ve doğrulama verilerini alın
for train_index, val_index in stratified_kfold_final.split(X, y):
    X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
    y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

    final_train_model.fit(X_train_fold, y_train_fold)
    y_pred = final_train_model.predict(X_val_fold)

    # Modelin tahminlerine göre performans metriklerini hesaplayın
    accuracy = accuracy_score(y_val_fold, y_pred)
    f1 = f1_score(y_val_fold, y_pred, average='weighted')
    y_proba = final_train_model.predict_proba(X_val_fold)
    roc_auc = roc_auc_score(y_val_fold, y_proba, multi_class='ovr')

    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("ROC AUC Score:", roc_auc)
    print("")

#Accuracy: 0.9016867469879518
#F1 Score: 0.9012256716552685
#ROC AUC Score: 0.989246380949021

##STREAMLIT HAZIRLIK

#Trainde ve Test datasinda son modelin basarimi ayni, hatta biraz daha iyi. Bu nedenle Train ve Test datasini birlestiriyoruz

# Train ve test verilerini birleştirme
df_all = pd.concat([train_data_encoded, test_data_encoded], ignore_index=True)

# Model için gerekli olan hedef değişken ve bağımsız değişkenler
X_final = df_all.drop(['NObeyesdad', 'id'], axis=1)
y_final = df_all['NObeyesdad']

# Modelin final veri üzerinde eğitilmesi
final_train_model.fit(X_final, y_final)

# Modelin kaydedilmesi
import joblib

# Modeli diske kaydetmek
joblib.dump(final_train_model, "Miuul_Final.pkl")

# Modelin tekrar yüklenmesi
loaded_model = joblib.load("/Users/ugurcanodabasi/Desktop/PycharmProjects/Data_Scientist_Bootcamp/Final/Data/Miuul_Final.pkl")

# Modelin yüklenip tahmin yapılması
loaded_model.predict(X_final)

print("Modelin Eğitim Başarısı:", accuracy)
print("Modelin F1 Score Başarısı:", f1)
print("Modelin ROC AUC Score Başarısı:", roc_auc)

#Accuracy: 0.9016867469879518
#F1 Score: 0.9012256716552685
#ROC AUC Score: 0.9988776831769022
#Modelin Eğitim Başarısı: 0.9012048192771084
#Modelin F1 Score Başarısı: 0.9011491650331063
#Modelin ROC AUC Score Başarısı: 0.9891384565520835
#datayı csv olarak kaydet

# Tahmin edilen değerleri içeren DataFrame oluşturma
predicted_df = df_all

# Tahminleri CSV dosyasına kaydetme
predicted_df.to_csv('Miuul_Final_data.csv', index=False)

df = pd.read_csv("final_predictions.csv")

