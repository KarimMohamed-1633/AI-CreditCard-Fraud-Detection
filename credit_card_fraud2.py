import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from sklearn.tree import DecisionTreeClassifier

# Read the dataset
df = pd.read_csv("creditcard.csv")

# Plotting Fraudulent vs Non-fraudulent data
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df)
plt.tight_layout()
plt.show()

# Visualizing the distribution of transaction times
plt.figure(figsize=(12, 6))
sns.distplot(df['Time'] / 3600, bins=48, kde=False)
plt.xticks(np.arange(0, 54, 6))
plt.xlabel('Time After First Transaction (hr)')
plt.ylabel('Count')
plt.title('Transaction Times')
plt.show()

# Differentiating fraudulent and non-fraudulent transactions based on time
plt.figure(figsize=(12, 6))
sns.distplot(df[df['Class'] == 0]['Time'] / 3600, bins=48)
plt.xticks(np.arange(0, 54, 6))
plt.xlabel('Time After First Transaction (hr)')
plt.ylabel('Count')
plt.title('Non-Fraud Transactions')
plt.show()

plt.figure(figsize=(12, 6))
sns.distplot(df[df['Class'] == 1]['Time'] / 3600, bins=48)
plt.xticks(np.arange(0, 54, 6))
plt.xlabel('Time After First Transactions (hr)')
plt.ylabel('Count')
plt.title('Fraud Transactions')
plt.show()

# Plotting the Amount
plt.figure(figsize=(20, 6))
sns.distplot(df['Amount'])
plt.show()

# Perform IQR analysis to remove outliers
upper_limit = df['Amount'].quantile(0.75) + (1.5 * (df['Amount'].quantile(0.75) - df['Amount'].quantile(0.25)))
print("Upper Limit for Outliers:", upper_limit)
print("Fraudulent Cases with Amount Greater Than Upper Limit:")
print(df[df['Amount'] > upper_limit]['Class'].value_counts())

# Removing Outliers
df_copy = df[df['Amount'] <= 8000]
print("Class Distribution after Removing Outliers:")
print(df_copy['Class'].value_counts())

# Percentage of Fraudulent Activity
print('Percentage of Fraudulent Activity: {:.2%}'.format(df_copy[df_copy['Class'] == 1].shape[0] / df_copy.shape[0]))

# Visualizing Amount after removing outliers
plt.figure(figsize=(10, 15))
plt.subplot(2, 1, 1)
sns.boxplot(x=df_copy[df_copy['Class'] == 1]['Amount'])
plt.title('Fraudulent Transactions')
plt.subplot(2, 1, 2)
sns.boxplot(x=df_copy[df_copy['Class'] == 0]['Amount'])
plt.title('Real Transactions')
plt.show()

# Understanding Correlation Between the Features in the Data
corr = df.corr()
plt.figure(figsize=(9, 7))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, linewidths=0.1, cmap='RdBu')
plt.tight_layout()
plt.show()

# Balancing the data using undersampling
non_fraud = df_copy[df_copy['Class'] == 0].sample(2000)
fraud = df_copy[df_copy['Class'] == 1]
df_balanced = pd.concat([non_fraud, fraud]).sample(frac=1).reset_index(drop=True)

# Visualizing balanced data using t-SNE
x = df_balanced.drop(['Class'], axis=1).values
y = df_balanced['Class'].values

p = TSNE(n_components=2, random_state=24).fit_transform(x)
color_map = {0: 'red', 1: 'blue'}
plt.figure()
for index, cl in enumerate(np.unique(y)):
    plt.scatter(x=p[y == cl, 0], y=p[y == cl, 1], c=color_map[index], label=cl)

plt.xlabel('X in t-SNE')
plt.ylabel('Y in t-SNE')
plt.legend(loc='upper left')
plt.title('t-SNE visualization of balanced data')
plt.show()

# Preparing data for autoencoder
x_scale = StandardScaler().fit_transform(x)
x_norm, x_fraud = x_scale[y == 0], x_scale[y == 1]

# Defining autoencoder architecture
autoencoder = Sequential([
    Dense(x.shape[1], activation='tanh'),
    Dense(100, activation='tanh'),
    Dense(50, activation='relu'),
    Dense(50, activation='tanh'),
    Dense(100, activation='tanh'),
    Dense(x.shape[1], activation='relu')
])

autoencoder.compile(optimizer='adadelta', loss="mse")
autoencoder.fit(x_norm, x_norm, batch_size=256, epochs=10, shuffle=True, validation_split=0.2)

# Extracting hidden representations
hidden_representation = Sequential()
for layer in autoencoder.layers[:-1]:
    hidden_representation.add(layer)

norm_hid_rep = hidden_representation.predict(x_norm)
fraud_hid_representation = hidden_representation.predict(x_fraud)

rep_x = np.append(norm_hid_rep, fraud_hid_representation, axis=0)
y_n = np.zeros(norm_hid_rep.shape[0])
y_f = np.ones(fraud_hid_representation.shape[0])
rep_y = np.append(y_n, y_f)

# Visualizing hidden representations
p2 = TSNE(n_components=2, random_state=24).fit_transform(rep_x)
plt.figure()
for index, cl in enumerate(np.unique(y)):
    plt.scatter(x=p2[y == cl, 0], y=p2[y == cl, 1], c=color_map[index], label=cl)

plt.xlabel('X in t-SNE')
plt.ylabel('Y in t-SNE')
plt.legend(loc='upper left')
plt.title('t-SNE visualization of hidden representations')
plt.show()

# Training models on the extracted data
x_train, val_x, y_train, val_y = train_test_split(rep_x, rep_y, test_size=0.25)

# Logistic Regression
clf = LogisticRegression(solver="lbfgs").fit(x_train, y_train)
predict_y = clf.predict(val_x)
print("Logistic Regression:")
print(classification_report(val_y, predict_y))

# Confusion Matrix for Logistic Regression
plt.figure()
sns.heatmap(confusion_matrix(val_y, predict_y, normalize='true'), annot=True)
plt.title("Confusion Matrix for Logistic Regression")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Accuracy for Logistic Regression
accuracy = np.trace(confusion_matrix(val_y, predict_y)) / np.sum(confusion_matrix(val_y, predict_y))
print("Accuracy:", accuracy)

# Decision Tree Classifier
model_tree = DecisionTreeClassifier(max_depth=4, criterion="entropy")
model_tree.fit(x_train, y_train)
y_pred_tree = model_tree.predict(val_x)
