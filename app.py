from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import os, re

import nltk
nltk.download("wordnet")
nltk.download("omw-1.4")
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc

import pandas as pd

# Logica
class Articulo:
  def __init__(self, n, p, k):
    self.file_name= n
    self.path = p
    self.keywords = k

root = Tk()

tabs = ttk.Notebook(root)
tabs.pack(fill=BOTH, expand=TRUE)
trainFrame = ttk.Frame(tabs)
clasiFrame = ttk.Frame(tabs)
tabs.add(trainFrame, text="Entrenamiento")
tabs.add(clasiFrame, text="Clasificacion")

def getOdioDirectory():
    odio_input.delete(0, 'end')
    odio_input.insert(0, filedialog.askdirectory(title="Elige carpeta Odio"))

def getNoOdioDirectory():
    no_odio_input.delete(0, 'end')
    no_odio_input.insert(0, filedialog.askdirectory(title="Elige carpeta No Odio"))

def tokenize(x):
    return RegexpTokenizer(r'\w+').tokenize(x.lower())

def removeStopwords(x):
    with open("stopWords_es.txt") as f:
        text = f.read()
        prohibitedWords = text.split("\n")
        return [word for word in x if not word in prohibitedWords]

def lemmatize(x):
 lemmatizer = WordNetLemmatizer()
 return ' '.join([lemmatizer.lemmatize(word) for word in x])

def stemming(x):
    stemmer = SnowballStemmer(language="spanish")
    return ' '.join([stemmer.stem(word) for word in x])

def generateDF(path, label):
    df = pd.DataFrame({"name": [], "odio": [], "content":[]})
    files = os.listdir(path)
    for file in files:
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding="ISO-8859-1") as f:
                df = df.append({"name": file, "odio": label, "content": f.read()}, ignore_index=True)
    return df


def train():
    try:
        df = pd.DataFrame({"name": [], "odio": [], "content":[]})
        df = df.append(generateDF(odio_input.get(), 1))
        df = df.append(generateDF(no_odio_input.get(), 0))
        print(df.head())
        df['tokens'] = df['content'].map(tokenize)
        print(df.head())
        df['tokens'] = df['tokens'].map(removeStopwords)
        print(df.head())
        df['lemma'] = df['tokens'].map(stemming)
        print(df.head())
        print(df['lemma'])

        X_train, X_test, y_train, y_test = train_test_split(df['lemma'],df['odio'],test_size=0.2,shuffle=True)
        tfidf_vectorizer = TfidfVectorizer(use_idf=True)
        X_train_vectors_tfidf = tfidf_vectorizer.fit_transform(X_train) 
        X_test_vectors_tfidf = tfidf_vectorizer.transform(X_test)

        lr_tfidf=LogisticRegression(solver = 'liblinear', C=10, penalty = 'l2')
        lr_tfidf.fit(X_train_vectors_tfidf, y_train)

        y_predict = lr_tfidf.predict(X_test_vectors_tfidf)
        y_prob = lr_tfidf.predict_proba(X_test_vectors_tfidf)[:,1]
        print(classification_report(y_test,y_predict))
        print('Confusion Matrix:',confusion_matrix(y_test, y_predict))
        
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        print('AUC:', roc_auc)

    except FileNotFoundError:
        messagebox.showwarning("At least one of the paths you provided isn't valid")
    # except:
    #     print("Error occured")
    #     messagebox.showerror("Something went wrong")

odio_files = []
no_odio_files = []
def updateSummaryFiles(var, index, mode):
    global odio_files
    global no_odio_files
    if odio_path.get() != "":
        try:
            odio_files = os.listdir(odio_path.get())
            articles_odio.config(text = str(len(odio_files)))
        except FileNotFoundError:
            messagebox.showwarning("Ivalid Odio path", "The path given is not a correct path to a directory")
    if no_odio_path.get() != "":
        try:
            no_odio_files = os.listdir(no_odio_path.get())
            articles_no_odio.config(text = str(len(no_odio_files)))
        except FileNotFoundError:
            messagebox.showwarning("Ivalid No Odio path", "The path given is not a correct path to a directory")
    articles_total.config(text = str(len(odio_files) + len(no_odio_files)))

def updateSummaryAlgorithm(var, index, mode):
    algorithm_selected.config(text = algorithm.get())


# Odio Label
ttk.Label(trainFrame, text="Noticias de Odio:").grid(column=0, row=0, sticky=W)
# Odio path input
odio_path = StringVar()
odio_path.trace_add("write", updateSummaryFiles)
odio_input = ttk.Entry(trainFrame, textvariable=odio_path)
odio_input.grid(column=1, row=0, columnspan=2, sticky=(W, E))
# Odio Button
ttk.Button(trainFrame, text="Abrir", command=getOdioDirectory).grid(column=3, row=0, sticky=E)

# No Odio Label
ttk.Label(trainFrame, text="Noticias de No Odio:").grid(column=0, row=1, sticky=W)
# Odio path input
no_odio_path = StringVar()
no_odio_path.trace_add("write", updateSummaryFiles)
no_odio_input = ttk.Entry(trainFrame, textvariable=no_odio_path)
no_odio_input.grid(column=1, row=1, columnspan=2, sticky=(W, E))
# Odio Button
ttk.Button(trainFrame, text="Abrir", command=getNoOdioDirectory).grid(column=3, row=1, sticky=E)

# Algorithm Label
ttk.Label(trainFrame, text="Seleccionar Algoritmo:").grid(column=0, row=2, sticky=W)
# Algorithm select
algorithm = StringVar()
algorithm.trace_add("write", updateSummaryAlgorithm)
algo_choice = ["1", "2", "3"]
algo_select = ttk.Combobox(trainFrame, textvariable=algorithm, state="readonly", values=algo_choice)
algo_select.grid(column=1, row=2, sticky=(E,W))
# Execute button
ttk.Button(trainFrame, text="Execute", default="active", command=train).grid(column=2, columnspan=2, row=2, sticky=(E,W))

# padding
for child in trainFrame.winfo_children(): 
    child.grid_configure(padx=5, pady=5)

ttk.Label(trainFrame, text="Ejemplares de Odio:").grid(column=0, row=4, sticky=W)
ttk.Label(trainFrame, text="Ejemplares de No Odio:").grid(column=0, row=5, sticky=W)
ttk.Label(trainFrame, text="Total:").grid(column=0, row=6, sticky=W)
ttk.Label(trainFrame, text="Algoritmo seleccionado:").grid(column=0, row=7, sticky=W)

articles_odio = ttk.Label(trainFrame, text="-")
articles_odio.grid(column=1, row=4, sticky=W)
articles_no_odio = ttk.Label(trainFrame, text="-")
articles_no_odio.grid(column=1, row=5, sticky=W)
articles_total = ttk.Label(trainFrame, text="-")
articles_total.grid(column=1, row=6, sticky=W)
algorithm_selected = ttk.Label(trainFrame, text="-")
algorithm_selected.grid(column=1, row=7, sticky=W)

# growing
trainFrame.columnconfigure(0, weight=1, minsize=150)
trainFrame.columnconfigure(1, weight=2, minsize=100)
trainFrame.columnconfigure(2, weight=2, minsize=50)
trainFrame.columnconfigure(3, weight=1, minsize=100)

# --------------------------------------------------------
# ------- Clasificador -----------------------------------
# --------------------------------------------------------

# noticias Label
ttk.Label(clasiFrame, text="Noticias para clasificar").grid(column=0, row=0, sticky=W)
# Odio path input
noticias_path = StringVar()
noticias_input = ttk.Entry(clasiFrame, textvariable=noticias_path)
noticias_input.grid(column=1, row=0, columnspan=2, sticky=(W, E))
# noticias Button
ttk.Button(clasiFrame, text="Abrir", command=getOdioDirectory).grid(column=3, row=0, sticky=E)

# Model Label
ttk.Label(clasiFrame, text="Model clasificador:").grid(column=0, row=1, sticky=W)
# Model input
model_path = StringVar()
model_input = ttk.Entry(clasiFrame, textvariable=model_path)
model_input.grid(column=1, row=1, columnspan=2, sticky=(W, E))
# model Button
ttk.Button(clasiFrame, text="Abrir", command=getNoOdioDirectory).grid(column=3, row=1, sticky=E)

# growing
clasiFrame.columnconfigure(0, weight=1, minsize=150)
clasiFrame.columnconfigure(1, weight=2, minsize=100)
clasiFrame.columnconfigure(2, weight=2, minsize=50)
clasiFrame.columnconfigure(3, weight=1, minsize=100)

# padding
for child in clasiFrame.winfo_children(): 
    child.grid_configure(padx=5, pady=5)

root.geometry("800x400")
root.title("Python classificador")
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
root.mainloop()