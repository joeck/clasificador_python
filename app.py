from textwrap import fill
from tkinter import *
from tkinter import ttk
from tkinter import Text
from tkinter import filedialog
from tkinter import messagebox
from IPython.display import display
import os, pickle, time

from pandas.core.frame import DataFrame

from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix

import pandas as pd

root = Tk()

_encoding="UTF-8"
tabs = ttk.Notebook(root)
tabs.pack(fill=BOTH, expand=TRUE)
trainFrame = ttk.Frame(tabs)
clasiFrame = ttk.Frame(tabs)
tabs.add(trainFrame, text="Entrenamiento")
tabs.add(clasiFrame, text="Clasificacion")
clf = ""
tfidf_vectorizer = ""
result = DataFrame()

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

def stemming(x):
    stemmer = SnowballStemmer(language="spanish")
    return ' '.join([stemmer.stem(word) for word in x])

def generateDF(path, label):
    df = pd.DataFrame({"name": [], "odio": [], "content":[]})
    files = os.listdir(path)
    for file in files:
        try: 
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding=_encoding) as f:
                    df = df.append({"name": file, "odio": label, "content": f.read()}, ignore_index=True)
        except:
            pass
            # Skip when reading error occurs

    return df

def showPerformance(report, matrix):
    performanceText.insert("1.0", report)
    tp_label["text"] = "TP: " + str(matrix[0][0])
    fp_label["text"] = "FP: " + str(matrix[0][1])
    fn_label["text"] = "FN: " + str(matrix[1][0])
    tn_label["text"] = "TN: " + str(matrix[1][1])

def decisionTree(X_train_vector, X_test_vector,y_train, y_test):
    global clf
    clf = DecisionTreeClassifier()
    clf.fit(X_train_vector,y_train)
    y_pred = clf.predict(X_test_vector)
    showPerformance(classification_report(y_test, y_pred), confusion_matrix(y_test, y_pred))

def logisticRegression(X_train_vector, X_test_vector,y_train, y_test):
    global clf
    clf=LogisticRegression(solver = 'liblinear', C=10, penalty = 'l2')
    clf.fit(X_train_vector, y_train)
    y_pred = clf.predict(X_test_vector)
    showPerformance(classification_report(y_test, y_pred), confusion_matrix(y_test, y_pred))

def naiveBayes(X_train_vector, X_test_vector,y_train, y_test):
    global clf
    clf = MultinomialNB()
    clf.fit(X_train_vector.toarray(), y_train)
    y_pred = clf.predict(X_test_vector)
    showPerformance(classification_report(y_test, y_pred), confusion_matrix(y_test, y_pred))

def gradientBoostedTree(X_train_vector, X_test_vector,y_train, y_test):
    global clf
    clf = GradientBoostingClassifier()
    clf.fit(X_train_vector, y_train)
    y_pred = clf.predict(X_test_vector)
    showPerformance(classification_report(y_test, y_pred), confusion_matrix(y_test, y_pred))

def split_transform(x, y):
    global tfidf_vectorizer
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,shuffle=True)
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)
    X_train_vectors_tfidf = tfidf_vectorizer.fit_transform(X_train) 
    X_test_vectors_tfidf = tfidf_vectorizer.transform(X_test)
    return [X_train_vectors_tfidf, X_test_vectors_tfidf, y_train, y_test]

def train():
    try:
        begin = time.time()
        train_btn["text"] = "Calculando..."
        trainFrame.update()
        df = pd.DataFrame({"name": [], "odio": [], "content":[]})
        df = df.append(generateDF(odio_input.get(), 1))
        df = df.append(generateDF(no_odio_input.get(), 0))
        df['tokens'] = df['content'].map(tokenize)
        df['tokens'] = df['tokens'].map(removeStopwords)
        df['lemma'] = df['tokens'].map(stemming)

        X_train_vectors_tfidf, X_test_vectors_tfidf, y_train, y_test = split_transform(df['lemma'],df['odio'])

        if algo_select.get() == "Decision Tree":
            decisionTree(X_train_vectors_tfidf, X_test_vectors_tfidf, y_train, y_test)
        elif algo_select.get() == "Logistic Regression":
            logisticRegression(X_train_vectors_tfidf, X_test_vectors_tfidf, y_train, y_test)
        elif algo_select.get() == "Naive Bayes":
            naiveBayes(X_train_vectors_tfidf, X_test_vectors_tfidf, y_train, y_test)
        elif algo_select.get() == "Gradient Boosted Tree":
            gradientBoostedTree(X_train_vectors_tfidf, X_test_vectors_tfidf, y_train, y_test)
        else:
            messagebox.showinfo("No algorithm", "Please select an algorithm")

    except FileNotFoundError:
        messagebox.showwarning("Can't resolve path", "At least one of the paths you provided isn't valid")
    except Exception as e:
        messagebox.showerror("Unexpected error", str(e))

    train_btn["text"] = "Entrenar"
    end = time.time()
    train_time["text"] = f"tiempo: {end - begin: .2f}s"

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

def saveModel():
    global clf
    if clf == "":
        messagebox.showwarning("No model", "Seems like you haven't calculated the model yet")
    elif model_path_input.get() == "":
        messagebox.showwarning("No Path", "Seems like you haven't set a path for the model")
    else:
        filename = model_path_input.get()
        pickle.dump(clf, open(filename, 'wb'))
        pickle.dump(tfidf_vectorizer, open(filename + "_vector", 'wb'))
        messagebox.showinfo("Success", "Model saved successfully!")

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
# No Odio path input
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
algo_choice = ["Logistic Regression", "Decision Tree", "Naive Bayes", "Gradient Boosted Tree"]
algo_select = ttk.Combobox(trainFrame, textvariable=algorithm, state="readonly", values=algo_choice)
algo_select.grid(column=1, row=2, sticky=(E,W))
# Execute button
train_btn = ttk.Button(trainFrame, text="Entrenar", default="active", command=train)
train_btn.grid(column=2, columnspan=2, row=2, sticky=(E,W))

tp_label = Label(trainFrame, text="TP", bg="green")
tp_label.grid(column=2, row=9, sticky=N)
fp_label = Label(trainFrame, text="FP", bg="red")
fp_label.grid(column=3, row=9, sticky=N)
fn_label = Label(trainFrame, text="FN", bg="red")
fn_label.grid(column=2, row=9)
tn_label = Label(trainFrame, text="TN", bg="green")
tn_label.grid(column=3, row=9)
train_time = Label(trainFrame, text="")
train_time.grid(column=2, columnspan=2, row=7, sticky=W)

# padding
for child in trainFrame.winfo_children(): 
    child.grid_configure(padx=5, pady=5)

# Summary
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

# Performance
ttk.Label(trainFrame, text="Performance:").grid(column=0, row=9, sticky=W)
performanceText = Text(trainFrame, height=8)
performanceText.grid(column=1, row=9, sticky=W)

# Save model
ttk.Label(trainFrame, text="Guardar modelo:").grid(column=0, row=10, sticky=W)
# No Odio path input
model_path = StringVar()
model_path_input = ttk.Entry(trainFrame, textvariable=model_path)
model_path_input.grid(column=1, row=10, columnspan=2, sticky=(W, E))
model_path_input.insert(0, "/path/to/model.file")
# Odio Button
ttk.Button(trainFrame, text="Guardar", command=saveModel).grid(column=3, row=10, sticky=E)

# growing
trainFrame.columnconfigure(0, weight=1, minsize=150)
trainFrame.columnconfigure(1, weight=2, minsize=100)
trainFrame.columnconfigure(2, weight=2, minsize=50)
trainFrame.columnconfigure(3, weight=1, minsize=100)

# --------------------------------------------------------
# ------- Clasificador -----------------------------------
# --------------------------------------------------------

def getUnlabeledDirectory():
    noticias_input.delete(0, 'end')
    noticias_input.insert(0, filedialog.askdirectory(title="Elige carpeta para clasificar"))

def getModel():
    model_input.delete(0, 'end')
    model_input.insert(0, filedialog.askopenfilename(title="Elige modelo del clasificador"))

def handleTableClick(e):
    filename, odio_pred = tree.item(tree.focus())["values"]
    pred_text = " (no odio)"
    if odio_pred == "1.0": pred_text = " (odio)"
    noticia = "no podia encontrar la noticia"
    path = os.path.join(noticias_input.get(), filename)
    if os.path.isfile(path):
        with open(path, 'r', encoding=_encoding) as f:
            noticia = f.read()
    openNewWindow(filename + pred_text, noticia)
    
def clasify():
    try:
        global result
        clas_btn["text"] = "Calculando..."
        clasiFrame.update()
        begin = time.time()
        df = pd.DataFrame({"name": [], "odio": [], "content":[]})
        df = df.append(generateDF(noticias_input.get(), 1))

        df['tokens'] = df['content'].map(tokenize)
        df['tokens'] = df['tokens'].map(removeStopwords)
        df['lemma'] = df['tokens'].map(stemming)
        result = DataFrame()
        result["name"] = df["name"].copy()

        if(os.path.isfile(model_input.get())):
            model = pickle.load(open(model_input.get(), 'rb'))
            tfidf = pickle.load(open(model_input.get() + "_vector", 'rb'))
            y_pred = model.predict(tfidf.transform(df["lemma"]))
            result["prediction"] = y_pred

            tree.delete(*tree.get_children())
            #add data 
            for index, row in result.iterrows():
                tree.insert(parent='',index='end',iid=index,text='',values=(row["name"],row["prediction"]))
        else:
            messagebox.showwarning("Modelo no encontrado","No podia cargar el modelo, checka la ruta")
        end = time.time()
        time_label["text"] = f"tiempo: {end - begin: .2f}s"
    except FileNotFoundError:
        messagebox.showwarning("Can't resolve path", "At least one of the paths you provided isn't valid")
    except Exception as e:
        messagebox.showerror("Unexpected error", str(e))
    clas_btn["text"] = "Clasificar"

def openNewWindow(title="Titulo", content="Texto"):
    newWindow = Toplevel(root)
    newWindow.title(title)
    newWindow.geometry("400x400")
    text = Text(newWindow)
    text.pack(fill=BOTH)
    text.insert("1.0", content)

def saveResults():
    try:
        result.to_csv(results_input.get(), index=FALSE)
        messagebox.showinfo("Resultos Guardados", "Resultos guardados!")
    except:
        messagebox.showwarning("Ruta incorecto","No se puede guardar el modelo en esta ruta: " + results_input.get())

# noticias Label
ttk.Label(clasiFrame, text="Noticias para clasificar:").grid(column=0, row=0, sticky=W)
# Odio path input
noticias_path = StringVar()
noticias_input = ttk.Entry(clasiFrame, textvariable=noticias_path)
noticias_input.insert(0, "/path/to/directory/")
noticias_input.grid(column=1, row=0, columnspan=2, sticky=(W, E))
# noticias Button
ttk.Button(clasiFrame, text="Abrir", command=getUnlabeledDirectory).grid(column=3, row=0, sticky=E)

# Model Label
ttk.Label(clasiFrame, text="Model clasificador:").grid(column=0, row=1, sticky=W)
# Model input
model_path = StringVar()
model_input = ttk.Entry(clasiFrame, textvariable=model_path)
model_input.insert(0, "/path/to/model.file")
model_input.grid(column=1, row=1, columnspan=2, sticky=(W, E))
# model Button
ttk.Button(clasiFrame, text="Abrir", command=getModel).grid(column=3, row=1, sticky=E)

# Execute button
clas_btn = ttk.Button(clasiFrame, text="Clasificar", default="active", command=clasify)
clas_btn.grid(column=3, row=2, sticky=(E,W,N))

# Time Label
time_label = ttk.Label(clasiFrame, text="")
time_label.grid(column=3, row=2, sticky=(E))

#define our column
tree = ttk.Treeview(clasiFrame)
tree['columns'] = ('name', 'odio',)

# format our column
tree.column("#0", width=0,  stretch=NO)
tree.column("name",anchor=W, width=200)
tree.column("odio",anchor=CENTER,width=40)

#Create Headings 
tree.heading("#0",text="",anchor=CENTER)
tree.heading("name",text="Name",anchor=W)
tree.heading("odio",text="Odio",anchor=CENTER)

tree.grid(column=0, columnspan=3, row=2, sticky=(E,W))
tree.bind("<Double-1>", handleTableClick)

# save results
ttk.Label(clasiFrame, text="Guardar Resultados").grid(column=0, row=3, sticky=W)
results_var = StringVar()
results_input = ttk.Entry(clasiFrame, textvariable=results_var)
results_input.insert(0, "/my/project/results.csv")
results_input.grid(column=1, row=3, columnspan=2, sticky=(W, E))
# noticias Button
ttk.Button(clasiFrame, text="Guardar", command=saveResults).grid(column=3, row=3, sticky=E)

# growing
clasiFrame.columnconfigure(0, weight=1, minsize=150)
clasiFrame.columnconfigure(1, weight=2, minsize=100)
clasiFrame.columnconfigure(2, weight=2, minsize=50)
clasiFrame.columnconfigure(3, weight=1, minsize=100)

# padding
for child in clasiFrame.winfo_children(): 
    child.grid_configure(padx=5, pady=5)

root.geometry("800x450")
root.title("Python classificador")
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
# no_odio_input.insert(0, "/Users/joelplambeck/Documents/ZHAW/5_Semester/Proyecto Computacion/Plambeck_Joel.PC1.A3/RapidMiner/No odio")
# odio_input.insert(0, "/Users/joelplambeck/Documents/ZHAW/5_Semester/Proyecto Computacion/Plambeck_Joel.PC1.A3/RapidMiner/Odio")
# model_path_input.insert(0, "/Users/joelplambeck/Documents/clasificador_python/model.clf")
# noticias_input.insert(0, "/Users/joelplambeck/Documents/ZHAW/5_Semester/Proyecto Computacion/Plambeck_Joel.PC1.A3/RapidMiner/unlabeled")
# model_input.insert(0, "/Users/joelplambeck/Documents/clasificador_python/model.clf")
root.mainloop()