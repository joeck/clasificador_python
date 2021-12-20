from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import os

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

def train():
    pass

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