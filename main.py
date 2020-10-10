import tkinter as tk
from tkinter import filedialog
import pandas as pd
from table import Table
import numpy as np
import matplotlib.pyplot as plt
import pruebaMeshgrid as msg
import loadCsv as cu

def show_entry_fields():
    print("First Name: %s\nLast Name: %s" % (e1.get(), e2.get()))

def kRanking():
    optimos = [
        ['K','Exactitud', 'Optimo'],
        [1,90, True],
        [1,34, False],
        [1,23, False],
        [1,45, False],
        [1,65, False],
        [1,76, False],
        [1,48, False],
        [1,15, False],
        [1,36, False],
        [9,90, False],
        [10,5, False],
    ]
    pos = 22
    
    kRankingTable = Table(master,optimos,pos);    

def getGraph(df):
    csvUtils = cu.CSVUtilities()
    tupleToPrint = csvUtils.getTupleToPrint(df)
    minValue = csvUtils.getMin(df)
    maxValue = csvUtils.getMax(df)
    tags = csvUtils.getTags(df)
    K = int(e1.get())
    step = float(e2.get())
    print(tupleToPrint)
    plotter = msg.Plotter()
    
    plotter.plotKnnGraphic(*tupleToPrint, K=K, minValue=minValue, maxValue=maxValue, step=step, etiquetas=tags)
    
    
    
def getCSV ():
    
    
    import_file_path = filedialog.askopenfilename()
    df = pd.read_csv(import_file_path)
    
   
    data = df.values
    csvButton.grid_remove()
    tk.Label(master, 
         text="Vista Previa de los datos del CSV:").grid(row=9,column=1)
    table = Table(master,data,10);
    tk.Button(master, 
          text='Calcular K Optimo', command=kRanking).grid(row=20, 
                                                       column=1, 
                                                       sticky=tk.W, 
                                                       pady=4)
    tk.Button(master, 
          text='Graficar', command=lambda: getGraph(df)).grid(row=21, 
                                                       column=1, 
                                                       sticky=tk.W, 
                                                       pady=4)
    tk.Button(master,text='Cargar Otro Archivo', command=getCSV).grid(row=22, column=0, pady=4)

    

master = tk.Tk()
master.title("KNN")
tk.Label(master, 
         text="Valor de K").grid(row=0)
tk.Label(master, 
         text="Step del Grid").grid(row=1)

e1 = tk.Entry(master)
e2 = tk.Entry(master)
e1.grid(row=0, column=1)
e2.grid(row=1, column=1)


csvButton = tk.Button(master,text='Cargar Archivo CSV', command=getCSV)
csvButton.grid(row=3, column=0, pady=4)




tk.mainloop()