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
        [1,56, True]
    ]
    pos = 22
    kRankingTable = Table(master,optimos,pos);    

def getGraph(df):
    csvUtils = cu.CSVUtilities()
    tupleToPrint = csvUtils.getTupleToPrint(df)
    minValue = csvUtils.getMin(df)
    maxValue = csvUtils.getMax(df)
    print(tupleToPrint)
    plotter = msg.Plotter()
    plotter.plotKnnGraphic(*tupleToPrint, K=5, minValue=minValue, maxValue=maxValue, step=0.5)
    
    
    
def getCSV ():
    
    
    import_file_path = filedialog.askopenfilename()
    df = pd.read_csv(import_file_path)
    
   
    data = df.values
    csvButton.grid_remove()
    tk.Label(master, 
         text="Vista Previa de los datos del CSV:").grid(row=6,column=1)
    table = Table(master,data,7);
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
    

master = tk.Tk()
master.title("KNN")
tk.Label(master, 
         text="Valor de K").grid(row=0)
tk.Label(master, 
         text="Step X").grid(row=1)
tk.Label(master, 
         text="Step Y").grid(row=2)

e1 = tk.Entry(master)
e2 = tk.Entry(master)
e3 = tk.Entry(master)
e1.grid(row=0, column=1)
e2.grid(row=1, column=1)
e3.grid(row=2, column=1)

csvButton = tk.Button(master,text='Cargar Archivo CSV', command=getCSV)
csvButton.grid(row=3, column=0, pady=4)




tk.mainloop()