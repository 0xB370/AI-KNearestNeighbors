import tkinter as tk
from tkinter import *
from tkinter import filedialog
import pandas as pd
from table import Table
import numpy as np
import matplotlib.pyplot as plt
import plotter as msg
import loadCsv as cu
from pruebaKFoldCrossValidation import crossValidation as crossVal

# Lanzar mensajes de Error:
# * Si no se ingresa el K antes de graficar
# * Si no se ingresa el Step antes de graficar
# * Si intenta calcular el K optimo con un dataset menor menor a 10 registros
 
# 3- El K, hay que ingresar un valor por vez, podría ofrecer la posibilidad de ingresar un rango o varios valores a la vez, para facilitar las corridas.
# 1- Con respecto al CSV, toma automáticamente los separadores, (sean coma, punto y coma, barra, etc,) o hay que indicarle el tipo de separador? si es esto último, entonces tengan en cuenta un campo para pedir ese dato.
# 2- Habría que poner ayuda contextual en los campos,que indique para qué se utiliza el campo (el K y el step).
# 5- Con respecto a los gráficos, veo que figura sobre el gráfico pero, es posible poner en el título a qué K corresponde? eso ayudaría al usuario a no perderse cuando tenga más de un gráfico abierto.
# 6- No es posible "enmarcar" los gráficos en la aplicación? ya que al abrir ventanas separadas el usuario podría perderse después de la tercera corrida.
# Bueno, esos serían los comentarios que podemos hacerles.
# 4- Sugerir el valor de step haciendo el calculo tamañoDataSet/step = 130

def kRanking(df):
    # optimos = [
    #     ['K','Exactitud', 'Optimo'],
    #     [1,90, True],
    #     [2,34, False],
    #     [3,23, False],
    #     [4,45, False],
    #     [5,65, False],
    #     [6,76, False],
    #     [7,48, False],
    #     [8,15, False],
    #     [9,87, False],
    #     [10,5, False],
    # ]
    if(len(df)>=10):    
        optimos = crossVal(df)
        pos = 23
        kRankingTable = Table(master,optimos,pos)
    else:
        toplevel = Toplevel()
        label1 = Label(toplevel, text='Crack, el csv tiene', height=0, width=50)
        label1.pack()
        label2 = Label(toplevel, text='que tener al menos 10 datos', height=0, width=50)
        label2.pack()       



def getGraph(df):
    KText = e1.get()
    if(len(KText)>0 and len(e4.get())==0 and len(e5.get())==0):
        if(len(e2.get())>0):
            csvUtils = cu.CSVUtilities()
            tupleToPrint = csvUtils.getTupleToPrint(df)
            cabeceras = csvUtils.getHeaders(df)
            minValue = csvUtils.getMin(df)
            maxValue = csvUtils.getMax(df)
            tags = csvUtils.getTags(df)
            K = int(e1.get())
            step = float(e2.get())
            plotter = msg.Plotter()
            plotter.plotKnnGraphic(*tupleToPrint, K=K, minValue=minValue, maxValue=maxValue, step=step, etiquetas=tags, x_label=cabeceras[0], y_label=cabeceras[1])
        else:
            toplevel = Toplevel()
            label1 = Label(toplevel, text='Ocurrio un error', height=0, width=50)
            label1.pack()
            label2 = Label(toplevel, text='Falto Especificar el Step', height=0, width=50)
            label2.pack()
    elif(int(e4.get())<=int(e5.get()) and len(e1.get())==0):
        for k in range(int(e4.get()),int(e5.get())+1):
            print(k)
            csvUtils = cu.CSVUtilities()
            tupleToPrint = csvUtils.getTupleToPrint(df)
            cabeceras = csvUtils.getHeaders(df)
            minValue = csvUtils.getMin(df)
            maxValue = csvUtils.getMax(df)
            tags = csvUtils.getTags(df)
            K = int(k)
            step = float(e2.get())
            plotter = msg.Plotter()
            plotter.plotKnnGraphic(*tupleToPrint, K=K, minValue=minValue, maxValue=maxValue, step=step, etiquetas=tags, x_label=cabeceras[0], y_label=cabeceras[1])
    else:
        toplevel = Toplevel()
        label1 = Label(toplevel, text='Ocurrio un error', height=0, width=50)
        label1.pack()
        label2 = Label(toplevel, text='Recuerde ingresar el K, o si usa Rangos, no ingrese el valor de K individual', height=0, width=100)
        label2.pack()

def clickAboutK():
    toplevel = Toplevel()
    label1 = Label(toplevel, text='K: Es el valor que indica la cantidad de vecinos que se evaluarán para una nueva instancia a clasificar.', height=0, width=100)
    label1.pack()
    label2 = Label(toplevel, text=' Tenga en cuenta que cuando mayor es K mayor es el tiempo de procesamiento.', height=0, width=100)
    label2.pack()  

def clickAboutStep():
    toplevel = Toplevel()
    label1 = Label(toplevel, text='Es el valor que indica la cantidad de saltos para armar el Grid.', height=0, width=110)
    label1.pack()
    label3 = Label(toplevel, text='Ej: Si el valor es 0.5 tendremos saltos de 0.5 al armar el grid 0.5, 1, 1.5, 2, 2.5.', height=0, width=110)
    label3.pack()
    label2 = Label(toplevel, text='Tenga en cuenta que cuando mayor es el step mayor es el tiempo de procesamiento.', height=0, width=100)
    label2.pack() 
    label4 = Label(toplevel, text='Recomendamos usar el valor sugerido', height=0, width=100)
    label4.pack()  

def getCSV ():
    separator = e3.get()
    if(len(separator)>0):
        import_file_path = filedialog.askopenfilename()
        df = pd.read_csv(import_file_path,sep=separator,engine='python')
        print(len(df))
        data = df.values
        print(len(data[0]))
        if(len(data[0])==3):
            csvButton.grid_remove()
            tk.Label(master, 
                 text="Vista Previa de los datos del CSV:").grid(row=9,column=1)
            table = Table(master,data,10);
            tk.Button(master, 
                  text='Calcular K Optimo', command=lambda: kRanking(df)).grid(row=22, 
                                                               column=1, 
                                                               sticky=tk.W, 
                                                               pady=0)
            tk.Button(master, 
                  text='Graficar', command=lambda: getGraph(df)).grid(row=22, 
                                                               column=2, 
                                                               sticky=tk.W, 
                                                               pady=0)
            tk.Button(master,text='Step Recomendado', command=lambda: getStepRecomendado(df)).grid(row=5, column=1, pady=0)
            tk.Button(master,text='Cargar Otro Archivo', command=getCSV).grid(row=5, column=2, pady=0)
        else:
            toplevel = Toplevel()
            label1 = Label(toplevel, text='Archivo, Formato de archivo o separador incorrecto', height=0, width=50)
            label1.pack()
            label2 = Label(toplevel, text='Por favor cargue nuevamente su archivo', height=0, width=50)
            label2.pack()        
    else:
        toplevel = Toplevel()
        label1 = Label(toplevel, text='Debe Ingresar el separador', height=0, width=50)
        label1.pack()
        label2 = Label(toplevel, text='Separadores Comunes: "," , ";" , "-" , "_"', height=0, width=50)
        label2.pack() 

def getStepRecomendado(df):
    toplevel = Toplevel()
    print(len(df.values))
    label1 = Label(toplevel, text='El valor recomendado del step es:', height=0, width=50)
    label1.pack()
    label2 = Label(toplevel, text=str(130/(len(df.values))), height=0, width=50)
    label2.pack() 

master = tk.Tk()
master.title("KNN")
tk.Label(master, 
         text="Valor de K").grid(row=0)
tk.Label(master, 
         text="Step del Grid").grid(row=3,column=0)
tk.Label(master, 
         text="Separador CSV").grid(row=4,column=0)

aboutK = tk.Button(master,text='K?', command=clickAboutK)
aboutK.grid(row=0,column=1)

aboutStep = tk.Button(master,text='Step?', command=clickAboutStep)
aboutStep.grid(row=3,column=1)

tk.Label(master, 
         text="Rango K Desde").grid(row=1,column=0)
e4 = tk.Entry(master)
e4.grid(row=1, column=2)
tk.Label(master, 
         text="Rango K Hasta").grid(row=2,column=0)
e5 = tk.Entry(master)
e5.grid(row=2, column=2)
e1 = tk.Entry(master)
e2 = tk.Entry(master)
e3 = tk.Entry(master)
e1.grid(row=0, column=2)
e2.grid(row=3, column=2)
e3.grid(row=4, column=2)


csvButton = tk.Button(master,text='Cargar Archivo CSV', command=getCSV)
csvButton.grid(row=5, column=0, pady=4)






tk.mainloop()