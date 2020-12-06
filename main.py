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
from numpy.core.defchararray import isdigit
from tkinter import messagebox
# from tkinter.tix import *
# from probandoScroll import ScrolledFrame
from tkinter import ttk

# 5. Mostrar el K optimo (el valor de true)
# 6. Mostrar la exactitud promedio

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
    if(len(df)>=10):
        optimos = crossVal(df)
        pos = 26
        tk.Label(second_frame, 
                 text="K").grid(row=25,column=0)
        tk.Label(second_frame, 
                 text="Promedios").grid(row=25,column=1)
        tk.Label(second_frame, 
                 text="Optimo/s").grid(row=25,column=2)
        kRankingTable = Table(second_frame,optimos[1],pos)
        tk.Label(text="Promedio: %.2f"%(optimos[0])).grid(row=26,column=3)
        tk.Label(text="K óptimo Verdadero: "+str((optimos[2])[0])).grid(row=27,column=3)
        tk.Label(text="Exactitud K Óptimo: %.2f"%optimos[2][1]).grid(row=28,column=3)

    else:
        toplevel = Toplevel()
        label1 = Label(toplevel, text='Ocurrió un error, el dataset tiene', height=0, width=50)
        label1.pack()
        label2 = Label(toplevel, text='que tener al menos 10 datos. Para Calcular el valor optimo utlizamos 10-fold cross validation', height=0, width=50)
        label2.pack()       



def getGraph(df, main, canvas):
    KText = e1.get()
    if(len(KText)>0 and len(e4.get())==0 and len(e5.get())==0):
        if(len(e2.get())>0):
            csvUtils = cu.CSVUtilities()
            tupleToPrint = csvUtils.getTupleToPrint(df)
            cabeceras = csvUtils.getHeaders(df)
            minValue = csvUtils.getMin(df)
            maxValue = csvUtils.getMax(df)
            tags = csvUtils.getTags(df)
            try:
                K = int(e1.get())
            except:
                toplevel = Toplevel()
                label1 = Label(toplevel, text='Ocurrio un error', height=0, width=50)
                label1.pack()
                label2 = Label(toplevel, text='El valor de K debe ser un entero', height=0, width=50)
                label2.pack()
            paso = e2.get()
            if ',' in paso:
                paso = paso.replace(",", ".")
            try:
                step = float(paso)
            except:
                toplevel = Toplevel()
                label1 = Label(toplevel, text='Ocurrio un error', height=0, width=50)
                label1.pack()
                label2 = Label(toplevel, text='El valor del Step debe ser un número', height=0, width=50)
                label2.pack()
            if ( step < ((maxValue/53.6486347)*0.7) ):
                MsgBox = tk.messagebox.askquestion ('Exit Application','El step ingresado es mucho menor al recomendado. Esto afectará considerablemente al tiempo de ejecución o podría provocar desbordamientos de memoria. ¿Está eguro que desea continuar?',icon = 'warning')
                if MsgBox == 'yes':
                    plotter = msg.Plotter()
                    canvas.delete('all')
                    plotter.plotKnnGraphic(*tupleToPrint, K=K, minValue=minValue, maxValue=maxValue, step=step, etiquetas=tags, x_label=cabeceras[0], y_label=cabeceras[1], main=main)
            else: 
                plotter = msg.Plotter()
                plotter.plotKnnGraphic(*tupleToPrint, K=K, minValue=minValue, maxValue=maxValue, step=step, etiquetas=tags, x_label=cabeceras[0], y_label=cabeceras[1], main=main)
        else:
            toplevel = Toplevel()
            label1 = Label(toplevel, text='Ocurrio un error', height=0, width=50)
            label1.pack()
            label2 = Label(toplevel, text='Debe ingresar el valor del Step', height=0, width=50)
            label2.pack()
    elif( (len(e4.get()) > 0) and (len(e5.get()) > 0) and (len(e1.get())==0) ):
        if(len(e2.get())>0):
            aux=[]
            try:
                e4int = int(e4.get())
                e5int = int(e5.get())
                if (e4int >= e5int):
                    toplevel = Toplevel()
                    label1 = Label(toplevel, text='Ocurrio un error', height=0, width=50)
                    label1.pack()
                    label2 = Label(toplevel, text='El valor de "Rango K Desde" debe ser menor que "Rango K Hasta"', height=0, width=65)
                    label2.pack()
            except:
                toplevel = Toplevel()
                label1 = Label(toplevel, text='Ocurrio un error', height=0, width=50)
                label1.pack()
                label2 = Label(toplevel, text='El valor de K debe ser un entero', height=0, width=50)
                label2.pack()
            for k in range(int(e4.get()), int(e5.get()) + 1):
                csvUtils = cu.CSVUtilities()
                tupleToPrint = csvUtils.getTupleToPrint(df)
                cabeceras = csvUtils.getHeaders(df)
                minValue = csvUtils.getMin(df)
                maxValue = csvUtils.getMax(df)
                tags = csvUtils.getTags(df)
                try:
                    K = int(k)
                except:
                    toplevel = Toplevel()
                    label1 = Label(toplevel, text='Ocurrio un error', height=0, width=50)
                    label1.pack()
                    label2 = Label(toplevel, text='El valor de K debe ser un entero', height=0, width=50)
                    label2.pack()
                paso = e2.get()
                if ',' in paso:
                    paso = paso.replace(",", ".")
                try:
                    step = float(paso)
                except:
                    toplevel = Toplevel()
                    label1 = Label(toplevel, text='Ocurrio un error', height=0, width=50)
                    label1.pack()
                    label2 = Label(toplevel, text='El valor del Step debe ser un número', height=0, width=50)
                    label2.pack()
                plotter = msg.Plotter()
                Kk=K
                minValue=minValue
                maxValue=maxValue
                step=step
                etiquetas=tags
                x_label=cabeceras[0]
                y_label=cabeceras[1]
                aux.append([*tupleToPrint, Kk, minValue, maxValue, step, etiquetas, x_label, y_label,int(e4.get()),int(e5.get()),len(tags)])
            plotter.variasGraficas(aux)
        else:
            toplevel = Toplevel()
            label1 = Label(toplevel, text='Ocurrio un error', height=0, width=50)
            label1.pack()
            label2 = Label(toplevel, text='Debe ingresar el valor del Step', height=0, width=50)
            label2.pack()            
    else:
        toplevel = Toplevel()
        label1 = Label(toplevel, text='Ocurrio un error', height=0, width=50)
        label1.pack()
        label2 = Label(toplevel, text='Recuerde ingresar el K, o si usa Rangos, no ingrese el valor de K individual', height=0, width=100)
        label2.pack()
    canvas.geometry("1120x700")

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
    label2 = Label(toplevel, text='Tenga en cuenta que cuando menor es el step, mayor es el tiempo de procesamiento.', height=0, width=100)
    label2.pack() 
    label4 = Label(toplevel, text='Recomendamos usar el valor sugerido', height=0, width=100)
    label4.pack()  

def getCSV (main, canvas):
    separator = e3.get()
    if(len(separator)>0):
        import_file_path = filedialog.askopenfilename()
        df = pd.read_csv(import_file_path,sep=separator,engine='python')
        data = df.values
        if(len(data[0])==3):
            csvButton.grid_remove()
            tk.Label(main, 
                 text="Vista Previa de los datos del CSV:").grid(row=10,column=1)
            tk.Label(main, 
                 text="X").grid(row=11,column=0)
            tk.Label(main, 
                     text="Y").grid(row=11,column=1)
            tk.Label(main, 
                 text="Clases").grid(row=11,column=2)
            # table = Table.limpiar(master)
            table = Table(main,[["","",""],["","",""],["","",""],["","",""],["","",""],["","",""],["","",""],["","",""],["","",""],["","",""],["","",""],],12)
            table = Table(main,data,12)
            tk.Button(main, 
                  text='Calcular K Optimo', command=lambda: kRanking(df)).grid(row=23, 
                                                               column=1, 
                                                               sticky=tk.W, 
                                                               pady=0)
            tk.Button(main, 
                  text='Graficar', command=lambda: getGraph(df, main, canvas=canvas)).grid(row=23, 
                                                               column=2, 
                                                               sticky=tk.W, 
                                                               pady=0)
            e2.delete(0, "end")
            csvUtils = cu.CSVUtilities()
            maxValue = csvUtils.getMax(df)
            stepRec = maxValue/53.6486347
            """ if(maxValue == 11.227815173855697):
                tk.Label(master,text='Step recomendado: %.4f'%(maxValue/93)).grid(row=5,column=0)
                e2.insert(0, maxValue/93)
            else: """
            tk.Label(main,text='Step recomendado: %.4f'%(stepRec)).grid(row=6,column=0)
            e2.insert(0, stepRec)
            # tk.Button(master,text='Step Recomendado', command=lambda: getStepRecomendado(df)).grid(row=5, column=1, pady=0)
            tk.Button(main,text='Cargar Otro Archivo', command= lambda: getCSV(main=second_frame, canvas=master)).grid(row=6, column=2, pady=0)
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
    canvas.geometry("600x500")

def getStepRecomendado(df):
    toplevel = Toplevel()
    label1 = Label(toplevel, text='El valor recomendado del step es:', height=0, width=50)
    label1.pack()
    label2 = Label(toplevel, text=str(130/(len(df.values))), height=0, width=50)
    label2.pack() 

def update_scrollregion(event):
    my_canvas.configure(scrollregion=my_canvas.bbox("all"))

master = tk.Tk()
master.geometry("450x220")

# Create a Mainframe
main_frame = Frame(master)
main_frame.pack(fill=BOTH, expand=1)

# Create a Canvas
my_canvas = Canvas(main_frame)
my_canvas.pack(side=LEFT, fill=BOTH, expand=1)

# Add a scrollbar to the Canvas
my_scrollbar = ttk.Scrollbar(main_frame, orient=VERTICAL, command=my_canvas.yview)
my_scrollbar.pack(side=RIGHT, fill=Y)

# Configure the Canvas
my_canvas.configure(yscrollcommand=my_scrollbar.set)
my_canvas.bind('<Configure>', update_scrollregion)

# Create another frame inside the canvas
second_frame = Frame(my_canvas)

# Add that new frame to a window inside the canvas
my_canvas.create_window((0,0), window=second_frame, anchor="nw")

master.title("KNN")
tk.Label(second_frame, 
         text="Valor de K").grid(row=1)
tk.Label(second_frame, 
         text="Step del Grid (Ej: 0.2, 0.5)").grid(row=4,column=0)
tk.Label(second_frame, 
         text="Separador CSV (Ej: , ; -)").grid(row=5,column=0)

aboutK = tk.Button(second_frame,text='K?', command=clickAboutK)
aboutK.grid(row=1,column=1)

aboutStep = tk.Button(second_frame,text='Step?', command=clickAboutStep)
aboutStep.grid(row=4,column=1)

tk.Label(second_frame, 
         text="Rango K Desde").grid(row=2,column=0)
e4 = tk.Entry(second_frame, state=tk.DISABLED)
e4.grid(row=2, column=2)
tk.Label(second_frame, 
         text="Rango K Hasta").grid(row=3,column=0)
e5 = tk.Entry(second_frame, state=tk.DISABLED)
e5.grid(row=3, column=2)

comma = tk.StringVar()
comma.set( "," )
comma = tk.StringVar()
comma.set( "," )
e1 = tk.Entry(second_frame)
e2 = tk.Entry(second_frame)
e3 = tk.Entry(second_frame, textvariable=comma)
e1.grid(row=1, column=2)
e2.grid(row=4, column=2)
e3.grid(row=5, column=2)


# Create a Tkinter variable
tkvar = StringVar(second_frame)
# Dictionary with options
choices = { 'Único', 'Rango' }
tkvar.set('Único') # set the default option
popupMenu = OptionMenu(second_frame, tkvar, *choices)
Label(second_frame, text="K a graficar").grid(row = 0, column = 0)
popupMenu.grid(row = 0, column =1)
# on change dropdown value
def option_changed(e1, e4, e5):
    print( tkvar.get() )
    if (tkvar.get() == 'Rango'):
        e1.delete(0, "end")
        e1.configure(state=tk.DISABLED)
        e4.configure(state=tk.NORMAL)
        e5.configure(state=tk.NORMAL)
        e1.grid(row=1, column=2)
        e4.grid(row=2, column=2)
        e5.grid(row=3, column=2)
    else:
        e4.delete(0, "end")
        e5.delete(0, "end")
        e1.configure(state=tk.NORMAL)
        e4.configure(state=tk.DISABLED)
        e5.configure(state=tk.DISABLED)
        e1.grid(row=1, column=2)
        e4.grid(row=2, column=2)
        e5.grid(row=3, column=2)
# link function to change dropdown
tkvar.trace('w', lambda *args: option_changed(e1, e4, e5))


csvButton = tk.Button(second_frame,text='Cargar Archivo CSV',  command= lambda: getCSV(main=second_frame, canvas=master))
csvButton.grid(row=6, column=0, pady=4)



""" def update_scrollregion(event):
    photoCanvas.configure(scrollregion=photoCanvas.bbox("all"))

photoFrame = tk.Frame(tk, width=250, height=190, bg="#EBEBEB")
photoFrame.grid()
photoFrame.rowconfigure(0, weight=1) 
photoFrame.columnconfigure(0, weight=1) 

photoCanvas = tk.Canvas(photoFrame, bg="#EBEBEB")
photoCanvas.grid(row=0, column=0, sticky="nsew")

canvasFrame = tk.Frame(photoCanvas, bg="#EBEBEB")
photoCanvas.create_window(0, 0, window=canvasFrame, anchor='nw')

photoScroll = tk.Scrollbar(photoFrame, orient=tk.VERTICAL)
photoScroll.config(command=photoCanvas.yview)
photoCanvas.config(yscrollcommand=photoScroll.set)
photoScroll.grid(row=0, column=1, sticky="ns")

canvasFrame.bind("<Configure>", update_scrollregion) """

""" masterFrame = tk.Frame(master, height=100)
scrollbar = Scrollbar(masterFrame)
scrollbar.grid(sticky=E, row = 0, rowspan = 100, column = 11, ipady = 1000) """

""" masterFrame = tk.Frame(master, height=100)
swin = ScrolledWindow(masterFrame, width=500, height=500) """

""" treedata = [('column 1', 'column 2'), ('column 1', 'column 2')]
column_names = ("heading1", "heading2")
scrollbar = tk.Scrollbar(master)
tree = tk.Treeview(master, columns = column_names, yscrollcommand = scrollbar.set)
scrollbar.pack(side = 'right', fill= Y)
for x in treedata:
    tree.insert('', 'end', values =x)
for col in column_names: 
    tree.heading(col, text = col.Title())
scrollbar.config(command=tree.yview)
tree.pack() """

""" container = tk.Frame(master)
canvas = tk.Canvas(container)
scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas)
scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set) """


""" master.geometry("1000x775")
container = tk.Frame(master)
canvas = tk.Canvas(container, height=240)
scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.grid(row=0, column=0)
scrollbar.grid(row=2, column=1, sticky="ns") """

# master.geometry("240x130+600+250")

""" scrollbar = tk.Scrollbar(master)
scrollbar.grid(row=0, column=1, sticky="ns")

listbox = tk.Listbox(master, yscrollcommand=scrollbar.set)
# load the listbox
for n in range(1, 26):
   listbox.insert('end', "This is line number " + str(n))

listbox.grid(row=0, column=0, sticky="ns")
scrollbar.config(command=master.yview) """



# mFrame = Tk()
""" master.geometry("1000x775")

frame_canvas = Frame(master)
frame_canvas.grid(row=0, column=0, sticky='news')

Can1 = Canvas(frame_canvas, width=2000, height=2000)
Can1.grid(row=0, column=0)

vsbar = Scrollbar(frame_canvas, orient="vertical", command=Can1.yview)
vsbar.grid(row=0, column=1, sticky='ns')
Can1.configure(yscrollcommand=vsbar.set)

janela = Frame(Can1)
Can1.create_window((0,0), window=janela,anchor='nw')

class Frames(object):
    def __init__(self):
        pass
    def main_frame(self, janela):
        return janela
    def Antunes_frame(self):
        return self

app = Frames()
app.main_frame(master) """
# janela.mainloop()


""" scrollbar = Scrollbar(master) 
scrollbar.grid()

area = Text(master, yscrollcommand = scrollbar.set, background = 'black', foreground = 'green', font = ('Courier New', 11), insertbackground = 'yellow', insertwidth = 5, selectbackground = 'red' ) 
area.grid()

scrollbar.config(command = area.yview)  """

""" sf = ScrolledFrame(tk, True, True) # 
sf.pack(fill='both', expand=True) # changed

display_players(sf.frame, master) """

""" scrollbar = tk.Scrollbar(master)
scrollbar.grid()

listbox = tk.Listbox(master, yscrollcommand=scrollbar.set)
for i in range(20):
    listbox.insert(tk.END, str(i))
listbox.grid()

scrollbar.config(command=listbox.yview) """

####################### LA POSTA ################################################################
# canvas = Canvas(master)
""" scrollbar = Scrollbar(canvas, orient="vertical", command=canvas.yview)
canvas.grid(row=0, column=0, sticky=W+E+N+S) """

""" scrollbar = tk.Scrollbar(canvas, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas)

scrollable_frame = tk.Frame(canvas)
scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)
scrollbar.grid(row=0, column=1, sticky=N+S)
canvas.configure(width=500, height=600)
canvas.grid() """

""" scrollbar = tk.Scrollbar(canvas, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)
scrollbar.grid() """

tk.mainloop()