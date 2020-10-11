# Python program to create a table 

from tkinter import *

class Table: 
	
    def __init__(self,root,data,pos):
        #code for creating table
        total_rows = 11 if len(data) > 11 else len(data)
        total_columns = len(data[0])
         
        for i in range(total_rows): 
            for j in range(total_columns): 
                
                self.e = Entry(root, width=13, fg='Black', 
                            font=('Arial',9,'bold'))
                self.e.grid(row=i + pos, column=j) 
                self.e.insert(END, data[i][j])
  

# take the data 
lst = [(1,'Raj','Mumbai',19), 
	(2,'Aaryan','Pune',18), 
	(3,'Vaishnavi','Mumbai',20), 
	(4,'Rachna','Mumbai',21), 
	(5,'Shubham','Delhi',21)] 

# find total number of rows and 
# columns in list 
total_rows = len(lst) 
total_columns = len(lst[0]) 


sys.path.append(".")