import tkinter as tk
from tkinter import *
import timeit

root = tk.Tk()
root.geometry("500x250")
root.title("GUI")
root.configure(bg='black')

label2 = tk.Label(root, text= "Enter a review about the last movie you watched:", font= ('Helvetica 12 '))
label2.place(x=80 , y=70)

t = Text(root, height=1, width=40)
t.place(x=85, y=120)

b = tk.Button(root, text="Search")
b.place(x=220, y=160)

Label(root, text="Result : ", font=('Helvetica 10 ')).place(x=80, y=185)

display_canvas2 = tk.Canvas(root, bg="white", width=100, height=20)
display_canvas2.place(x=80, y=220)



root.mainloop()
