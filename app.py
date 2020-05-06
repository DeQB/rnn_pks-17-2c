from tkinter import *
from tkinter.ttk import Combobox
import matplotlib.pyplot as plt
import datetime as dt
import fxcmpy
from tkinter import messagebox as mb
from tkinter.font import Font
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from defs_graph import open_window
from license_agreement import license_agreement
from faq import faq


def list_EUR(): #функция присваивания валютной пары
    valueEUR = comboEUR.get()
    print(valueEUR)

def P(): #функция присваивания периода
    valueP = comboP.get()

    if (valueP == 'День'):
        valueP='D1'
    elif (valueP == 'Неделя'):
        valueP='W1'
    elif (valueP == 'Месяц'):
        valueP='M1'
    else:
        mb.showerror('Error', 'Вы ввели неверное значение!')
    return valueP

#Выбор Точек старта и конца
def listYearStart():
    valueYStart = comboYStart.get()
    print(valueYStart)


def listDayStart():
    valueDStart = comboDStart.get()
    print(valueDStart)


def listMonthStart():
    valueMStart = comboMStart.get()
    print(valueMStart)


def listYearEnd():
    valueYEnd = comboYEnd.get()
    print(valueYEnd)


def listDayEnd():
    valueDEnd = comboDEnd.get()
    print(valueDEnd)


def listMonthEnd():
    valueMEnd = comboMEnd.get()
    print(valueMEnd)

def _quit():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate

#подключение|построение графика

def connect_fxcmpy():
    valueP = P()
    con = fxcmpy.fxcmpy(config_file='fxcm.cfg', server='demo')

    con.is_connected()
    if con.is_connected():
        print('True')
        mb.showinfo('Connected', 'Вы успешно подключились!')
    else:
        mb.showerror('Error', 'Ошибка подключения!')

    ys=int(comboYStart.get())
    ms=int(comboMStart.get())
    ds=int(comboDStart.get())
    ye=int(comboYEnd.get())
    me=int(comboMEnd.get())
    de=int(comboDEnd.get())
    start = dt.datetime(ys, ms, ds, 0, 0, 0)
    end = dt.datetime(ye, me, de, 0, 0, 0)

    c = con.get_candles(f'{comboEUR.get()}', period=f'{valueP}', start=start, end=end)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(c.index, c['bidopen'], lw=1, color='blue', label="open")
    ax2 = ax.twinx()
    ax2.plot(c.index, c['tickqty'], lw=1, color='green', label="Volume")

    canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.

    canvas.get_tk_widget().grid(row=1, column=10, rowspan=12, columnspan=12)

    """""Запись данных в файлы"""""
    series = c['bidopen']
    values = c['tickqty']
    df = pd.DataFrame(c)
    df.to_csv('c_data.csv')
    df2 = pd.DataFrame(series)
    df2.to_csv('series.csv')
    df3 = pd.DataFrame(values)
    df3.to_csv('values.csv')


root=Tk()
root.title('Forecasting from DeQB v.0.1')

"""Меню программы"""
main_menu= Menu(root)
root.config(menu=main_menu)


#creating file_menu
fileMenu=Menu(main_menu)
main_menu.add_cascade(label='Программа', menu=fileMenu)

#edit menu
editMenu=Menu(main_menu)
main_menu.add_cascade(label='ProTools', menu=editMenu)

fileMenu.add_command(label='Лицензионное соглашение', command=license_agreement)
fileMenu.add_command(label='Справка', command=faq)
fileMenu.add_separator()
fileMenu.add_command(label='Выход', command=_quit)
"""Меню программы"""

my_font=Font(family='Halvetica', size=10, weight='bold')

#Выбор валютной пары
labelEUR=Label(root, text='  Валютная пара', font=my_font)
labelEUR.grid(row=1, column=1)


vEUR = ['EUR/USD', 'USD/JPY', 'GBP/USD', 'USD/CHF',
   'EUR/CHF', 'AUD/USD', 'USD/CAD', 'NZD/USD',
   'EUR/GBP', 'EUR/JPY', 'GBP/JPY', 'AUD/JPY',
   'GBP/AUD', 'USD/CNH', 'XAU/USD', 'XAG/USD']
comboEUR= Combobox(root, values=vEUR, width=10)
comboEUR.set('EUR/USD')
comboEUR.grid(row=2, column=1)


#Точки старта и конца
#start
labelStart=Label(root, text=' \nТочка старта', font=my_font)
labelStart.grid(row=3, column=1)

labelYStart=Label(root, text='Год')
labelYStart.grid(row=4, column=1)

labelYStart=Label(root, text='День')
labelYStart.grid(row=4, column=2)

labelYStart=Label(root, text='Месяц')
labelYStart.grid(row=4, column=3)

vYStart = list(range(2000, 2021))
comboYStart = Combobox(root, values=vYStart, width=10)
comboYStart.set('2019')
comboYStart.grid(row=5, column=1)

vDStart = list(range(0, 32))
comboDStart = Combobox(root, values=vDStart, width=5)
comboDStart.set('1')
comboDStart.grid(row=5, column=2)

vMStart = list(range(0, 13))
comboMStart = Combobox(root, values=vMStart, width=5)
comboMStart.set('1')
comboMStart.grid(row=5, column=3)

#end
labelEnd=Label(root, text=' \nТочка конца', font=my_font)
labelEnd.grid(row=6, column=1)

labelYEnd=Label(root, text='Год')
labelYEnd.grid(row=7, column=1)

labelYEnd=Label(root, text='День')
labelYEnd.grid(row=7, column=2)

labelYEnd=Label(root, text='Месяц')
labelYEnd.grid(row=7, column=3)

vYEnd = list(range(2000, 2021))
comboYEnd = Combobox(root, values=vYEnd, width=10)
comboYEnd.set('2020')
comboYEnd.grid(row=8, column=1)

vDEnd = list(range(0, 32))
comboDEnd = Combobox(root, values=vDEnd, width=5)
comboDEnd.set('1')
comboDEnd.grid(row=8, column=2)

vMEnd = list(range(0, 13))
comboMEnd = Combobox(root, values=vMEnd, width=5)
comboMEnd.set('1')
comboMEnd.grid(row=8, column=3)

#Период
labelP=Label(root, text='\nПериод', font=my_font)
labelP.grid(row=9, column=1)

vP = ['День', 'Неделя', 'Месяц']
comboP= Combobox(root, values=vP, width=10)
comboP.set('День')
comboP.grid(row=10, column=1)


# #кнопка соединения
button1=Button(root, text='Подключиться', command=connect_fxcmpy, width=15)
button1.grid(row=12, column=2)
#Canvas
canvas = Canvas(root, width=700, height=500, bg='gray')
canvas.grid(row=1, column=10, rowspan=12, columnspan=12)

#quit
button5=Button(root, text='Выход', command=_quit, width=12)
button5.grid(row=21, column=21)


########################################################################
# #пустые label для отступа
labelSpace1=Label(root, text=' ')
labelSpace1.grid(row=0, column=1, columnspan=10)
labelSpace4=Label(root, text=' ')
labelSpace4.grid(row=11, column=1)
labelSpace3=Label(root, text=' ')
labelSpace3.grid(row=0, column=0)
labelSpace2=Label(root, text=' ')
labelSpace2.grid(row=1, column=0, rowspan=10)
labelSpace2=Label(root, text='                                       ')
labelSpace2.grid(row=1, column=4, rowspan=13, columnspan=6)
labelSpace2=Label(root, text=' ')
labelSpace2.grid(row=20, column=10, columnspan=2)
########################################################################



#Новое окно
button4=Button(root, text='Начать работу', command= open_window, width=12)
button4.grid(row=21, column=10)




#geo,loop
root.geometry('1200x600+10+40')
root.mainloop()
