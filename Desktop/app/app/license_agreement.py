"""
This Source Code Form is subject to the terms of the Mozilla
Public License, v. 2.0. If a copy of the MPL was not distributed
with this file, You can obtain one at https://github.com/DeQB/rnn_pks-17-2c.
© Кобелев Д. Е.
"""

from tkinter import *

def license_agreement():
    top2=Toplevel()
    top2.title('MPL v2.0 (Mozilla Public License Version 2.0)')
    top2.geometry('500x700+800+50')
    top2.resizable(False, False)

    with open("config/license.txt", "r", encoding='utf-8-sig') as file:
        license_txt = file.read()


    frame = Frame(top2, bd=2, relief=SUNKEN)

    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)


    yscrollbar = Scrollbar(frame)
    yscrollbar.grid(row=0, column=1, sticky=N + S)

    text_info = Text(frame, wrap=NONE, width=52, height=42, padx=5, pady=5, yscrollcommand=yscrollbar.set)
    text_info.insert(INSERT, f'{license_txt}')
    text_info.configure(state='disabled', wrap=WORD)


    text_info.grid(row=0, column=0, sticky=N + S + E + W)


    yscrollbar.config(command=text_info.yview)

    frame.pack()

