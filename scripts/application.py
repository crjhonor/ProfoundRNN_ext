"""
Application scripts for the DPM
"""
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from utils.mainmenu import MainMenu
import numpy as np
import pandas as pd
import utils.tripledicer as tripledicer
import utils.readit as readit
import utils.persistencesavingmodels as psm

class Application(tk.Tk):
    """Application root window"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.persistencesavingmodel = psm.HyperparameterSaving()
        self._load_hyperparameters()

        self.title("DICINGER PRO MAX")
        self.columnconfigure(0, weight=1)

        # Hyper Parameters

        menu = MainMenu(self)
        self.configure(menu=menu)

        # Defining callbacks used in menu
        event_callbakcs = {
            '<<TRIPLE DICER>>': self._to_tripledicer_page,
            '<<Quit>>': lambda _: self.quit(),
            '<<Coming Soon>>': self._show_more
        }
        for sequence, callback in event_callbakcs.items():
            self.bind(sequence, callback)

        # Packing application frames
        self.mainpage = tripledicer.TripleDicer(self, self._hyper_params)
        self.mainpage.grid(row=0, padx=10, sticky=tk.W+tk.E)

        # Statusbar
        self.status = tk.StringVar()
        self.status.set('Process begins...')
        self.statusbar = ttk.Label(self, textvariable=self.status)
        self.statusbar.grid(row=99, padx=10, sticky=tk.W+tk.E)

    def _to_status(self, text):
        self.status.set(text)

    def _to_tripledicer_page(self, *_):
        print("Done")

    def _load_hyperparameters(self):
        """Load the hyperparameters into dictionary"""

        vartypes = {
            'bool': tk.BooleanVar,
            'str': tk.StringVar,
            'int': tk.IntVar,
            'float': tk.DoubleVar
        }
        self._hyper_params = dict()
        for key, data in self.persistencesavingmodel.fields.items():
            vartype = vartypes.get(data['type'], tk.StringVar)
            if key in readit.values_fields_list:
                self._hyper_params[key] = vartype(value=data['values'])
            else:
                self._hyper_params[key] = vartype(value=data['value'])

        # Add a trace on the variables so they get stored when changed.
        for var in self._hyper_params.values():
            var.trace_add('write', self._save_settings)

    def _save_settings(self, *_):
        """Save the current settings to a preferences file"""

        for key, variable in self._hyper_params.items():
            self.persistencesavingmodel.set(key, variable.get())
        self.persistencesavingmodel.save()

    def _show_more(self, *_):
        about_message = "TO BE CONTINUED..."
        about_detail = (
            'There will be more coming.'
        )

        messagebox.showinfo(
            title='COMING SOON', message=about_message, detail=about_detail
        )