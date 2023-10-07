import tkinter as tk
from tkinter import ttk
import utils.widgets as w
import numpy as np
import pandas as pd
import utils.readit as readit
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import models.tripledicermodel as tdm

class TripleDicer(tk.Frame):
    """Set the hyperparameters page"""

    def __init__(self, parent, _hyper_parameters, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.columnconfigure(0, weight=1)
        self._vars = _hyper_parameters

        # Main Frame with Model Name
        main_frame = self._add_frame(label='TRIPLE DICER MODEL', cols=3)
        main_frame.grid()

        # Set the hyperparameters frame
        hp_frame = ttk.LabelFrame(main_frame, text='hyperparameters')
        hp_frame.grid(row=0, column=0, columnspan=3, sticky=tk.W+tk.E)
        for i in range(11):
            w.LabelInput(
                hp_frame, f"V{i}",
                input_class=w.ValidatedCombobox,
                var=self._vars[f'eleven_v{i}'],
                input_args={'values': readit.indexList, 'width': 13}
            ).grid(row=0, column=i)

        self.elevengroups = ['CU0 Related', 'RB0 Related', 'SCM Related', 'Finance Related']
        self.elevengroup = tk.StringVar()
        self.elevengroup.set(value="Finance Related")

        elevengroupselect = w.LabelInput(
            hp_frame, "Select The Preset Eleven Group",
            input_class=w.ValidatedCombobox,
            var=self.elevengroup,
            input_args={'values': self.elevengroups, 'width': 13}
        )
        elevengroupselect.grid(row=1, column=0, columnspan=3)

        # when we change the group, when focusout, we should update every value of the eleven group
        self.elevengroup.trace_add('write', self._set_eleven_group)

        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=1, column=0, columnspan=1, sticky=tk.W+tk.E)

        # Set the model parameter frame
        mp_frame = ttk.LabelFrame(left_frame, text='model parameters')
        mp_frame.grid(row=0, column=0, sticky=tk.W+tk.E)

        w.LabelInput(
            mp_frame, 'COUNT THRESHOLD',
            field_spec={'type': 'int', 'min': 0, 'max': 66, 'inc': 1},
            var=self._vars['triple dicer count threshold']
        ).grid(row=0, column=0)
        w.LabelInput(
            mp_frame, 'NEGATIVE COUNT',
            field_spec={'type': 'bool', },
            var=self._vars['triple dicer count type']
        ).grid(row=1, column=0)
        w.LabelInput(
            mp_frame, 'STOP CURRENT COUNT',
            field_spec={'type': 'bool', },
            var=self._vars['triple dicer stop flag']
        ).grid(row=2, column=0)
        w.LabelInput(
            mp_frame, 'X PREDICT DATE',
            field_spec={'type': 'str', },
            var=self._vars['triple dicer predict date']
        ).grid(row=3, column=0)

        # Set the command frame
        cmd_frame = ttk.LabelFrame(left_frame, text='model commands')
        cmd_frame.grid(row=2, column=0, sticky=tk.W+tk.E)

        runBtn = ttk.Button(
            cmd_frame,
            text='RUN',
            command=lambda: self._run()
        )
        runBtn.grid(row=0, column=0)

        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=1, column=1, columnspan=2, sticky=tk.W+tk.E)

        # Set the model output frame
        op_frame = ttk.LabelFrame(right_frame, text='results and outputs')
        op_frame.grid(row=0, column=0, sticky=tk.W+tk.E)

        # Visualizing Outputs
        ttk.Label(op_frame, text='visualizing outputs').grid(row=0, column=0, sticky=tk.W+tk.E)
        self.figure = Figure(figsize=(15, 6), dpi=100)
        self.canvas_tkagg = FigureCanvasTkAgg(self.figure, op_frame)
        self.canvas_tkagg.get_tk_widget().grid(row=1, column=0, sticky=tk.W+tk.E)

        # Textual Outputs
        self.resultsText = w.LabelInput(
            op_frame, 'textual results',
            field_spec={'type': 'BoundText'},
            var=self._vars['triple dicer textual results'],
            input_args={'width': 100, 'height': 7}
        )
        self.resultsText.grid(row=2, column=0, sticky=tk.W+tk.E)

    def _add_frame(self, label, cols=3):
        frame = ttk.LabelFrame(self, text=label)
        frame.grid(sticky=(tk.W + tk.E))
        for i in range(cols):
            frame.columnconfigure(i, weight=1)
        return frame

    def _set_eleven_group(self, *_):
        """Set the individual even group"""
        elevengroup = self.elevengroup.get()
        if elevengroup == 'CU0 Related':
            indexWanted = readit.indexWanted_CU0
        elif elevengroup == 'RB0 Related':
            indexWanted = readit.indexWanted_RB0
        elif elevengroup == 'SCM Related':
            indexWanted = readit.indexWanted_SCM
        else:
            indexWanted = readit.indexWanted_FINANCE

        for i in range(11):
            self._vars[f'eleven_v{i}'].set(indexWanted[i])
        self._vars['indexWanted_default'].set(indexWanted)

        self.master._to_status('Changes are auto saved.')

    def _run(self):
        """Runing the Triple Dicinger Model
        Try to have as fewer codes here as possible."""
        print("Run the model.")