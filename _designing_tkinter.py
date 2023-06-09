"""
Then let me design the tkinter for pretty face.
"""
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
FigureCanvasTkAgg,
NavigationToolbar2Tk
)

# More definition of commodity indexes
indexWanted_CU0 = ['CU0', 'P0', 'Y0', 'AG0', 'BU0', 'ZN0', 'C0', 'AL0', 'RM0', 'M0', 'CF0']
indexWanted_RB0 = ['RB0', 'HC0', 'I0', 'V0', 'BU0', 'PP0', 'L0', 'FG0', 'MA0', 'SA0', 'SR0']
indexWanted_SCM = ['SCM', 'AU0', 'PG0', 'EB0', 'FU0', 'TA0', 'PP0', 'L0', 'V0', 'MA0', 'RU0']

# Include all the interested commodity indexes and making the target index as the first one.
indexList = list(np.unique(indexWanted_CU0 + indexWanted_RB0 + indexWanted_SCM))

# Define extra tkinter class
class BoundText(tk.Text):
    def __init__(self, *args, textvariable=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._variable = textvariable
        if self._variable:
            self.insert('1.0', self._variable.get())
            self._variable.trace_add('write', self._set_content)
            self.bind('<<Modified>>', self._set_var)

    def _set_content(self, *_):
        self.delete('1.0', tk.END)
        self.insert('1.0', self._variable.get())

    def _set_var(self, *_):
        if self.edit_modified():
            content = self.get('1.0', 'end-1chars')
            self._variable.set(content)
            self.edit_modified(False)

class LabelInput(tk.Frame):
    def __init__(self, parent, label, var, input_class=tk.Entry,
                 input_args=None, label_args=None, **kwargs):
        super().__init__(parent, **kwargs)
        input_args = input_args or {}
        label_args = label_args or {}
        self.variable = var
        self.variable.label_widget = self

        if input_class in (ttk.Checkbutton, ttk.Button):
            input_args['text'] = label
        else:
            self.label = ttk.Label(self, text=label, **label_args)
            self.label.grid(row=0, column=0, sticky=(tk.W+tk.E))

        if input_class in (
            ttk.Checkbutton, ttk.Button, ttk.Radiobutton
        ):
            input_args['variable'] = self.variable
        else:
            input_args['textvariable'] = self.variable

        # setup the input
        if input_class == ttk.Radiobutton:
            # for Radiobutton, create one input per value
            self.input = tk.Frame(self)
            for v in input_args.pop('values', []):
                button = ttk.Radiobutton(
                    self.input, value=v, text=v, **input_args
                )
                button.pack(side=tk.LEFT, ipadx=10, ipady=2, expand=True, fill='x')
        else:
            self.input = input_class(self, **input_args)

        self.input.grid(row=1, column=0, sticky=(tk.W+tk.E))
        self.columnconfigure(0, weight=1)

    def grid(self, sticky=(tk.E + tk.W), **kwargs):
        super().grid(sticky=sticky, **kwargs)

"""
Widget and frame creation.
"""

class processingWindow(ttk.Frame):
    def __init__(self, *args, indexWanted_CU0=None, indexWanted_RB0=None, indexWanted_SCM=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.indexWanted_CU0 = indexWanted_CU0
        self.indexWanted_RB0 = indexWanted_RB0
        self.indexWanted_SCM = indexWanted_SCM
        self._vars = {
            'count threshold': tk.IntVar(),
            'count type': tk.IntVar(),
            'output text': tk.StringVar()
        }

        self.columnconfigure(0, weight=1)

        self._vars['count threshold'].set(value=39)
        self._vars['count type'].set(value=0)

        left_frame = self._add_frame(label="OUTPUTS", cols=2)
        left_frame.grid(row=0, column=0, sticky=(tk.E + tk.W + tk.N))

        ttk.Label(left_frame, text='HEATING MAP').grid(row=0, column=0, sticky=(tk.E + tk.W))

        ttk.Label(left_frame, text=''.join([" " * 50, "DOWN MAP"]), foreground="green").grid(row=1, column=0, sticky=(tk.W))
        self.figureDN = Figure(figsize=(4, 4), dpi=100)
        self.canvas_tkagg_DN = FigureCanvasTkAgg(self.figureDN, left_frame)
        self.canvas_tkagg_DN.get_tk_widget().grid(row=2, column=0, sticky=(tk.W+tk.E+tk.N+tk.S), padx=50, pady=20)

        ttk.Label(left_frame, text=''.join(["UP MAP", " " * 50]), foreground="red").grid(row=1, column=1, sticky=(tk.E))
        self.figureUP = Figure(figsize=(4, 4), dpi=100)
        self.canvas_tkagg_UP = FigureCanvasTkAgg(self.figureUP, left_frame)
        self.canvas_tkagg_UP.get_tk_widget().grid(row=2, column=1, sticky=(tk.W+tk.E+tk.N+tk.S), padx=50, pady=20)

        self.outputText = LabelInput(
            left_frame, 'COUNTING NUMBERS',
            input_class=BoundText,
            var=self._vars['output text'],
            input_args={'width': 80, 'height': 10, 'font': ('TkDefault', 10)}
        )
        self.outputText.grid(row=3, column=0, columnspan=2, sticky=(tk.W+tk.E))

        right_frame = self._add_frame(label='SELECT THE GROUP', cols=1)
        right_frame.grid(row=0, column=1, sticky=(tk.E + tk.W + tk.N))
        btnCU0 = ttk.Button(
            right_frame,
            text='COPPER RELATED',
            command=lambda: self._run_CU0(indexWanted=self.indexWanted_CU0)
        )
        btnCU0.grid(row=0, column=0, sticky=(tk.W + tk.E), pady=5, padx=5)
        btnRB0 = ttk.Button(
            right_frame,
            text='REBAR RELATED',
            command=lambda: self._run_RB0(indexWanted=self.indexWanted_RB0)
        )
        btnRB0.grid(row=1, column=0, sticky=(tk.W + tk.E), pady=5, padx=5)
        btnSCM = ttk.Button(
            right_frame,
            text='CRUDE RELATED',
            command=lambda: self._run_SCM(indexWanted=self.indexWanted_SCM)
        )
        btnSCM.grid(row=2, column=0, sticky=(tk.W + tk.E), pady=5, padx=5)
        LabelInput(right_frame, "COUNT THRESHOLD", input_class=ttk.Spinbox,
                   var=self._vars['count threshold'],
                   input_args={'from': 33, 'to': 66, 'increment': 1}
                   ).grid(row=3, column=0, sticky=(tk.W + tk.E), pady=5, padx=5)
        LabelInput(right_frame, "NEGATIVE", input_class=ttk.Checkbutton,
                   var=self._vars['count type'],
                   input_args={'onvalue': 1, 'offvalue': 0}
                   ).grid(row=4, column=0, sticky=(tk.W + tk.E), pady=5, padx=5)

    def _add_frame(self, label, cols=3):
        frame = ttk.LabelFrame(self, text=label)
        frame.grid(sticky=(tk.W + tk.E))
        for i in range(cols):
            frame.columnconfigure(i, weight=1)
        return frame

    def get(self):
        data = dict()
        for key, variable in self._vars.items():
            data[key] = ''
        return data

    def _run_CU0(self, indexWanted):
        self.figureUP.clf()
        figUP = self.figureUP.add_subplot(1, 1, 1)

        self.figureDN.clf()
        figDN = self.figureDN.add_subplot(1, 1, 1)

        negative_count, positive_count=[], []

        for i in range(1000):
            var = np.random.uniform(-.1, .1)
            self.master._to_status(str(var))
            self._vars['output text'].set(str(var))
            if var >= 0:
                positive_count.append(var)
                figUP.scatter(var, var, s=round(abs(var)*50000, 0), c='red', alpha=0.5)
                figUP.set_xlabel(' '.join(['POSITIVE COUNT:', str(len(positive_count))]))
            else:
                negative_count.append(var)
                figDN.scatter(var, var, s=round(abs(var)*50000, 0), c='green', alpha=0.5)
                figDN.set_xlabel(' '.join(['NEGATIVE COUNT:', str(len(negative_count))]))
            self.canvas_tkagg_UP.draw()
            self.canvas_tkagg_DN.draw()
            self.master.update()

    def _run_RB0(self, indexWanted):
        print(indexWanted)

    def _run_SCM(self, indexWanted):
        print(indexWanted)

    def _generate_xy(self, output, classesTable):
        nor_eta = classesTable.shape[0]
        y = output - nor_eta / 2
        x = np.arange(0, len(output))
        radiant = np.linspace(1, 0, len(output))
        y_p_radiant = np.round([a * b for a, b in zip(y, radiant)], 0)
        y_radiant = np.cumsum(y_p_radiant)
        return x, y, y_radiant

    def _single_predict(self, ind):
        # Setting the master's status
        self.master._to_status(" ".join(['Prediction of single target to', ind]))

        # Visualize results output
        self.figure.clf()
        self.fig1 = self.figure.add_subplot(2, 3, (1, 2))
        x = np.linspace(1, 10, 25)
        line1 = self.fig1.plot(x, color='black')
        self.fig1.set_xlabel('FUTURE TIME')
        self.fig1.set_ylabel('CLASSES')
        self.fig1.set_title('PREDICTION HEAT MAP')
        self.fig1.legend(['Prediction', 'Heating'], loc='lower center')
        self.fig2 = self.figure.add_subplot(2, 3, 3)
        text2 = self.fig2.text(x=0, y=0.5,
                               ha='left', va='center', color='black',
                               bbox=dict(facecolor='red', alpha=0.5),
                               fontsize=18,
                               s='Hello world!\nIt is a nice world.')
        self.fig2.axis('off')
        self.canvas_tkagg.draw()

    def _all_predict(self, ind):
        # Setting the master's status
        self.master._to_status(" ".join(['Prediction of single target to', " ".join(ind)]))

        # Visualize results output
        self.figure.clf()
        self.fig = self.figure.add_subplot(2, 3, (1, 6))
        x = np.linspace(1, 10, 25)
        line1 = self.fig.plot(x, color='black')
        self.fig.set_xlabel('TIME')
        self.fig.set_ylabel('CLASSES')
        self.fig.set_title('PREDICTION HEAT MAP')
        self.fig.legend(['Prediction', 'Heating'], loc='lower center')
        self.canvas_tkagg.draw()

class Application(tk.Tk):
    def __init__(self, *args, indexWanted_CU0=None, indexWanted_RB0=None, indexWanted_SCM=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.indexWanted_CU0 = indexWanted_CU0
        self.indexWanted_RB0 = indexWanted_RB0
        self.indexWanted_SCM = indexWanted_SCM
        self.title('IV_many2one_dicingerpro')
        self.columnconfigure(0, weight=1)
        ttk.Label(
            self,
            text='DICINGER PRO (DAILY DATA)',
            font=('TkDefault', 16)
        ).grid(row=0, padx=10)

        self.processingWindow = processingWindow(self,
                                                 indexWanted_CU0=indexWanted_CU0,
                                                 indexWanted_RB0=indexWanted_RB0,
                                                 indexWanted_SCM=indexWanted_SCM)
        self.processingWindow.grid(row=1, padx=10, sticky=(tk.W+tk.E))

        self.status = tk.StringVar()
        ttk.Label(
            self, textvariable=self.status
        ).grid(row=99, padx=10, sticky=tk.W+tk.E)

    def _to_status(self, text):
        self.status.set(text)

App = Application(indexWanted_CU0=indexWanted_CU0,
                  indexWanted_RB0=indexWanted_RB0,
                  indexWanted_SCM=indexWanted_SCM)
App.mainloop()
