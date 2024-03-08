"""
Mission IV: DicingerPro
In the previous verssion, features includes only 12 raw data, 4 from email sentiment, 4 from weibo sentiment and another
4 yields.
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
import math
import datetime

# Part I:
# Prepare the features and labels for models.===========================================================================
dataDirName = "/run/user/1000/gvfs/smb-share:server=crjlambda-pc,share=dataforlater"
emailReadfilename = Path(dataDirName, "II_seq2seq_moon2sun_cook_email_feature_forlater.json")
weiboReadfilename = Path(dataDirName, "II_seq2seq_moon2sun_cook_weibo_feature_forlater.json")
emailFeatures_df = pd.read_json(emailReadfilename)
weiboFeatures_df = pd.read_json(weiboReadfilename)

"""
Get labels and process both the features and labels for deep learning.
"""
# Also read the indexes
dataDirName = "/run/user/1000/gvfs/smb-share:server=crjlambda-pc,share=dailytds"
TD_indexes = pd.read_csv(Path(dataDirName, 'ref_TD.csv'))
TD_yields_indexes = pd.read_csv(Path(dataDirName, 'ref_yields.csv'))
TD_Currency_indexes = pd.read_csv(Path(dataDirName, 'ref_Currency.csv'))

# And generate wanted dataset
indexesAll = TD_indexes.join(TD_Currency_indexes, rsuffix='_Currency')
# indexesAll = indexesAll.join(TD_yields_indexes, rsuffix='_yields')
indexesAll_ind = indexesAll.iloc[0,]

"""
To get labels for deep learning.
"""
# class to get labels and yields feature.----------

indexWanted_CU0 = ['CU0', 'P0', 'Y0', 'AG0', 'BU0', 'ZN0', 'C0', 'AL0', 'RM0', 'M0', 'CF0']
indexWanted_RB0 = ['RB0', 'HC0', 'I0', 'V0', 'BU0', 'JM0', 'UR0', 'FG0', 'MA0', 'SA0', 'SR0']
indexWanted_SCM = ["SCM", 'AU0', 'PG0', 'EB0', 'FU0', 'TA0', 'PP0', 'L0', 'V0', 'LUM', 'RU0']
indexWanted_FINANCE = ['IH00C1', 'IF00C1', 'IC00C1', 'IM00C1', 'TS00C1', 'T00C1', 'TF00C1', 'TL00C1', 'CU0', 'RB0', 'SCM']
indexList = list(np.unique(indexWanted_CU0 + indexWanted_RB0 + indexWanted_SCM + indexWanted_FINANCE))

"""
=PART II, Deep Learning.================================================================================================
=There are three different types of deep learning network models to be implemented. Although they are quite pre-mature =
=I am counting on it to generate odds for dicing.                                                                      =
=1.<Simple Linear Model.>                                                                                              =
=2.<Simple Complete Learning Network Model.>                                                                           =
=3.<Simple Convolutional Network Model.>
========================================================================================================================
"""
# in version 1, I was adding bond yields as more features.
import IV_many2one_getYields as gt
yieldsWanted = ['CN_10yry', 'US_10yry', 'CN_5yry', 'CN_2yry']
gtReturn = gt.readingYields(yieldsWanted)
featuresYieldsDL_df = gtReturn.returnFeatures

import IV_many2one_getLabels as gl
from tqdm import tqdm
import IV_many2one_simpleDeeplearning as sdl
import IV_many2one_simpleCompleteln as scln
import IV_simpleConvolutionnetwork as scnn
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

dataDirName = "/run/user/1000/gvfs/smb-share:server=crjlambda-pc,share=allproba"

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
    def __init__(self, *args, indexWanted_CU0=None, indexWanted_RB0=None,
                 indexWanted_SCM=None, indexWanted_FINANCE=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.indexWanted_CU0 = indexWanted_CU0
        self.indexWanted_RB0 = indexWanted_RB0
        self.indexWanted_SCM = indexWanted_SCM
        self.indexWanted_FINANCE = indexWanted_FINANCE
        self._vars = {
            'count threshold': tk.IntVar(),
            'count type': tk.IntVar(),
            'output text': tk.StringVar(),
            'stop flag': tk.BooleanVar(),
            'X predict date': tk.StringVar()
        }

        self.columnconfigure(0, weight=1)

        self._vars['count threshold'].set(value=44)
        self._vars['count type'].set(value=0)
        self._vars['stop flag'].set(value=False)

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
            left_frame, 'TEXT OUTPUT',
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
        btnFINANCE = ttk.Button(
            right_frame,
            text='FINANCE RELATED',
            command=lambda: self._run_FINANCE(indexWanted=self.indexWanted_FINANCE)
        )
        btnFINANCE.grid(row=3, column=0, sticky=(tk.W + tk.E), pady=5, padx=5)
        LabelInput(right_frame, "COUNT THRESHOLD", input_class=ttk.Spinbox,
                   var=self._vars['count threshold'],
                   input_args={'from': 33, 'to': 66, 'increment': 1}
                   ).grid(row=4, column=0, sticky=(tk.W + tk.E), pady=5, padx=5)
        LabelInput(right_frame, "NEGATIVE", input_class=ttk.Checkbutton,
                   var=self._vars['count type'],
                   input_args={'onvalue': 1, 'offvalue': 0}
                   ).grid(row=5, column=0, sticky=(tk.W + tk.E), pady=5, padx=5)
        LabelInput(right_frame, "STOP CURRENT COUNT", input_class=ttk.Checkbutton,
                   var=self._vars['stop flag'],
                   input_args={'onvalue': True, 'offvalue': False}
                   ).grid(row=6, column=0, sticky=(tk.W + tk.E), pady=5, padx=5)

        right_b_frame = ttk.LabelFrame(master=right_frame, text='ADDITION INFO.')
        right_b_frame.grid(row=7, column=0, sticky=(tk.E + tk.W + tk.N))
        LabelInput(right_b_frame, "X Predict Date", input_class=ttk.Label,
                   var=self._vars['X predict date']
                   ).grid(row=0, column=0, sticky=(tk.W+tk.E))

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

        countingResults = []

        countThreshold = self._vars['count threshold'].get()
        countType = self._vars['count type'].get()
        maxCount = 0
        while True:
            positiveCount = 0
            negativeCount = 0
            # Implement simple linear model.
            sdlResults = []
            for i in tqdm(range(len(indexWanted)), ncols=100, desc="SIMPLE LINEAR NETWORK", colour="blue"):
                ind = [indexWanted[i]]
                glReturn = gl.getLabels(indexWanted=ind)
                labelsDL_df = glReturn.returnLabels
                sdlReturn = sdl.simpleDeeplearning(ind, emailFeatures_df, weiboFeatures_df, featuresYieldsDL_df,
                                                   labelsDL_df)
                for var in sdlReturn.results.iloc[0].values[2:]:
                    rndV_negative = np.random.uniform(-.1, 0)
                    rndV_positive = np.random.uniform(0, .1)
                    self.master._to_status(' '.join(["Simple Deep Learning prediction of", ind[0], ":", str(sdlReturn.results.iloc[0].values[2:])]))
                    self._vars['X predict date'].set(value=sdlReturn.X_predict_date)
                    if var >= 0:
                        positive_count.append(var)
                        figUP.scatter(rndV_positive, rndV_positive, s=round(abs(rndV_positive) * 5e4, 0), c='red', alpha=0.1)
                        figUP.set_xlabel(' '.join(['POSITIVE COUNT:', str(len(positive_count))]))
                        positiveCount = positiveCount + 1
                    else:
                        negative_count.append(var)
                        figDN.scatter(rndV_negative, rndV_negative, s=round(abs(rndV_negative) * 5e4, 0), c='green', alpha=0.1)
                        figDN.set_xlabel(' '.join(['NEGATIVE COUNT:', str(len(negative_count))]))
                        negativeCount = negativeCount + 1
                    self.canvas_tkagg_UP.draw()
                    self.canvas_tkagg_DN.draw()
                    self.master.update()
                sdlResults.append(sdlReturn.results)
                self._vars['output text'].set(str(sdlReturn.results))
                self.master.update()

            # Implement simple complete learning network.
            sclnResults = []
            for i in tqdm(range(len(indexWanted)), ncols=100, desc="SIMPLE COMPLETE NETWORK", colour="blue"):
                ind = [indexWanted[i]]
                glReturn = gl.getLabels(indexWanted=ind)
                labelsDL_df = glReturn.returnLabels
                sclnReturn = scln.simpleCompleteln(ind, emailFeatures_df, weiboFeatures_df, featuresYieldsDL_df, labelsDL_df)
                for var in sclnReturn.results.iloc[0].values[2:]:
                    rndV_negative = np.random.uniform(-.1, 0)
                    rndV_positive = np.random.uniform(0, .1)
                    self.master._to_status(' '.join(["Simple Complete LN prediction of", ind[0], ":", str(sclnReturn.results.iloc[0].values[2:])]))
                    self._vars['X predict date'].set(value=sclnReturn.X_predict_date)
                    if var >= 0:
                        positive_count.append(var)
                        figUP.scatter(rndV_positive, rndV_positive, s=round(abs(rndV_positive) * 5e4, 0), c='red', alpha=0.1)
                        figUP.set_xlabel(' '.join(['POSITIVE COUNT:', str(len(positive_count))]))
                        positiveCount = positiveCount + 1
                    else:
                        negative_count.append(var)
                        figDN.scatter(rndV_negative, rndV_negative, s=round(abs(rndV_negative) * 5e4, 0), c='green', alpha=0.1)
                        figDN.set_xlabel(' '.join(['NEGATIVE COUNT:', str(len(negative_count))]))
                        negativeCount = negativeCount + 1
                    self.canvas_tkagg_UP.draw()
                    self.canvas_tkagg_DN.draw()
                    self.master.update()
                sclnResults.append(sclnReturn.results)
                self._vars['output text'].set(str(sclnReturn.results))
                self.master.update()

            # The 3rd network, simple convolution network.
            """
            I need a 4x3 features inorder to use the convolution network. And features array should be transformed into a 4D 
            data, [batch_size, 1, 3, 4], the '1' is a channel.
            """
            scnnResults = []
            for i in tqdm(range(len(indexWanted)), ncols=100, desc="SIMPLE CONVOLUTION NETWORK", colour="blue"):
                ind = [indexWanted[i]]
                glReturn = gl.getLabels(indexWanted=ind)
                labelsDL_df = glReturn.returnLabels
                scnnReturn = scnn.simpleConvolutionnetwork(ind, emailFeatures_df, weiboFeatures_df, featuresYieldsDL_df, labelsDL_df)
                for var in scnnReturn.results.iloc[0].values[2:]:
                    rndV_negative = np.random.uniform(-.1, 0)
                    rndV_positive = np.random.uniform(0, .1)
                    self.master._to_status(' '.join(["Simple Convolution Network prediction of", ind[0], ":", str(scnnReturn.results.iloc[0].values[2:])]))
                    self._vars['X predict date'].set(value=scnnReturn.X_predict_date)
                    if var >= 0:
                        positive_count.append(var)
                        figUP.scatter(rndV_positive, rndV_positive, s=round(abs(rndV_positive) * 5e4, 0), c='red', alpha=0.1)
                        figUP.set_xlabel(' '.join(['POSITIVE COUNT:', str(len(positive_count))]))
                        positiveCount = positiveCount + 1
                    else:
                        negative_count.append(var)
                        figDN.scatter(rndV_negative, rndV_negative, s=round(abs(rndV_negative) * 5e4, 0), c='green', alpha=0.1)
                        figDN.set_xlabel(' '.join(['NEGATIVE COUNT:', str(len(negative_count))]))
                        negativeCount = negativeCount + 1
                scnnResults.append(scnnReturn.results)
                self._vars['output text'].set(str(scnnReturn.results))
                self.master.update()

            # Showing the counting results in the text box.
            countResults = "\nCounting Results generated at time: " + str(datetime.datetime.now()) + '\n' + \
                           "Overall Group Repeated at : " + str(len(countingResults)+1) + "\n" + \
                           'COUNT TYPE: ' + ('NEGATIVE' if countType == 1 else 'POSITIVE') + '\n' + \
                           "Positive Count: " + str(positiveCount) + ";  " + \
                           "Negative Count: " + str(negativeCount) + "\n" + \
                           "Total Count: " + str(positiveCount * (1 - countType) + negativeCount * countType) + "\n"
            countingResults.append(countResults)
            self._vars['output text'].set(str(countResults))
            self.master.update()

            if positiveCount * (1 - countType) + negativeCount * countType > maxCount:
                # In order to use tkinter for pretty output, I need to package all the results into one string.
                outputString = '\nCOUNT TYPE: ' + ('NEGATIVE' if countType == 1 else 'POSITIVE') + \
                               '\nTOTAL COUNT: ' + str(positiveCount * (1 - countType) + negativeCount * countType)
                for i in range(len(indexWanted)):
                    singleIndexresult = '\n'.join(["\nDeep Learning results of " + indexWanted[i],
                                                   '*' * 100,
                                                   str(sdlResults[i]),
                                                   '.' * 100,
                                                   str(sclnResults[i]),
                                                   str(scnnResults[i]),
                                                   "=" * 100])
                    outputString = "\n".join([outputString, singleIndexresult])
                    self._vars['output text'].set(outputString)
                    self.master.update()
                    # Saving output string to text
                    saving_to_file = open(Path(dataDirName, '-'.join([str(datetime.datetime.now().date()), "IV_many2one_outputStringCU0.txt"])), 'w')
                    saving_to_file.write(countResults + outputString)
                    saving_to_file.close()
                    maxCount = positiveCount * (1 - countType) + negativeCount * countType

            if positiveCount * (1 - countType) + negativeCount * countType >= countThreshold:
                break
            elif self._vars['stop flag'].get():
                self._vars['stop flag'].set(value=False)
                self.master.update()
                break

    def _run_RB0(self, indexWanted):
        self.figureUP.clf()
        figUP = self.figureUP.add_subplot(1, 1, 1)

        self.figureDN.clf()
        figDN = self.figureDN.add_subplot(1, 1, 1)

        negative_count, positive_count=[], []

        countingResults = []

        countThreshold = self._vars['count threshold'].get()
        countType = self._vars['count type'].get()
        maxCount = 0
        while True:
            positiveCount = 0
            negativeCount = 0
            # Implement simple linear model.
            sdlResults = []
            for i in tqdm(range(len(indexWanted)), ncols=100, desc="SIMPLE LINEAR NETWORK", colour="blue"):
                ind = [indexWanted[i]]
                glReturn = gl.getLabels(indexWanted=ind)
                labelsDL_df = glReturn.returnLabels
                sdlReturn = sdl.simpleDeeplearning(ind, emailFeatures_df, weiboFeatures_df, featuresYieldsDL_df,
                                                   labelsDL_df)
                for var in sdlReturn.results.iloc[0].values[2:]:
                    rndV_negative = np.random.uniform(-.1, 0)
                    rndV_positive = np.random.uniform(0, .1)
                    self.master._to_status(' '.join(["Simple Deep Learning prediction of", ind[0], ":", str(sdlReturn.results.iloc[0].values[2:])]))
                    self._vars['X predict date'].set(value=sdlReturn.X_predict_date)
                    if var >= 0:
                        positive_count.append(var)
                        figUP.scatter(rndV_positive, rndV_positive, s=round(abs(rndV_positive) * 5e4, 0), c='red', alpha=0.1)
                        figUP.set_xlabel(' '.join(['POSITIVE COUNT:', str(len(positive_count))]))
                        positiveCount = positiveCount + 1
                    else:
                        negative_count.append(var)
                        figDN.scatter(rndV_negative, rndV_negative, s=round(abs(rndV_negative) * 5e4, 0), c='green', alpha=0.1)
                        figDN.set_xlabel(' '.join(['NEGATIVE COUNT:', str(len(negative_count))]))
                        negativeCount = negativeCount + 1
                    self.canvas_tkagg_UP.draw()
                    self.canvas_tkagg_DN.draw()
                    self.master.update()
                sdlResults.append(sdlReturn.results)
                self._vars['output text'].set(str(sdlReturn.results))
                self.master.update()

            # Implement simple complete learning network.
            sclnResults = []
            for i in tqdm(range(len(indexWanted)), ncols=100, desc="SIMPLE COMPLETE NETWORK", colour="blue"):
                ind = [indexWanted[i]]
                glReturn = gl.getLabels(indexWanted=ind)
                labelsDL_df = glReturn.returnLabels
                sclnReturn = scln.simpleCompleteln(ind, emailFeatures_df, weiboFeatures_df, featuresYieldsDL_df, labelsDL_df)
                for var in sclnReturn.results.iloc[0].values[2:]:
                    rndV_negative = np.random.uniform(-.1, 0)
                    rndV_positive = np.random.uniform(0, .1)
                    self.master._to_status(' '.join(["Simple Complete LN prediction of", ind[0], ":", str(sclnReturn.results.iloc[0].values[2:])]))
                    self._vars['X predict date'].set(value=sclnReturn.X_predict_date)
                    if var >= 0:
                        positive_count.append(var)
                        figUP.scatter(rndV_positive, rndV_positive, s=round(abs(rndV_positive) * 5e4, 0), c='red', alpha=0.1)
                        figUP.set_xlabel(' '.join(['POSITIVE COUNT:', str(len(positive_count))]))
                        positiveCount = positiveCount + 1
                    else:
                        negative_count.append(var)
                        figDN.scatter(rndV_negative, rndV_negative, s=round(abs(rndV_negative) * 5e4, 0), c='green', alpha=0.1)
                        figDN.set_xlabel(' '.join(['NEGATIVE COUNT:', str(len(negative_count))]))
                        negativeCount = negativeCount + 1
                    self.canvas_tkagg_UP.draw()
                    self.canvas_tkagg_DN.draw()
                    self.master.update()
                sclnResults.append(sclnReturn.results)
                self._vars['output text'].set(str(sclnReturn.results))
                self.master.update()

            # The 3rd network, simple convolution network.
            """
            I need a 4x3 features inorder to use the convolution network. And features array should be transformed into a 4D 
            data, [batch_size, 1, 3, 4], the '1' is a channel.
            """
            scnnResults = []
            for i in tqdm(range(len(indexWanted)), ncols=100, desc="SIMPLE CONVOLUTION NETWORK", colour="blue"):
                ind = [indexWanted[i]]
                glReturn = gl.getLabels(indexWanted=ind)
                labelsDL_df = glReturn.returnLabels
                scnnReturn = scnn.simpleConvolutionnetwork(ind, emailFeatures_df, weiboFeatures_df, featuresYieldsDL_df, labelsDL_df)
                for var in scnnReturn.results.iloc[0].values[2:]:
                    rndV_negative = np.random.uniform(-.1, 0)
                    rndV_positive = np.random.uniform(0, .1)
                    self.master._to_status(' '.join(["Simple Convolution Network prediction of", ind[0], ":", str(scnnReturn.results.iloc[0].values[2:])]))
                    self._vars['X predict date'].set(value=scnnReturn.X_predict_date)
                    if var >= 0:
                        positive_count.append(var)
                        figUP.scatter(rndV_positive, rndV_positive, s=round(abs(rndV_positive) * 5e4, 0), c='red', alpha=0.1)
                        figUP.set_xlabel(' '.join(['POSITIVE COUNT:', str(len(positive_count))]))
                        positiveCount = positiveCount + 1
                    else:
                        negative_count.append(var)
                        figDN.scatter(rndV_negative, rndV_negative, s=round(abs(rndV_negative) * 5e4, 0), c='green', alpha=0.1)
                        figDN.set_xlabel(' '.join(['NEGATIVE COUNT:', str(len(negative_count))]))
                        negativeCount = negativeCount + 1
                scnnResults.append(scnnReturn.results)
                self._vars['output text'].set(str(scnnReturn.results))
                self.master.update()

            # Showing the counting results in the text box.
            countResults = "\nCounting Results generated at time: " + str(datetime.datetime.now()) + '\n' + \
                           "Overall Group Repeated at : " + str(len(countingResults)+1) + "\n" + \
                           'COUNT TYPE: ' + ('NEGATIVE' if countType == 1 else 'POSITIVE') + '\n' + \
                           "Positive Count: " + str(positiveCount) + ";  " + \
                           "Negative Count: " + str(negativeCount) + "\n" + \
                           "Total Count: " + str(positiveCount * (1 - countType) + negativeCount * countType) + "\n"
            countingResults.append(countResults)
            self._vars['output text'].set(str(countResults))
            self.master.update()

            if positiveCount * (1 - countType) + negativeCount * countType > maxCount:
                # In order to use tkinter for pretty output, I need to package all the results into one string.
                outputString = '\nCOUNT TYPE: ' + ('NEGATIVE' if countType == 1 else 'POSITIVE') + \
                               '\nTOTAL COUNT: ' + str(positiveCount * (1 - countType) + negativeCount * countType)
                for i in range(len(indexWanted)):
                    singleIndexresult = '\n'.join(["\nDeep Learning results of " + indexWanted[i],
                                                   '*' * 100,
                                                   str(sdlResults[i]),
                                                   '.' * 100,
                                                   str(sclnResults[i]),
                                                   str(scnnResults[i]),
                                                   "=" * 100])
                    outputString = "\n".join([outputString, singleIndexresult])
                    self._vars['output text'].set(outputString)
                    self.master.update()
                    # Saving output string to text
                    saving_to_file = open(Path(dataDirName, '-'.join([str(datetime.datetime.now().date()), "IV_many2one_outputStringRB0.txt"])), 'w')
                    saving_to_file.write(countResults + outputString)
                    saving_to_file.close()
                    maxCount = positiveCount * (1 - countType) + negativeCount * countType

            if positiveCount * (1 - countType) + negativeCount * countType >= countThreshold:
                break
            elif self._vars['stop flag'].get():
                self._vars['stop flag'].set(value=False)
                self.master.update()
                break

    def _run_SCM(self, indexWanted):
        self.figureUP.clf()
        figUP = self.figureUP.add_subplot(1, 1, 1)

        self.figureDN.clf()
        figDN = self.figureDN.add_subplot(1, 1, 1)

        negative_count, positive_count=[], []

        countingResults = []

        countThreshold = self._vars['count threshold'].get()
        countType = self._vars['count type'].get()
        maxCount = 0
        while True:
            positiveCount = 0
            negativeCount = 0
            # Implement simple linear model.
            sdlResults = []
            for i in tqdm(range(len(indexWanted)), ncols=100, desc="SIMPLE LINEAR NETWORK", colour="blue"):
                ind = [indexWanted[i]]
                glReturn = gl.getLabels(indexWanted=ind)
                labelsDL_df = glReturn.returnLabels
                sdlReturn = sdl.simpleDeeplearning(ind, emailFeatures_df, weiboFeatures_df, featuresYieldsDL_df,
                                                   labelsDL_df)
                for var in sdlReturn.results.iloc[0].values[2:]:
                    rndV_negative = np.random.uniform(-.1, 0)
                    rndV_positive = np.random.uniform(0, .1)
                    self.master._to_status(' '.join(["Simple Deep Learning prediction of", ind[0], ":", str(sdlReturn.results.iloc[0].values[2:])]))
                    self._vars['X predict date'].set(value=sdlReturn.X_predict_date)
                    if var >= 0:
                        positive_count.append(var)
                        figUP.scatter(rndV_positive, rndV_positive, s=round(abs(rndV_positive) * 5e4, 0), c='red', alpha=0.1)
                        figUP.set_xlabel(' '.join(['POSITIVE COUNT:', str(len(positive_count))]))
                        positiveCount = positiveCount + 1
                    else:
                        negative_count.append(var)
                        figDN.scatter(rndV_negative, rndV_negative, s=round(abs(rndV_negative) * 5e4, 0), c='green', alpha=0.1)
                        figDN.set_xlabel(' '.join(['NEGATIVE COUNT:', str(len(negative_count))]))
                        negativeCount = negativeCount + 1
                    self.canvas_tkagg_UP.draw()
                    self.canvas_tkagg_DN.draw()
                    self.master.update()
                sdlResults.append(sdlReturn.results)
                self._vars['output text'].set(str(sdlReturn.results))
                self.master.update()

            # Implement simple complete learning network.
            sclnResults = []
            for i in tqdm(range(len(indexWanted)), ncols=100, desc="SIMPLE COMPLETE NETWORK", colour="blue"):
                ind = [indexWanted[i]]
                glReturn = gl.getLabels(indexWanted=ind)
                labelsDL_df = glReturn.returnLabels
                sclnReturn = scln.simpleCompleteln(ind, emailFeatures_df, weiboFeatures_df, featuresYieldsDL_df, labelsDL_df)
                for var in sclnReturn.results.iloc[0].values[2:]:
                    rndV_negative = np.random.uniform(-.1, 0)
                    rndV_positive = np.random.uniform(0, .1)
                    self.master._to_status(' '.join(["Simple Complete LN prediction of", ind[0], ":", str(sclnReturn.results.iloc[0].values[2:])]))
                    self._vars['X predict date'].set(value=sclnReturn.X_predict_date)
                    if var >= 0:
                        positive_count.append(var)
                        figUP.scatter(rndV_positive, rndV_positive, s=round(abs(rndV_positive) * 5e4, 0), c='red', alpha=0.1)
                        figUP.set_xlabel(' '.join(['POSITIVE COUNT:', str(len(positive_count))]))
                        positiveCount = positiveCount + 1
                    else:
                        negative_count.append(var)
                        figDN.scatter(rndV_negative, rndV_negative, s=round(abs(rndV_negative) * 5e4, 0), c='green', alpha=0.1)
                        figDN.set_xlabel(' '.join(['NEGATIVE COUNT:', str(len(negative_count))]))
                        negativeCount = negativeCount + 1
                    self.canvas_tkagg_UP.draw()
                    self.canvas_tkagg_DN.draw()
                    self.master.update()
                sclnResults.append(sclnReturn.results)
                self._vars['output text'].set(str(sclnReturn.results))
                self.master.update()

            # The 3rd network, simple convolution network.
            """
            I need a 4x3 features inorder to use the convolution network. And features array should be transformed into a 4D 
            data, [batch_size, 1, 3, 4], the '1' is a channel.
            """
            scnnResults = []
            for i in tqdm(range(len(indexWanted)), ncols=100, desc="SIMPLE CONVOLUTION NETWORK", colour="blue"):
                ind = [indexWanted[i]]
                glReturn = gl.getLabels(indexWanted=ind)
                labelsDL_df = glReturn.returnLabels
                scnnReturn = scnn.simpleConvolutionnetwork(ind, emailFeatures_df, weiboFeatures_df, featuresYieldsDL_df, labelsDL_df)
                for var in scnnReturn.results.iloc[0].values[2:]:
                    rndV_negative = np.random.uniform(-.1, 0)
                    rndV_positive = np.random.uniform(0, .1)
                    self.master._to_status(' '.join(["Simple Convolution Network prediction of", ind[0], ":", str(scnnReturn.results.iloc[0].values[2:])]))
                    self._vars['X predict date'].set(value=scnnReturn.X_predict_date)
                    if var >= 0:
                        positive_count.append(var)
                        figUP.scatter(rndV_positive, rndV_positive, s=round(abs(rndV_positive) * 5e4, 0), c='red', alpha=0.1)
                        figUP.set_xlabel(' '.join(['POSITIVE COUNT:', str(len(positive_count))]))
                        positiveCount = positiveCount + 1
                    else:
                        negative_count.append(var)
                        figDN.scatter(rndV_negative, rndV_negative, s=round(abs(rndV_negative) * 5e4, 0), c='green', alpha=0.1)
                        figDN.set_xlabel(' '.join(['NEGATIVE COUNT:', str(len(negative_count))]))
                        negativeCount = negativeCount + 1
                scnnResults.append(scnnReturn.results)
                self._vars['output text'].set(str(scnnReturn.results))
                self.master.update()

            # Showing the counting results in the text box.
            countResults = "\nCounting Results generated at time: " + str(datetime.datetime.now()) + '\n' + \
                           "Overall Group Repeated at : " + str(len(countingResults)+1) + "\n" + \
                           'COUNT TYPE: ' + ('NEGATIVE' if countType == 1 else 'POSITIVE') + '\n' + \
                           "Positive Count: " + str(positiveCount) + ";  " + \
                           "Negative Count: " + str(negativeCount) + "\n" + \
                           "Total Count: " + str(positiveCount * (1 - countType) + negativeCount * countType) + "\n"
            countingResults.append(countResults)
            self._vars['output text'].set(str(countResults))
            self.master.update()

            if positiveCount * (1 - countType) + negativeCount * countType > maxCount:
                # In order to use tkinter for pretty output, I need to package all the results into one string.
                outputString = '\nCOUNT TYPE: ' + ('NEGATIVE' if countType == 1 else 'POSITIVE') + \
                               '\nTOTAL COUNT: ' + str(positiveCount * (1 - countType) + negativeCount * countType)
                for i in range(len(indexWanted)):
                    singleIndexresult = '\n'.join(["\nDeep Learning results of " + indexWanted[i],
                                                   '*' * 100,
                                                   str(sdlResults[i]),
                                                   '.' * 100,
                                                   str(sclnResults[i]),
                                                   str(scnnResults[i]),
                                                   "=" * 100])
                    outputString = "\n".join([outputString, singleIndexresult])
                    self._vars['output text'].set(outputString)
                    self.master.update()
                    # Saving output string to text
                    saving_to_file = open(Path(dataDirName, '-'.join([str(datetime.datetime.now().date()), "IV_many2one_outputStringSCM.txt"])), 'w')
                    saving_to_file.write(countResults + outputString)
                    saving_to_file.close()
                    maxCount = positiveCount * (1 - countType) + negativeCount * countType

            if positiveCount * (1 - countType) + negativeCount * countType >= countThreshold:
                break
            elif self._vars['stop flag'].get():
                self._vars['stop flag'].set(value=False)
                self.master.update()
                break

    def _run_FINANCE(self, indexWanted):
        self.figureUP.clf()
        figUP = self.figureUP.add_subplot(1, 1, 1)

        self.figureDN.clf()
        figDN = self.figureDN.add_subplot(1, 1, 1)

        negative_count, positive_count=[], []

        countingResults = []

        countThreshold = self._vars['count threshold'].get()
        countType = self._vars['count type'].get()
        maxCount = 0
        while True:
            positiveCount = 0
            negativeCount = 0
            # Implement simple linear model.
            sdlResults = []
            for i in tqdm(range(len(indexWanted)), ncols=100, desc="SIMPLE LINEAR NETWORK", colour="blue"):
                ind = [indexWanted[i]]
                glReturn = gl.getLabels(indexWanted=ind)
                labelsDL_df = glReturn.returnLabels
                sdlReturn = sdl.simpleDeeplearning(ind, emailFeatures_df, weiboFeatures_df, featuresYieldsDL_df,
                                                   labelsDL_df)
                for var in sdlReturn.results.iloc[0].values[2:]:
                    rndV_negative = np.random.uniform(-.1, 0)
                    rndV_positive = np.random.uniform(0, .1)
                    self.master._to_status(' '.join(["Simple Deep Learning prediction of", ind[0], ":", str(sdlReturn.results.iloc[0].values[2:])]))
                    self._vars['X predict date'].set(value=sdlReturn.X_predict_date)
                    if var >= 0:
                        positive_count.append(var)
                        figUP.scatter(rndV_positive, rndV_positive, s=round(abs(rndV_positive) * 5e4, 0), c='red', alpha=0.1)
                        figUP.set_xlabel(' '.join(['POSITIVE COUNT:', str(len(positive_count))]))
                        positiveCount = positiveCount + 1
                    else:
                        negative_count.append(var)
                        figDN.scatter(rndV_negative, rndV_negative, s=round(abs(rndV_negative) * 5e4, 0), c='green', alpha=0.1)
                        figDN.set_xlabel(' '.join(['NEGATIVE COUNT:', str(len(negative_count))]))
                        negativeCount = negativeCount + 1
                    self.canvas_tkagg_UP.draw()
                    self.canvas_tkagg_DN.draw()
                    self.master.update()
                sdlResults.append(sdlReturn.results)
                self._vars['output text'].set(str(sdlReturn.results))
                self.master.update()

            # Implement simple complete learning network.
            sclnResults = []
            for i in tqdm(range(len(indexWanted)), ncols=100, desc="SIMPLE COMPLETE NETWORK", colour="blue"):
                ind = [indexWanted[i]]
                glReturn = gl.getLabels(indexWanted=ind)
                labelsDL_df = glReturn.returnLabels
                sclnReturn = scln.simpleCompleteln(ind, emailFeatures_df, weiboFeatures_df, featuresYieldsDL_df, labelsDL_df)
                for var in sclnReturn.results.iloc[0].values[2:]:
                    rndV_negative = np.random.uniform(-.1, 0)
                    rndV_positive = np.random.uniform(0, .1)
                    self.master._to_status(' '.join(["Simple Complete LN prediction of", ind[0], ":", str(sclnReturn.results.iloc[0].values[2:])]))
                    self._vars['X predict date'].set(value=sclnReturn.X_predict_date)
                    if var >= 0:
                        positive_count.append(var)
                        figUP.scatter(rndV_positive, rndV_positive, s=round(abs(rndV_positive) * 5e4, 0), c='red', alpha=0.1)
                        figUP.set_xlabel(' '.join(['POSITIVE COUNT:', str(len(positive_count))]))
                        positiveCount = positiveCount + 1
                    else:
                        negative_count.append(var)
                        figDN.scatter(rndV_negative, rndV_negative, s=round(abs(rndV_negative) * 5e4, 0), c='green', alpha=0.1)
                        figDN.set_xlabel(' '.join(['NEGATIVE COUNT:', str(len(negative_count))]))
                        negativeCount = negativeCount + 1
                    self.canvas_tkagg_UP.draw()
                    self.canvas_tkagg_DN.draw()
                    self.master.update()
                sclnResults.append(sclnReturn.results)
                self._vars['output text'].set(str(sclnReturn.results))
                self.master.update()

            # The 3rd network, simple convolution network.
            """
            I need a 4x3 features inorder to use the convolution network. And features array should be transformed into a 4D 
            data, [batch_size, 1, 3, 4], the '1' is a channel.
            """
            scnnResults = []
            for i in tqdm(range(len(indexWanted)), ncols=100, desc="SIMPLE CONVOLUTION NETWORK", colour="blue"):
                ind = [indexWanted[i]]
                glReturn = gl.getLabels(indexWanted=ind)
                labelsDL_df = glReturn.returnLabels
                scnnReturn = scnn.simpleConvolutionnetwork(ind, emailFeatures_df, weiboFeatures_df, featuresYieldsDL_df, labelsDL_df)
                for var in scnnReturn.results.iloc[0].values[2:]:
                    rndV_negative = np.random.uniform(-.1, 0)
                    rndV_positive = np.random.uniform(0, .1)
                    self.master._to_status(' '.join(["Simple Convolution Network prediction of", ind[0], ":", str(scnnReturn.results.iloc[0].values[2:])]))
                    self._vars['X predict date'].set(value=scnnReturn.X_predict_date)
                    if var >= 0:
                        positive_count.append(var)
                        figUP.scatter(rndV_positive, rndV_positive, s=round(abs(rndV_positive) * 5e4, 0), c='red', alpha=0.1)
                        figUP.set_xlabel(' '.join(['POSITIVE COUNT:', str(len(positive_count))]))
                        positiveCount = positiveCount + 1
                    else:
                        negative_count.append(var)
                        figDN.scatter(rndV_negative, rndV_negative, s=round(abs(rndV_negative) * 5e4, 0), c='green', alpha=0.1)
                        figDN.set_xlabel(' '.join(['NEGATIVE COUNT:', str(len(negative_count))]))
                        negativeCount = negativeCount + 1
                scnnResults.append(scnnReturn.results)
                self._vars['output text'].set(str(scnnReturn.results))
                self.master.update()

            # Showing the counting results in the text box.
            countResults = "\nCounting Results generated at time: " + str(datetime.datetime.now()) + '\n' + \
                           "Overall Group Repeated at : " + str(len(countingResults)+1) + "\n" + \
                           'COUNT TYPE: ' + ('NEGATIVE' if countType == 1 else 'POSITIVE') + '\n' + \
                           "Positive Count: " + str(positiveCount) + ";  " + \
                           "Negative Count: " + str(negativeCount) + "\n" + \
                           "Total Count: " + str(positiveCount * (1 - countType) + negativeCount * countType) + "\n"
            countingResults.append(countResults)
            self._vars['output text'].set(str(countResults))
            self.master.update()

            if positiveCount * (1 - countType) + negativeCount * countType > maxCount:
                # In order to use tkinter for pretty output, I need to package all the results into one string.
                outputString = '\nCOUNT TYPE: ' + ('NEGATIVE' if countType == 1 else 'POSITIVE') + \
                               '\nTOTAL COUNT: ' + str(positiveCount * (1 - countType) + negativeCount * countType)
                for i in range(len(indexWanted)):
                    singleIndexresult = '\n'.join(["\nDeep Learning results of " + indexWanted[i],
                                                   '*' * 100,
                                                   str(sdlResults[i]),
                                                   '.' * 100,
                                                   str(sclnResults[i]),
                                                   str(scnnResults[i]),
                                                   "=" * 100])
                    outputString = "\n".join([outputString, singleIndexresult])
                    self._vars['output text'].set(outputString)
                    self.master.update()
                    # Saving output string to text
                    saving_to_file = open(Path(dataDirName, '-'.join([str(datetime.datetime.now().date()), "IV_many2one_outputStringFINANCE.txt"])), 'w')
                    saving_to_file.write(countResults + outputString)
                    saving_to_file.close()
                    maxCount = positiveCount * (1 - countType) + negativeCount * countType

            if positiveCount * (1 - countType) + negativeCount * countType >= countThreshold:
                break
            elif self._vars['stop flag'].get():
                self._vars['stop flag'].set(value=False)
                self.master.update()
                break

class Application(tk.Tk):
    def __init__(self, *args, indexWanted_CU0=None, indexWanted_RB0=None,
                 indexWanted_SCM=None, indexWanted_FINANCE=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.indexWanted_CU0 = indexWanted_CU0
        self.indexWanted_RB0 = indexWanted_RB0
        self.indexWanted_SCM = indexWanted_SCM
        self.indexWanted_FINANCE = indexWanted_FINANCE
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
                                                 indexWanted_SCM=indexWanted_SCM,
                                                 indexWanted_FINANCE=indexWanted_FINANCE)
        self.processingWindow.grid(row=1, padx=10, sticky=(tk.W+tk.E))

        self.status = tk.StringVar()
        ttk.Label(
            self, textvariable=self.status
        ).grid(row=99, padx=10, sticky=tk.W+tk.E)

    def _to_status(self, text):
        self.status.set(text)

App = Application(indexWanted_CU0=indexWanted_CU0,
                  indexWanted_RB0=indexWanted_RB0,
                  indexWanted_SCM=indexWanted_SCM,
                  indexWanted_FINANCE=indexWanted_FINANCE)
App.mainloop()
print('\nPRETTY DONE AS WELL!', '='*200)
