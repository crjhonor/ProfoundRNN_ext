"""
The Main Menu class for DPM
"""

import tkinter as tk
from tkinter import messagebox

class MainMenu(tk.Menu):
    """The Application's main menu"""

    def __init__(self, parent, **kwargs):
        """Constructor for main menu"""
        super().__init__(parent, **kwargs)

        # The HELP menu
        help_menu = tk.Menu(self, tearoff=False)
        help_menu.add_command(label="About...", command=self.show_about)

        # The Dicingers menu
        dicingers_menu = tk.Menu(self, tearoff=False)
        dicingers_menu.add_command(
            label='TRIPLE DICER',
            command=self._event('<<TRIPLE DICER>>')
        )
        dicingers_menu.add_separator()
        dicingers_menu.add_command(
            label='Quit',
            command=self._event('<<Quit>>')
        )

        # The More menu
        more_menu = tk.Menu(self, tearoff=False)
        more_menu.add_command(
            label='Coming Soon...',
            command=self._event('<<Coming Soon>>')
        )

        # Add the menus in order to the main menu
        self.add_cascade(label='DICINGERS', menu=dicingers_menu)
        self.add_cascade(label='MORE', menu=more_menu)
        self.add_cascade(label='HELP', menu=help_menu)


    def show_about(self):
        """Show the about dialog"""
        about_message = "Dicinger Pro Max"
        about_detail = (
            'By: Chen Ru Jie\n'
            'Date: October, 2023\n'
            'For: Dicing The probabilities of tradings.\n'
            'Version: 1.0.0'
        )
        messagebox.showinfo(
            title='About', message=about_message, detail=about_detail
        )

    def _event(self, sequence):
        """Return a callback function that generates the sequence"""
        def callback(*_):
            root = self.master.winfo_toplevel()
            root.event_generate(sequence)
        return callback