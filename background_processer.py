import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image as pillow
from PIL import ImageColor
from pathlib import Path as path

import biref_process


class TextRedirector:
    """Pipes stdout/stderr writes to a tkinter Text widget, thread-safely."""

    def __init__(self, widget):
        self.widget = widget

    def write(self, text):
        self.widget.after(0, self._append, text)

    def _append(self, text):
        self.widget.configure(state='normal')
        # Handle \r (tqdm in-place progress updates): overwrite current line
        parts = text.split('\r')
        self.widget.insert('end', parts[0])
        for part in parts[1:]:
            self.widget.delete('end-1l linestart', 'end-1c')
            self.widget.insert('end', part)
        self.widget.see('end')
        self.widget.configure(state='disabled')

    def flush(self):
        pass


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Background Processor")
        self.resizable(True, True)

        self.input_dir      = tk.StringVar()
        self.transparent_dir = tk.StringVar()
        self.background_dir  = tk.StringVar()
        self.do_background   = tk.BooleanVar(value=False)
        self.background_hex  = tk.StringVar()

        self.config_frame   = tk.Frame(self)
        self.progress_frame = tk.Frame(self)

        self._build_config()
        self._build_progress()
        self._show(self.config_frame)

    def _show(self, frame):
        self.config_frame.pack_forget()
        self.progress_frame.pack_forget()
        frame.pack(fill='both', expand=True, padx=20, pady=20)
        self.geometry('620x360' if frame is self.config_frame else '820x540')

    # ── Config page ───────────────────────────────────────────────────────────

    def _build_config(self):
        f = self.config_frame

        tk.Label(f, text="Background Processor", font=('Helvetica', 14, 'bold')).grid(
            row=0, column=0, columnspan=3, pady=(0, 14))

        tk.Checkbutton(f, text="Add background colour to images",
                       variable=self.do_background,
                       command=self._toggle_bg).grid(
            row=1, column=0, columnspan=3, sticky='w')

        tk.Label(f, text="Background colour (hex):").grid(row=2, column=0, sticky='w', pady=4)
        self.hex_entry = tk.Entry(f, textvariable=self.background_hex, width=12)
        self.hex_entry.grid(row=2, column=1, sticky='w')
        tk.Label(f, text='e.g. #ff0000 or ff0000').grid(row=2, column=2, sticky='w', padx=8)

        self._dir_row(f, 3, "Input directory:",        self.input_dir)
        self._dir_row(f, 4, "Transparent output:",     self.transparent_dir)

        tk.Label(f, text="Background output:").grid(row=5, column=0, sticky='w', pady=4)
        self.bg_dir_entry = tk.Entry(f, textvariable=self.background_dir, width=38)
        self.bg_dir_entry.grid(row=5, column=1, sticky='w')
        self.bg_dir_btn = tk.Button(f, text="Browse",
                                    command=lambda: self._pick(self.background_dir))
        self.bg_dir_btn.grid(row=5, column=2, padx=6)

        tk.Button(f, text="Start Processing", command=self._start,
                  bg='#2e7d32', fg='white', font=('Helvetica', 11, 'bold'),
                  padx=12, pady=6).grid(
            row=6, column=0, columnspan=3, pady=(20, 0))

        self._toggle_bg()

    def _dir_row(self, parent, row, label, var):
        tk.Label(parent, text=label).grid(row=row, column=0, sticky='w', pady=4)
        tk.Entry(parent, textvariable=var, width=38).grid(row=row, column=1, sticky='w')
        tk.Button(parent, text="Browse",
                  command=lambda v=var: self._pick(v)).grid(row=row, column=2, padx=6)

    def _toggle_bg(self):
        state = 'normal' if self.do_background.get() else 'disabled'
        self.hex_entry.config(state=state)
        self.bg_dir_entry.config(state=state)
        self.bg_dir_btn.config(state=state)

    def _pick(self, var):
        d = filedialog.askdirectory()
        if d:
            var.set(d)

    def _start(self):
        if not self.input_dir.get():
            messagebox.showerror("Missing field", "Select an input directory.")
            return
        if not self.transparent_dir.get():
            messagebox.showerror("Missing field", "Select a transparent output directory.")
            return
        if self.do_background.get():
            if not self.background_hex.get():
                messagebox.showerror("Missing field", "Enter a background hex colour.")
                return
            if not self.background_dir.get():
                messagebox.showerror("Missing field", "Select a background output directory.")
                return

        self._show(self.progress_frame)
        redir = TextRedirector(self.log)
        sys.stdout = redir
        sys.stderr = redir
        threading.Thread(target=self._run, daemon=True).start()

    # ── Progress page ─────────────────────────────────────────────────────────

    def _build_progress(self):
        f = self.progress_frame

        self.status_label = tk.Label(f, text="Processing…", font=('Helvetica', 13, 'bold'))
        self.status_label.pack(anchor='w', pady=(0, 8))

        text_frame = tk.Frame(f)
        text_frame.pack(fill='both', expand=True)

        self.log = tk.Text(text_frame, state='disabled', bg='#1e1e1e', fg='#d4d4d4',
                           font=('Courier', 10), wrap='none')
        vsb = tk.Scrollbar(text_frame, orient='vertical',   command=self.log.yview)
        hsb = tk.Scrollbar(text_frame, orient='horizontal', command=self.log.xview)
        self.log.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.pack(side='right',  fill='y')
        hsb.pack(side='bottom', fill='x')
        self.log.pack(fill='both', expand=True)

    # ── Processing (background thread) ───────────────────────────────────────

    def _run(self):
        try:
            biref_process.process_images(self.input_dir.get(), self.transparent_dir.get())

            if self.do_background.get():
                print("\nApplying background colour…")
                hex_val = self.background_hex.get().strip()
                if not hex_val.startswith('#'):
                    hex_val = '#' + hex_val
                bg_color = np.array(ImageColor.getrgb(hex_val) + (255,), dtype=np.uint8)

                out_files = list(path(self.transparent_dir.get()).glob('*.png'))
                path(self.background_dir.get()).mkdir(parents=True, exist_ok=True)
                for img_path in out_files:
                    img_np = np.array(pillow.open(img_path).convert("RGBA"))
                    img_np[img_np[:, :, 3] == 0] = bg_color
                    pillow.fromarray(img_np, 'RGBA').save(
                        path(self.background_dir.get()) / img_path.name, "PNG")
                    print(f"Background applied: {img_path.name}")

            print("\nFinished.")
            self.after(0, lambda: self.status_label.config(text="Done!"))
        except Exception as e:
            print(f"\nError: {e}")
            self.after(0, lambda: self.status_label.config(text="Error — see log"))


if __name__ == '__main__':
    App().mainloop()
