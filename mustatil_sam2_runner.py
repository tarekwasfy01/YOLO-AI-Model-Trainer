#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Small launcher for the full external Mustatil workflow.

This gives direct access to the external scripts from outside QGIS.
The QGIS plugin itself exposes the same tools in its dock.
"""
from __future__ import annotations
import subprocess, sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

BASE = Path(__file__).resolve().parent
PY = sys.executable

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Mustatil External Toolkit")
        self.geometry("760x420")
        self.project = tk.StringVar()
        self.data = tk.StringVar()
        self.images = tk.StringVar()
        self.labels = tk.StringVar()
        self.model = tk.StringVar(value="yolov8n.pt")
        self.log = tk.Text(self)
        self._build()

    def _build(self):
        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")
        ttk.Label(top, text="Project").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.project, width=70).grid(row=0, column=1, sticky="ew")
        ttk.Button(top, text="Browse", command=self.pick_project).grid(row=0, column=2)
        ttk.Button(top, text="Create Project", command=self.create_project).grid(row=1, column=1, sticky="ew", pady=3)

        ttk.Label(top, text="Training images").grid(row=2, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.images, width=70).grid(row=2, column=1, sticky="ew")
        ttk.Button(top, text="Browse", command=self.pick_images).grid(row=2, column=2)

        ttk.Label(top, text="Training labels").grid(row=3, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.labels, width=70).grid(row=3, column=1, sticky="ew")
        ttk.Button(top, text="Browse", command=self.pick_labels).grid(row=3, column=2)

        ttk.Label(top, text="YOLO data.yaml").grid(row=4, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.data, width=70).grid(row=4, column=1, sticky="ew")
        ttk.Button(top, text="Browse", command=self.pick_data).grid(row=4, column=2)

        ttk.Button(top, text="Create data.yaml + sort selected training data", command=self.prepare_data).grid(row=5, column=1, sticky="ew", pady=3)

        ttk.Label(top, text="Base model").grid(row=6, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.model, width=70).grid(row=6, column=1, sticky="ew")
        ttk.Button(top, text="Train YOLO", command=self.train_yolo).grid(row=7, column=1, sticky="ew", pady=3)
        ttk.Button(top, text="Train FormLearner", command=self.train_form).grid(row=8, column=1, sticky="ew", pady=3)
        top.columnconfigure(1, weight=1)

        self.log.pack(fill="both", expand=True, padx=8, pady=8)

    def pick_project(self):
        d = filedialog.askdirectory()
        if d:
            self.project.set(d)
            self.images.set(str(Path(d) / "images"))
            self.labels.set(str(Path(d) / "labels"))
            self.data.set(str(Path(d) / "yolo_datasets" / "data.yaml"))

    def pick_images(self):
        d = filedialog.askdirectory()
        if d:
            self.images.set(d)

    def pick_labels(self):
        d = filedialog.askdirectory()
        if d:
            self.labels.set(d)

    def pick_data(self):
        p = filedialog.askopenfilename(filetypes=[("YOLO data", "data.yaml *.yaml *.yml"), ("All", "*.*")])
        if p:
            self.data.set(p)

    def run(self, args):
        self.log.insert("end", "$ " + " ".join(args) + "\n")
        self.log.see("end")
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        def poll():
            line = p.stdout.readline()
            if line:
                self.log.insert("end", line)
                self.log.see("end")
                self.after(20, poll)
            elif p.poll() is None:
                self.after(100, poll)
            else:
                self.log.insert("end", f"\nProcess finished: {p.returncode}\n")
        self.after(20, poll)

    def create_project(self):
        if not self.project.get():
            self.pick_project()
        if self.project.get():
            self.run([PY, str(BASE/"mustatil_project_tools.py"), "--create-project", self.project.get()])

    def prepare_data(self):
        if not self.project.get():
            messagebox.showerror("Missing", "Select project folder first.")
            return
        args = [PY, str(BASE/"mustatil_project_tools.py"), "--create-yaml", self.project.get()]
        if self.images.get():
            args += ["--images-dir", self.images.get()]
        if self.labels.get():
            args += ["--labels-dir", self.labels.get()]
        self.data.set(str(Path(self.project.get()) / "yolo_datasets" / "data.yaml"))
        self.run(args)

    def train_yolo(self):
        if not self.data.get():
            messagebox.showerror("Missing", "Select data.yaml first.")
            return
        self.run([PY, str(BASE/"mustatil_yolo_trainer.py"), "--data", self.data.get(), "--model", self.model.get() or "yolov8n.pt"])

    def train_form(self):
        if not self.project.get():
            messagebox.showerror("Missing", "Select project folder first.")
            return
        out = str(Path(self.project.get()) / "trained_form_models" / "formlearner_model.json")
        self.run([PY, str(BASE/"mustatil_formlearner_trainer.py"), "--project", self.project.get(), "--output", out])

if __name__ == "__main__":
    App().mainloop()