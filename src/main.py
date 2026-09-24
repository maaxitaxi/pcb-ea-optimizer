# main.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import copy
import math
import os
import queue
import re
import threading
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import config
from config import POP_SIZE, NM_TO_MM, MM_TO_NM
from models import Genome
from ea_engine import (
    create_scenario, random_placement, normalize_population_fitness, evolve_one_generation,
    compute_tracelength_fitness, compute_overlap_penalty, compute_bbox_area, compute_crossing_penalty
)
import dsn_parser
import pipeline


def _fmt_mm(value_nm: float) -> str:
    return f"{value_nm * NM_TO_MM:.1f}".rstrip("0").rstrip(".")


class PCBOptimizerApp:
    CANVAS_PADDING = 40
    CANVAS_SIZE    = 720

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("PCB Placement Optimizer")
        self.root.configure(bg="#121214")

        self.footprints, self.netlist = create_scenario()
        self.design: Optional[dsn_parser.DsnDesign] = None  # gesetzt, sobald eine KiCad-DSN geladen ist
        self.dsn_path: Optional[str] = None
        self._best_signature = None
        self.stagnant_generations = 0
        self.is_routing = False
        self._route_queue: "queue.Queue[Tuple[str, object]]" = queue.Queue()

        self.population: List[Genome] = []
        self.fitness_vals: List[float] = []
        self.generation = 0
        self.best_genome: Optional[Genome] = None
        self.best_fitness: float = 0.0
        self.is_running = False
        self.history: dict = {"gen": [], "fitness": [], "trace_mm": [], "bbox_mm2": [], "overlap": []}
        self.stats_window: Optional[tk.Toplevel] = None
        self._stats_canvas = None
        self._stats_axes = None

        self._update_scale()
        self._build_gui()
        self.reset()

    def _update_scale(self):
        self.scale = (self.CANVAS_SIZE - 2 * self.CANVAS_PADDING) / max(config.BOARD_W, config.BOARD_H)

    def _build_gui(self):
        main_frame = tk.Frame(self.root, bg="#121214")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        canvas_frame = tk.Frame(main_frame, bg="#1E1E24", bd=1, relief=tk.SOLID)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, width=self.CANVAS_SIZE, height=self.CANVAS_SIZE, bg="#151518", highlightthickness=0)
        self.canvas.pack(padx=10, pady=10)

        sidebar = tk.Frame(main_frame, bg="#1E1E24", width=320, bd=1, relief=tk.SOLID)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=(15, 0))
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="🤖 PCB EA OPTIMIZER", font=("Segoe UI", 14, "bold"), fg="#4E9F3D", bg="#1E1E24").pack(pady=(15, 2))
        tk.Label(sidebar, text="EDA Automation Framework", font=("Segoe UI", 9, "italic"), fg="#6E6E73", bg="#1E1E24").pack(pady=(0, 20))

        stats_card = tk.LabelFrame(sidebar, text=" TELEMETRIE ", font=("Consolas", 9, "bold"), fg="#4E9F3D", bg="#151518", bd=1, padx=10, pady=10)
        stats_card.pack(fill=tk.X, padx=15, pady=5)

        self.lbl_gen = self._add_stat_row(stats_card, "Generation:", "0")
        self.lbl_fit = self._add_stat_row(stats_card, "Beste Fitness:", "Initializing...")
        self.lbl_overlap = self._add_stat_row(stats_card, "Overlap Pen:", "0.0")

        btn_zone = tk.Frame(sidebar, bg="#1E1E24")
        btn_zone.pack(fill=tk.X, padx=15, pady=10)

        self.btn_next = tk.Button(btn_zone, text="▶ Nächste Generation", font=("Segoe UI", 10, "bold"), bg="#2E3A46", fg="#E4E4E7", bd=0, pady=8, cursor="hand2", activebackground="#3E4A56", command=self.next_generation)
        self.btn_next.pack(fill=tk.X, pady=3)

        self.btn_auto = tk.Button(btn_zone, text="⚡ Auto-Run (Dauerlauf)", font=("Segoe UI", 10, "bold"), bg="#4E9F3D", fg="#121214", bd=0, pady=8, cursor="hand2", activebackground="#5EA44D", command=self.toggle_auto_run)
        self.btn_auto.pack(fill=tk.X, pady=3)

        self.btn_reset = tk.Button(btn_zone, text="↺ System Reset", font=("Segoe UI", 10, "bold"), bg="#3A1F2D", fg="#F07171", bd=0, pady=8, cursor="hand2", activebackground="#4A2F3D", command=self.reset)
        self.btn_reset.pack(fill=tk.X, pady=3)

        self.btn_stats = tk.Button(btn_zone, text="📊 Statistik anzeigen", font=("Segoe UI", 10, "bold"), bg="#2E3A46", fg="#E4E4E7", bd=0, pady=8, cursor="hand2", activebackground="#3E4A56", command=self.show_statistics)
        self.btn_stats.pack(fill=tk.X, pady=3)

        self.btn_export = tk.Button(btn_zone, text="💾 Als .DSN speichern", font=("Segoe UI", 10, "bold"), bg="#1D4ED8", fg="#FFFFFF", bd=0, pady=8, cursor="hand2", activebackground="#2563EB", command=self.export_dsn)
        self.btn_export.pack(fill=tk.X, pady=3)

        kicad_card = tk.LabelFrame(sidebar, text=" KICAD ", font=("Consolas", 9, "bold"), fg="#4E9F3D", bg="#151518", bd=1, padx=10, pady=8)
        kicad_card.pack(fill=tk.X, padx=15, pady=5)

        tk.Button(kicad_card, text="📂 KiCad-DSN laden", font=("Segoe UI", 10, "bold"), bg="#2E3A46", fg="#E4E4E7", bd=0, pady=6, cursor="hand2", activebackground="#3E4A56", command=self.load_dsn).pack(fill=tk.X, pady=2)

        margin_row = tk.Frame(kicad_card, bg="#151518")
        margin_row.pack(fill=tk.X, pady=4)
        tk.Label(margin_row, text="Bauteil-Abstand (mm):", font=("Consolas", 9), fg="#71717A", bg="#151518").pack(side=tk.LEFT)
        self.margin_var = tk.StringVar(value=_fmt_mm(config.COURTYARD_MARGIN))
        margin_box = tk.Spinbox(margin_row, from_=0.0, to=10.0, increment=0.5, width=5, textvariable=self.margin_var, command=self._apply_margin, bg="#1E1E24", fg="#E4E4E7", buttonbackground="#2E3A46", relief=tk.FLAT)
        margin_box.pack(side=tk.RIGHT)
        margin_box.bind("<Return>", self._apply_margin)
        margin_box.bind("<FocusOut>", self._apply_margin)

        self.btn_route = tk.Button(kicad_card, text="🔀 Routen → .ses speichern", font=("Segoe UI", 10, "bold"), bg="#1D4ED8", fg="#FFFFFF", bd=0, pady=6, cursor="hand2", activebackground="#2563EB", command=self.route_to_ses)
        self.btn_route.pack(fill=tk.X, pady=2)

        self.lbl_kicad = tk.Label(kicad_card, text="Demo-Szenario aktiv", font=("Segoe UI", 8), fg="#A1A1AA", bg="#151518", wraplength=260, justify=tk.LEFT, anchor="w")
        self.lbl_kicad.pack(fill=tk.X, pady=(4, 0))

        legend_card = tk.LabelFrame(sidebar, text=" BAUTEILE ", font=("Consolas", 9, "bold"), fg="#4E9F3D", bg="#151518", bd=1, padx=10, pady=10)
        legend_card.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        legend_scroll = tk.Scrollbar(legend_card, bg="#1E1E24", troughcolor="#151518", bd=0)
        legend_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        # Text-Widget statt einzelner Labels: scrollbar, auch bei Boards mit vielen Bauteilen
        self.legend_text = tk.Text(legend_card, bg="#151518", fg="#A1A1AA", font=("Segoe UI", 9), bd=0, highlightthickness=0,
                                   spacing1=2, spacing3=2, cursor="arrow", yscrollcommand=legend_scroll.set)
        self.legend_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        legend_scroll.config(command=self.legend_text.yview)
        self._refresh_legend()

    def _refresh_legend(self):
        self.legend_text.config(state=tk.NORMAL)
        self.legend_text.delete("1.0", tk.END)
        for fp in self.footprints:
            self.legend_text.tag_config(fp.color, foreground=fp.color, font=("Consolas", 12))
            self.legend_text.insert(tk.END, "■ ", fp.color)
            lock = " 🔒" if fp.fixed_placement is not None else ""
            self.legend_text.insert(tk.END, f"{fp.ref} ({_fmt_mm(fp.width)}x{_fmt_mm(fp.height)}mm){lock}\n")
        self.legend_text.config(state=tk.DISABLED)

    def _add_stat_row(self, parent, label: str, default_val: str) -> tk.Label:
        row = tk.Frame(parent, bg="#151518")
        row.pack(fill=tk.X, pady=3)
        tk.Label(row, text=label, font=("Consolas", 10), fg="#71717A", bg="#151518", width=14, anchor="w").pack(side=tk.LEFT)
        val_lbl = tk.Label(row, text=default_val, font=("Consolas", 10, "bold"), fg="#E4E4E7", bg="#151518")
        val_lbl.pack(side=tk.RIGHT)
        return val_lbl

    def reset(self):
        if self.is_running:
            self.toggle_auto_run()
        self.generation = 0
        self.best_fitness = 0.0
        self.best_genome = None
        self.history = {"gen": [], "fitness": [], "trace_mm": [], "bbox_mm2": [], "overlap": []}

        self.population = [random_placement(self.footprints) for _ in range(POP_SIZE)]
        if self.design is not None:
            self.population[-1] = copy.deepcopy(self.design.initial_genome)  # KiCad-Platzierung als Startindividuum
        self.fitness_vals = normalize_population_fitness(self.population, self.netlist)
        self._best_signature, self.stagnant_generations = None, 0

        self._update_best()
        self._update_status()
        self._draw()

    def next_generation(self):
        self.population, self.fitness_vals = evolve_one_generation(self.population, self.fitness_vals, self.netlist)
        self.generation += 1
        self._update_best()
        self._update_status()
        self._draw()
        self._refresh_statistics()

    def toggle_auto_run(self):
        if self.is_running:
            self.is_running = False
            self.btn_auto.config(text="⚡ Auto-Run (Dauerlauf)", bg="#4E9F3D", fg="#121214")
        else:
            self.is_running = True
            self.btn_auto.config(text="⏹ Stoppen", bg="#D94343", fg="white")
            self._auto_run_loop()

    def _auto_run_loop(self):
        if not self.is_running:
            return
        for _ in range(5):
            self.population, self.fitness_vals = evolve_one_generation(self.population, self.fitness_vals, self.netlist)
            self.generation += 1
            self._update_best()
        self._update_status()
        self._draw()
        self._refresh_statistics()

        # Im KiCad-Modus terminiert der EA wie die CLI-Pipeline und bietet das Routing an
        if (self.design is not None and self.stagnant_generations >= pipeline.DEFAULT_PATIENCE
                and compute_overlap_penalty(self.best_genome) == 0):
            self.toggle_auto_run()
            self.lbl_kicad.config(text=f"Konvergiert nach {self.generation} Generationen "
                                       f"(seit {self.stagnant_generations} ohne Verbesserung).")
            if messagebox.askyesno("EA terminiert", f"Keine Verbesserung seit {self.stagnant_generations} Generationen.\n\n"
                                                    "Jetzt mit FreeRouting verdrahten und als .ses speichern?"):
                self.route_to_ses()
            return
        self.root.after(20, self._auto_run_loop)

    def _update_best(self):
        best_idx = max(range(len(self.fitness_vals)), key=lambda i: self.fitness_vals[i])
        self.best_fitness = self.fitness_vals[best_idx]
        self.best_genome = copy.deepcopy(self.population[best_idx])

        signature = tuple((c.x, c.y, c.rot) for c in self.best_genome)
        self.stagnant_generations = self.stagnant_generations + 1 if signature == self._best_signature else 0
        self._best_signature = signature
        self._record_history()

    def _record_history(self):
        trace = compute_tracelength_fitness(self.best_genome, self.netlist)
        bbox = compute_bbox_area(self.best_genome)
        overlap = compute_overlap_penalty(self.best_genome)
        self.history["gen"].append(self.generation)
        self.history["fitness"].append(self.best_fitness)
        self.history["trace_mm"].append(trace * NM_TO_MM if trace != float('inf') else math.nan)
        self.history["bbox_mm2"].append(bbox * NM_TO_MM ** 2 if bbox != float('inf') else math.nan)
        self.history["overlap"].append(overlap)

    def _update_status(self):
        overlap = compute_overlap_penalty(self.best_genome)
        crossings = compute_crossing_penalty(self.best_genome, self.netlist)
        self.lbl_gen.config(text=str(self.generation))
        self.lbl_fit.config(text=f"{self.best_fitness:.4f}", fg="#4E9F3D" if overlap == 0 and crossings == 0 else "#F07171")
        self.lbl_overlap.config(text=f"{overlap:.2f} (X:{int(crossings)})", fg="#4E9F3D" if overlap == 0 and crossings == 0 else "#F07171")

    def _nm_to_canvas(self, x_nm: int, y_nm: int) -> Tuple[float, float]:
        if self.design is not None:
            y_nm = config.BOARD_H - y_nm  # DSN-Koordinaten haben y nach oben – so sieht es aus wie in KiCad
        cx = self.CANVAS_PADDING + x_nm * self.scale
        cy = self.CANVAS_PADDING + y_nm * self.scale
        return cx, cy

    def _draw(self):
        self.canvas.delete("all")
        genome = self.best_genome
        if genome is None:
            return

        grid = 25 * MM_TO_NM if max(config.BOARD_W, config.BOARD_H) > 100 * MM_TO_NM else 10 * MM_TO_NM
        for x in range(0, int(config.BOARD_W) + 1, grid):
            cx1, cy1 = self._nm_to_canvas(x, 0)
            cx2, cy2 = self._nm_to_canvas(x, config.BOARD_H)
            self.canvas.create_line(cx1, cy1, cx2, cy2, fill="#1F1F24")
        for y in range(0, int(config.BOARD_H) + 1, grid):
            cx1, cy1 = self._nm_to_canvas(0, y)
            cx2, cy2 = self._nm_to_canvas(config.BOARD_W, y)
            self.canvas.create_line(cx1, cy1, cx2, cy2, fill="#1F1F24")

        x1, y1 = self._nm_to_canvas(0, 0)
        x2, y2 = self._nm_to_canvas(config.BOARD_W, config.BOARD_H)
        self.canvas.create_rectangle(x1, y1, x2, y2, outline="#4E9F3D", width=2)
        self.canvas.create_text(min(x1, x2) + 10, min(y1, y2) + 15, text=f"Canvas: {_fmt_mm(config.BOARD_W)}x{_fmt_mm(config.BOARD_H)}mm", fill="#4E9F3D", font=("Consolas", 9), anchor="w")

        pin_positions = {}
        for comp in genome:
            for pin_id, _, abs_x, abs_y in comp.get_pin_positions():
                pin_positions[(comp.footprint.ref, pin_id)] = (abs_x, abs_y)

        for net_id, connections in self.netlist.items():
            net_pins = [pin_positions[key] for ref, pid in connections if (key := (ref, pid)) in pin_positions]
            for i in range(len(net_pins)):
                for j in range(i + 1, len(net_pins)):
                    px1, py1 = self._nm_to_canvas(*net_pins[i])
                    px2, py2 = self._nm_to_canvas(*net_pins[j])
                    self.canvas.create_line(px1, py1, px2, py2, fill="#3F4E4F", width=1, dash=(2, 4))

        for comp in genome:
            w, h = comp._rotated_dims()
            x_min, y_min = comp.x - w // 2, comp.y - h // 2
            x_max, y_max = x_min + w, y_min + h
            cx1, cy1 = self._nm_to_canvas(x_min, y_min)
            cx2, cy2 = self._nm_to_canvas(x_max, y_max)

            locked = comp.footprint.fixed_placement is not None
            self.canvas.create_rectangle(cx1+3, cy1+3, cx2+3, cy2+3, fill="#000000", outline="")
            self.canvas.create_rectangle(cx1, cy1, cx2, cy2, fill=comp.footprint.color,
                                         outline="#FFD27A" if locked else "#FFFFFF", width=2 if locked else 1)
            self.canvas.create_text((cx1+cx2)/2, (cy1+cy2)/2, text=comp.footprint.ref, fill="#121214", font=("Consolas", 10, "bold"))
            if comp.rot != 0 and abs(cx2 - cx1) > 40 and abs(cy2 - cy1) > 30:  # bei kleinen Bauteilen würde der Winkel den Namen verdecken
                self.canvas.create_text(min(cx1, cx2)+4, min(cy1, cy2)+4, text=f"{comp.rot}°", fill="#FFFFFF", font=("Consolas", 7), anchor="nw")

            for _, _, abs_x, abs_y in comp.get_pin_positions():
                pcx, pcy = self._nm_to_canvas(abs_x, abs_y)
                self.canvas.create_rectangle(pcx-3, pcy-3, pcx+3, pcy+3, fill="#FFB344", outline="#D97706")

    def show_statistics(self):
        if not self.history["gen"]:
            return

        if self.stats_window is None or not self.stats_window.winfo_exists():
            self.stats_window = tk.Toplevel(self.root)
            self.stats_window.title("EA Statistik — Verlauf über Generationen")
            self.stats_window.configure(bg="#121214")
            self.stats_window.geometry("780x1080")

            fig = Figure(figsize=(7.6, 11.2), dpi=100, facecolor="#121214")
            self._stats_axes = fig.subplots(4, 1, sharex=True)
            fig.subplots_adjust(left=0.15, right=0.96, top=0.97, bottom=0.06, hspace=0.45)

            self._stats_canvas = FigureCanvasTkAgg(fig, master=self.stats_window)
            self._stats_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        else:
            self.stats_window.lift()

        self._draw_statistics()

    def _draw_statistics(self):
        if self._stats_canvas is None:
            return

        gens = self.history["gen"]
        series = [
            (self.history["fitness"], "Beste Fitness (>=1.0 ist gültig)", "#4E9F3D"),
            (self.history["trace_mm"], "Leitungslänge (mm)", "#FFB344"),
            (self.history["bbox_mm2"], "Bounding-Box (mm²)", "#5FB3D9"),
            (self.history["overlap"], "Overlap-Penalty", "#F07171"),
        ]

        for ax, (values, label, color) in zip(self._stats_axes, series):
            ax.clear()
            ax.plot(gens, values, color=color, linewidth=1.5)
            ax.set_facecolor("#151518")
            ax.set_ylabel(label, color="#A1A1AA", fontsize=9)
            ax.tick_params(colors="#71717A", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#3F3F46")
            ax.grid(True, color="#27272A", linewidth=0.5)

        self._stats_axes[-1].set_xlabel("Generation", color="#A1A1AA", fontsize=9)
        self._stats_canvas.draw()

    def _refresh_statistics(self):
        if self.stats_window is not None and self.stats_window.winfo_exists():
            self._draw_statistics()

    def load_dsn(self):
        path = filedialog.askopenfilename(
            title="KiCad-DSN laden (KiCad: Datei → Exportieren → Specctra DSN)",
            filetypes=[("Specctra DSN", "*.dsn"), ("Alle Dateien", "*.*")],
        )
        if not path:
            return
        try:
            design = dsn_parser.load_dsn(path)
        except Exception as e:
            messagebox.showerror("Fehler", f"Die DSN-Datei konnte nicht gelesen werden:\n{e}")
            return

        if self.is_running:
            self.toggle_auto_run()
        self.design, self.dsn_path = design, path
        self.footprints, self.netlist = design.footprints, design.netlist
        config.set_board_size(*design.board_size)
        self._update_scale()
        self._refresh_legend()
        self.reset()

        locked = sum(1 for fp in self.footprints if fp.fixed_placement is not None)
        self.lbl_kicad.config(text=f"{os.path.basename(path)}: {len(self.footprints)} Bauteile "
                                   f"({locked} gesperrt), {len(self.netlist)} Netze")
        if design.warnings:
            messagebox.showwarning("Hinweise beim Einlesen", "\n".join(design.warnings[:15]))

    def _apply_margin(self, _event=None):
        try:
            margin_mm = float(self.margin_var.get().replace(",", "."))
        except ValueError:
            return
        margin_nm = int(max(0.0, margin_mm) * MM_TO_NM)
        if margin_nm == config.COURTYARD_MARGIN:
            return
        config.COURTYARD_MARGIN = margin_nm
        self.fitness_vals = normalize_population_fitness(self.population, self.netlist)
        self._update_best()
        self._update_status()
        self._draw()

    def route_to_ses(self):
        if self.design is None:
            messagebox.showinfo("Keine KiCad-Datei", "Bitte zuerst eine KiCad-DSN laden.\n\n"
                                                     "In KiCad: Datei → Exportieren → Specctra DSN …")
            return
        if self.is_routing:
            return
        overlap = compute_overlap_penalty(self.best_genome)
        if overlap > 0:
            messagebox.showwarning("Layout überlappt",
                                   f"Das beste Layout hat noch Überlappungen (Overlap Penalty: {overlap:.2f}).\n\n"
                                   "EA weiterlaufen lassen oder den Bauteil-Abstand verkleinern.")
            return
        try:
            jar = pipeline.find_freerouting_jar()
        except pipeline.PipelineError as e:
            messagebox.showerror("FreeRouting fehlt", str(e))
            return

        stem = os.path.splitext(self.dsn_path)[0]
        ses_path = filedialog.asksaveasfilename(
            title="Specctra-Session speichern",
            defaultextension=".ses",
            initialdir=os.path.dirname(stem),
            initialfile=os.path.basename(stem) + ".ses",
            filetypes=[("Specctra Session", "*.ses"), ("Alle Dateien", "*.*")],
        )
        if not ses_path:
            return
        if self.is_running:
            self.toggle_auto_run()

        optimized_dsn = os.path.splitext(ses_path)[0] + "_ea.dsn"
        try:
            dsn_parser.write_dsn(self.design, self.best_genome, optimized_dsn)
        except OSError as e:
            messagebox.showerror("Fehler", f"Optimierte DSN konnte nicht geschrieben werden:\n{e}")
            return

        self.is_routing = True
        self.btn_route.config(state=tk.DISABLED, text="⏳ FreeRouting läuft …")
        self.lbl_kicad.config(text="FreeRouting startet …")
        threading.Thread(target=self._route_worker, args=(jar, optimized_dsn, ses_path), daemon=True).start()
        self.root.after(200, self._poll_routing)

    def _route_worker(self, jar: str, dsn_path: str, ses_path: str):
        # Läuft im Hintergrund-Thread: nur über die Queue mit Tkinter kommunizieren
        try:
            complete = pipeline.run_freerouting(jar, dsn_path, ses_path, pipeline.DEFAULT_PASSES,
                                                log=lambda line: self._route_queue.put(("log", line)))
            self._route_queue.put(("done", (ses_path, complete)))
        except Exception as e:
            self._route_queue.put(("error", str(e)))

    def _poll_routing(self):
        try:
            while True:
                kind, payload = self._route_queue.get_nowait()
                if kind == "log":
                    # Zeitstempel und Job-ID von FreeRouting abschneiden
                    self.lbl_kicad.config(text=re.sub(r"^.*?\]\s*", "", payload)[:160])
                else:
                    self._finish_routing(kind, payload)
                    return
        except queue.Empty:
            pass
        self.root.after(200, self._poll_routing)

    def _finish_routing(self, kind: str, payload):
        self.is_routing = False
        self.btn_route.config(state=tk.NORMAL, text="🔀 Routen → .ses speichern")
        if kind == "error":
            self.lbl_kicad.config(text="Routing fehlgeschlagen.")
            messagebox.showerror("FreeRouting", payload)
            return

        ses_path, complete = payload
        self.lbl_kicad.config(text=f"Gespeichert: {os.path.basename(ses_path)}")
        message = (f"Session gespeichert:\n{ses_path}\n\n"
                   "In KiCad: Datei → Importieren → Specctra Session …")
        if complete:
            messagebox.showinfo("Routing abgeschlossen", message)
        else:
            messagebox.showwarning("Routing unvollständig",
                                   "Nicht alle Verbindungen konnten geroutet werden.\n\n" + message)

    def export_dsn(self):
        if not self.best_genome:
            messagebox.showwarning("Warnung", "Kein Layout zum Exportieren vorhanden!")
            return

        overlap = compute_overlap_penalty(self.best_genome)
        if overlap > 0:
            messagebox.showwarning(
                "Layout überlappt",
                f"Das Layout hat noch Bauteil-Überlappungen (Overlap Penalty: {overlap:.2f}).\n\n"
                "Bitte lass den EA weiterlaufen, bis Overlap = 0.0 erreicht ist!",
            )
            return

        crossings = compute_crossing_penalty(self.best_genome, self.netlist)
        if crossings > 0:
            proceed = messagebox.askyesno(
                "Warnung: Kreuzungen vorhanden",
                f"Das Layout hat noch {int(crossings)} Netz-Kreuzungen.\n\n"
                "FreeRouting kann evtl. mehr Durchkontaktierungen (Vias) benötigen.\n"
                "Trotzdem als .DSN exportieren?",
            )
            if not proceed:
                return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".dsn",
            filetypes=[("Specctra DSN", "*.dsn"), ("Alle Dateien", "*.*")],
            title="Layout als DSN speichern",
        )
        if not file_path:
            return

        try:
            if self.design is not None:
                dsn_content = dsn_parser.render_dsn(self.design, self.best_genome)
            else:
                dsn_content = self._generate_dsn()
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(dsn_content)
            messagebox.showinfo("Erfolg", f"Layout erfolgreich gespeichert unter:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Speichern der DSN-Datei:\n{e}")

    def _generate_dsn(self) -> str:
        lines = []
        lines.append('(pcb pcb_optimizer')
        lines.append('  (parser')
        lines.append('    (string_quote ")')
        lines.append('    (space_in_quoted_tokens on)')
        lines.append('    (host_cad "KiCad")')
        lines.append('  )')
        lines.append('  (resolution mm 1000)')
        lines.append('  (unit mm)')
        
        lines.append('  (structure')
        # Assign explicit routing directions to force 2-layer routing
        lines.append('    (layer F.Cu (type signal) (property (index 0)) (direction horizontal))')
        lines.append('    (layer B.Cu (type signal) (property (index 1)) (direction vertical))')
        lines.append('    (boundary')
        lines.append(f'      (rect pcb 0 0 {config.BOARD_WIDTH_MM} {config.BOARD_HEIGHT_MM})')
        lines.append('    )')
        lines.append('    (rule')
        lines.append('      (width 0.20)')
        lines.append('      (clearance 0.12)')
        lines.append('      (via Via_Default)')
        lines.append('    )')
        lines.append('  )')

        lines.append('  (placement')
        for comp in self.best_genome:
            x_mm = comp.x * NM_TO_MM
            y_mm = comp.y * NM_TO_MM
            lines.append(f'    (component "LIB_{comp.footprint.ref}"')
            lines.append(f'      (place {comp.footprint.ref} {x_mm:.4f} {y_mm:.4f} front {comp.rot:.1f})')
            lines.append('    )')
        lines.append('  )')

        lines.append('  (library')
        lines.append('    (padstack "Pad_SMD"')
        lines.append('      (shape (rect F.Cu -0.35 -0.35 0.35 0.35))')
        lines.append('      (attach on)')
        lines.append('    )')
        lines.append('    (padstack "Via_Default"')
        lines.append('      (shape (circle F.Cu 0.5 0 0))')
        lines.append('      (shape (circle B.Cu 0.5 0 0))')
        lines.append('      (attach on)')
        lines.append('    )')
        
        unique_footprints = {comp.footprint.ref: comp.footprint for comp in self.best_genome}
        for ref, fp in unique_footprints.items():
            lines.append(f'    (image "LIB_{ref}"')
            for pin in fp.pins:
                px_mm = pin.rel_x * NM_TO_MM
                py_mm = pin.rel_y * NM_TO_MM
                lines.append(f'      (pin "Pad_SMD" "{pin.pin_id}" {px_mm:.4f} {py_mm:.4f})')
            lines.append('    )')
        lines.append('  )')

        lines.append('  (network')
        for net_id, connections in self.netlist.items():
            pins_str = " ".join([f'"{ref}"-"{pid}"' for ref, pid in connections])
            lines.append(f'    (net "{net_id}" (pins {pins_str}))')
        
        net_names = " ".join([f'"{net_id}"' for net_id in self.netlist.keys()])
        lines.append(f'    (class "default" {net_names}')
        lines.append('      (circuit')
        lines.append('        (use_layer F.Cu B.Cu)')
        lines.append('        (use_via Via_Default)')
        lines.append('      )')
        lines.append('      (rule')
        lines.append('        (width 0.20)')
        lines.append('        (clearance 0.12)')
        lines.append('        (via Via_Default)')
        lines.append('      )')
        lines.append('    )')
        lines.append('  )')

        lines.append(')')
        return "\n".join(lines)


def main():
    root = tk.Tk()
    root.geometry("1120x800")
    root.configure(bg="#121214")
    app = PCBOptimizerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()