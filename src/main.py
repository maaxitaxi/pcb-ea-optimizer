# main.py
import tkinter as tk
from tkinter import ttk
import copy
from typing import List, Optional, Tuple

from config import BOARD_W, BOARD_H, BOARD_WIDTH_MM, BOARD_HEIGHT_MM, POP_SIZE, NM_TO_MM
from models import Genome
from ea_engine import create_scenario, random_placement, normalize_population_fitness, evolve_one_generation

class PCBOptimizerApp:
    CANVAS_PADDING = 40
    CANVAS_SIZE    = 720

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("PCB Placement Optimizer")
        self.root.configure(bg="#121214")

        self.footprints, self.netlist = create_scenario()

        self.population: List[Genome] = []
        self.fitness_vals: List[float] = []
        self.generation = 0
        self.best_genome: Optional[Genome] = None
        self.best_fitness: float = 0.0
        self.is_running = False

        self.scale = (self.CANVAS_SIZE - 2 * self.CANVAS_PADDING) / max(BOARD_W, BOARD_H)
        self._build_gui()
        self.reset()

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
        self.lbl_inf = self._add_stat_row(stats_card, "Infeasible:", "0 / 0")

        btn_zone = tk.Frame(sidebar, bg="#1E1E24")
        btn_zone.pack(fill=tk.X, padx=15, pady=20)

        self.btn_next = tk.Button(btn_zone, text="▶ Nächste Generation", font=("Segoe UI", 10, "bold"), bg="#2E3A46", fg="#E4E4E7", bd=0, pady=10, cursor="hand2", activebackground="#3E4A56", command=self.next_generation)
        self.btn_next.pack(fill=tk.X, pady=4)

        self.btn_auto = tk.Button(btn_zone, text="⚡ Auto-Run (Dauerlauf)", font=("Segoe UI", 10, "bold"), bg="#4E9F3D", fg="#121214", bd=0, pady=10, cursor="hand2", activebackground="#5EA44D", command=self.toggle_auto_run)
        self.btn_auto.pack(fill=tk.X, pady=4)

        self.btn_reset = tk.Button(btn_zone, text="↺ System Reset", font=("Segoe UI", 10, "bold"), bg="#3A1F2D", fg="#F07171", bd=0, pady=10, cursor="hand2", activebackground="#4A2F3D", command=self.reset)
        self.btn_reset.pack(fill=tk.X, pady=4)

        legend_card = tk.LabelFrame(sidebar, text=" BAUTEILE ", font=("Consolas", 9, "bold"), fg="#4E9F3D", bg="#151518", bd=1, padx=10, pady=10)
        legend_card.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        for fp in self.footprints:
            row = tk.Frame(legend_card, bg="#151518")
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text="■", fg=fp.color, bg="#151518", font=("Consolas", 12)).pack(side=tk.LEFT)
            tk.Label(row, text=f"{fp.ref} ({fp.width*NM_TO_MM:.0f}x{fp.height*NM_TO_MM:.0f}mm)", fg="#A1A1AA", bg="#151518", font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=5)

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

        self.population = [random_placement(self.footprints) for _ in range(POP_SIZE)]
        self.fitness_vals = normalize_population_fitness(self.population, self.netlist)

        self._update_best()
        self._update_status()
        self._draw()

    def next_generation(self):
        self.population, self.fitness_vals = evolve_one_generation(self.population, self.fitness_vals, self.netlist)
        self.generation += 1
        self._update_best()
        self._update_status()
        self._draw()

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
        self.root.after(20, self._auto_run_loop)

    def _update_best(self):
        """Always track the best individual of the current generation (higher = better)."""
        best_idx = max(range(len(self.fitness_vals)), key=lambda i: self.fitness_vals[i])
        self.best_fitness = self.fitness_vals[best_idx]   # score in [0, 1]; 1.0 = best
        self.best_genome = copy.deepcopy(self.population[best_idx])

    def _update_status(self):
        self.lbl_gen.config(text=str(self.generation))
        self.lbl_fit.config(text=f"{self.best_fitness:.4f}", fg="#4E9F3D")

    def _nm_to_canvas(self, x_nm: int, y_nm: int) -> Tuple[float, float]:
        cx = self.CANVAS_PADDING + x_nm * self.scale
        cy = self.CANVAS_PADDING + y_nm * self.scale
        return cx, cy

    def _draw(self):
        self.canvas.delete("all")
        genome = self.best_genome
        if genome is None:
            return

        # Grid
        for x in range(0, int(BOARD_W) + 1, 25 * 1000000):
            cx1, cy1 = self._nm_to_canvas(x, 0)
            cx2, cy2 = self._nm_to_canvas(x, BOARD_H)
            self.canvas.create_line(cx1, cy1, cx2, cy2, fill="#1F1F24")
        for y in range(0, int(BOARD_H) + 1, 25 * 1000000):
            cx1, cy1 = self._nm_to_canvas(0, y)
            cx2, cy2 = self._nm_to_canvas(BOARD_W, y)
            self.canvas.create_line(cx1, cy1, cx2, cy2, fill="#1F1F24")

        # Board border
        x1, y1 = self._nm_to_canvas(0, 0)
        x2, y2 = self._nm_to_canvas(BOARD_W, BOARD_H)
        self.canvas.create_rectangle(x1, y1, x2, y2, outline="#4E9F3D", width=2)
        self.canvas.create_text(x1 + 10, y1 + 15, text=f"KiCad Canvas: {BOARD_WIDTH_MM}x{BOARD_HEIGHT_MM}mm", fill="#4E9F3D", font=("Consolas", 9), anchor="w")

        # Nets (ratsnest / Luftlinien)
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

        # Components
        for comp in genome:
            x_min, y_min, x_max, y_max = comp.get_bbox()
            cx1, cy1 = self._nm_to_canvas(x_min, y_min)
            cx2, cy2 = self._nm_to_canvas(x_max, y_max)

            self.canvas.create_rectangle(cx1+3, cy1+3, cx2+3, cy2+3, fill="#000000", outline="")
            self.canvas.create_rectangle(cx1, cy1, cx2, cy2, fill=comp.footprint.color, outline="#FFFFFF", width=1)
            self.canvas.create_text((cx1+cx2)/2, (cy1+cy2)/2, text=comp.footprint.ref, fill="#121214", font=("Consolas", 10, "bold"))
            if comp.rot != 0:
                self.canvas.create_text(cx1+4, cy1+4, text=f"{comp.rot}°", fill="#FFFFFF", font=("Consolas", 7), anchor="nw")

            for _, _, abs_x, abs_y in comp.get_pin_positions():
                pcx, pcy = self._nm_to_canvas(abs_x, abs_y)
                self.canvas.create_rectangle(pcx-3, pcy-3, pcx+3, pcy+3, fill="#FFB344", outline="#D97706")


def main():
    root = tk.Tk()
    root.geometry("1120x800")
    root.configure(bg="#121214")
    app = PCBOptimizerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()