# pipeline.py
"""KiCad → EA-Platzierung → FreeRouting → KiCad

  1. KiCad:   Datei → Exportieren → Specctra DSN …       → board.dsn
  2. Hier:    python src/pipeline.py board.dsn            → board.ses
  3. KiCad:   Datei → Importieren → Specctra Session …   ← board.ses
"""
import argparse
import glob
import os
import random
import subprocess
import sys
import time
from typing import Callable, List, Optional

import config
from config import MM_TO_NM, NM_TO_MM, POP_SIZE
from models import Footprint, Genome
from ea_engine import (
    random_placement, normalize_population_fitness, evolve_one_generation,
    compute_overlap_penalty, compute_tracelength_fitness, compute_crossing_penalty, compute_bbox_area,
)
import dsn_parser

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Eine als Snap installierte IDE (z.B. VS Code) setzt diese Variablen auf Snap-Bibliotheken,
# an denen Java/GTK beim Start von FreeRouting abstürzt.
SNAP_GTK_VARS = (
    "GTK_PATH", "GTK_EXE_PREFIX", "GTK_IM_MODULE_FILE", "GIO_MODULE_DIR",
    "GDK_PIXBUF_MODULE_FILE", "GDK_PIXBUF_MODULEDIR", "LOCPATH", "GSETTINGS_SCHEMA_DIR",
)

DEFAULT_PATIENCE = 60  # Generationen ohne Verbesserung bis zur Terminierung
DEFAULT_PASSES = 20    # max. FreeRouting-Durchläufe


class PipelineError(RuntimeError):
    pass


def run_ea(
    footprints: List[Footprint],
    netlist,
    seed_genome: Genome,
    max_generations: int,
    patience: int,
    log_every: int,
) -> Genome:
    """Evolviert, bis das beste Layout überlappungsfrei ist und sich `patience`
    Generationen lang nicht mehr verändert hat – spätestens nach `max_generations`."""
    population = [random_placement(footprints) for _ in range(POP_SIZE - 1)] + [seed_genome]
    fitness = normalize_population_fitness(population, netlist)

    best_signature, stagnant = None, 0
    best = seed_genome
    for generation in range(1, max_generations + 1):
        population, fitness = evolve_one_generation(population, fitness, netlist)
        best = population[max(range(len(fitness)), key=fitness.__getitem__)]

        signature = tuple((c.x, c.y, c.rot) for c in best)
        stagnant = stagnant + 1 if signature == best_signature else 0
        best_signature = signature
        overlap = compute_overlap_penalty(best)

        if generation % log_every == 0 or generation == 1:
            trace = compute_tracelength_fitness(best, netlist)
            trace_txt = f"{trace * NM_TO_MM:9.1f} mm" if trace != float("inf") else "   außerhalb"
            print(f"  Gen {generation:5d} | Overlap {overlap:6.2f} | Trace-Score {trace_txt} | "
                  f"Kreuzungen {int(compute_crossing_penalty(best, netlist)):3d} | unverändert seit {stagnant}")

        if overlap == 0.0 and stagnant >= patience:
            print(f"  → Terminiert nach {generation} Generationen (seit {patience} Generationen keine Verbesserung)")
            break
    else:
        print(f"  → Maximale Generationenzahl ({max_generations}) erreicht")
    return best


def find_freerouting_jar(explicit: Optional[str] = None) -> str:
    candidates = [explicit] if explicit else []
    if os.environ.get("FREEROUTING_JAR"):
        candidates.append(os.environ["FREEROUTING_JAR"])
    candidates += sorted(glob.glob(os.path.join(PROJECT_ROOT, "freerouting*.jar")), reverse=True)
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    raise PipelineError(f"freerouting-*.jar nicht gefunden (gesucht in {PROJECT_ROOT}).")


def run_freerouting(jar: str, dsn_path: str, ses_path: str, passes: int,
                    log: Callable[[str], None] = print) -> bool:
    """Routet `dsn_path` headless nach `ses_path`. Gibt True zurück, wenn alle Verbindungen geroutet wurden."""
    env = {k: v for k, v in os.environ.items() if not (k in SNAP_GTK_VARS and "/snap/" in v)}
    cmd = ["java", "-jar", jar, "-de", dsn_path, "-do", ses_path, "-mp", str(passes), "--gui.enabled=false"]
    if os.path.exists(ses_path):
        os.remove(ses_path)

    summary = None
    try:
        with subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
            for line in proc.stdout:
                line = line.rstrip()
                if "Gtk-WARNING" in line:
                    continue
                if "stage completed" in line or "WARN" in line or "ERROR" in line or "Exception" in line:
                    log(line)
                if "Auto-routing stage completed" in line:
                    summary = line
    except FileNotFoundError:
        raise PipelineError("Java nicht gefunden – bitte installieren (sudo apt install default-jre).")
    if proc.returncode != 0 or not os.path.exists(ses_path):
        raise PipelineError(f"FreeRouting ist fehlgeschlagen (Exit-Code {proc.returncode}), keine .ses erzeugt.")
    return summary is not None and "(0 unrouted" in summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Platziert die Bauteile einer KiCad-DSN per EA und routet sie mit FreeRouting.")
    parser.add_argument("dsn", help="aus KiCad exportierte Specctra-DSN-Datei")
    parser.add_argument("-o", "--output", help="Ziel-.ses (Standard: <dsn>.ses)")
    parser.add_argument("-g", "--generations", type=int, default=500, help="maximale Generationen (Standard: 500)")
    parser.add_argument("-p", "--patience", type=int, default=DEFAULT_PATIENCE,
                        help=f"Abbruch, wenn sich das beste Layout so viele Generationen nicht ändert (Standard: {DEFAULT_PATIENCE})")
    parser.add_argument("--margin", type=float, default=config.COURTYARD_MARGIN * NM_TO_MM,
                        help=f"Abstand um jedes Bauteil in mm (Standard: {config.COURTYARD_MARGIN * NM_TO_MM})")
    parser.add_argument("--passes", type=int, default=DEFAULT_PASSES, help=f"max. FreeRouting-Durchläufe (Standard: {DEFAULT_PASSES})")
    parser.add_argument("--freerouting", help="Pfad zu freerouting-*.jar (Standard: im Projektordner suchen)")
    parser.add_argument("--seed", type=int, help="Zufalls-Seed für reproduzierbare Läufe")
    parser.add_argument("--log-every", type=int, default=10, help="Fortschritt alle N Generationen ausgeben")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
    stem = os.path.splitext(args.dsn)[0]
    optimized_dsn = stem + "_ea.dsn"
    ses_path = args.output or stem + ".ses"
    try:
        jar = find_freerouting_jar(args.freerouting)
    except PipelineError as e:
        sys.exit(f"FEHLER: {e} Pfad mit --freerouting angeben.")

    print(f"[1/4] Lese {args.dsn}")
    design = dsn_parser.load_dsn(args.dsn)
    config.set_board_size(*design.board_size)
    config.COURTYARD_MARGIN = int(args.margin * MM_TO_NM)
    locked = [ref for ref, c in design.components.items() if c.locked]
    print(f"  Platine {config.BOARD_WIDTH_MM:.1f} × {config.BOARD_HEIGHT_MM:.1f} mm, "
          f"{len(design.footprints)} Bauteile ({len(locked)} gesperrt{': ' + ', '.join(locked) if locked else ''}), "
          f"{len(design.netlist)} Netze")
    for warning in design.warnings:
        print(f"  WARNUNG: {warning}")

    print(f"[2/4] Evolutionärer Algorithmus (Population {POP_SIZE}, max. {args.generations} Generationen)")
    start = time.time()
    best = run_ea(design.footprints, design.netlist, design.initial_genome,
                  args.generations, args.patience, args.log_every)
    overlap = compute_overlap_penalty(best)
    print(f"  Dauer {time.time() - start:.1f} s | Trace-Score {compute_tracelength_fitness(best, design.netlist) * NM_TO_MM:.1f} mm "
          f"(Länge + 50 mm je Kreuzung) | "
          f"Bounding-Box {compute_bbox_area(best) * NM_TO_MM ** 2:.0f} mm² | Overlap {overlap:.2f}")
    if overlap > 0:
        sys.exit("FEHLER: Das beste Layout hat noch Überlappungen – nicht geroutet.\n"
                 "        Mehr Generationen (-g) oder kleineren Abstand (--margin) versuchen.")

    print(f"[3/4] Schreibe optimierte Platzierung → {optimized_dsn}")
    dsn_parser.write_dsn(design, best, optimized_dsn)

    print(f"[4/4] FreeRouting ({os.path.basename(jar)}, max. {args.passes} Durchläufe)")
    try:
        complete = run_freerouting(jar, optimized_dsn, ses_path, args.passes, log=lambda line: print("  " + line))
    except PipelineError as e:
        sys.exit(f"FEHLER: {e}")
    if not complete:
        print("  WARNUNG: Nicht alle Verbindungen konnten geroutet werden – siehe Zeile oben.")

    print(f"\nFertig: {ses_path}")
    print("In KiCad: Datei → Importieren → Specctra Session … (übernimmt Platzierung und Leiterbahnen)")


if __name__ == "__main__":
    main()
