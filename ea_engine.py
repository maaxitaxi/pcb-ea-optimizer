# ea_engine.py
import random
import copy
import math
import csv
from typing import List, Dict, Tuple, Any, Optional

from config import (
    BOARD_W,
    BOARD_H,
    POP_SIZE,
    TOURNAMENT_K,
    CROSSOVER_RATE,
    MUTATION_RATE,
    SIGMA_XY,
    ROT_MUTATE_P,
    VALID_ROTATIONS,
    MM_TO_NM,
    NM_TO_MM,
    USE_DYNAMIC_SCHEDULE,
    SCHEDULE_GENERATIONS,
    START_OVERLAP_WEIGHT,
    END_OVERLAP_WEIGHT,
    START_MUTATION_RATE,
    END_MUTATION_RATE,
    START_SIGMA_XY_MM,
    END_SIGMA_XY_MM,
    DEFAULT_HISTORY_CSV,
    DEFAULT_HISTORY_PLOT,
)

from models import Footprint, PlacedComponent, Genome, Pin


# Statische Fallback-Gewichte
TRACE_WEIGHT = 0.4
OVERLAP_WEIGHT = 1 - TRACE_WEIGHT


def create_scenario() -> Tuple[List[Footprint], Dict[str, List[Tuple[str, str]]]]:
    nm = MM_TO_NM

    ic1 = Footprint(
        ref="IC1",
        width=30 * nm,
        height=30 * nm,
        color="#FF5555",
        pins=[
            Pin("P1", -15 * nm, 0, "VCC"),
            Pin("P2", -15 * nm, -8 * nm, "GND"),
            Pin("P3", -15 * nm, 8 * nm, "NET_SPI_CLK"),
            Pin("P4", 15 * nm, 0, "NET_SPI_MOSI"),
            Pin("P5", 15 * nm, -8 * nm, "NET_SPI_MISO"),
            Pin("P6", 15 * nm, 8 * nm, "NET_UART_TX"),
            Pin("P7", 0, -15 * nm, "NET_UART_RX"),
            Pin("P8", 0, 15 * nm, "GND"),
        ],
    )

    u1 = Footprint(
        ref="U1",
        width=15 * nm,
        height=10 * nm,
        color="#BB55BB",
        pins=[
            Pin("IN", -7 * nm, 0, "VIN"),
            Pin("OUT", 7 * nm, 0, "VCC"),
            Pin("GND", 0, 5 * nm, "GND"),
            Pin("EN", 0, -5 * nm, "NET_EN"),
        ],
    )

    u2 = Footprint(
        ref="U2",
        width=15 * nm,
        height=10 * nm,
        color="#7755CC",
        pins=[
            Pin("IN", -7 * nm, 0, "VIN"),
            Pin("OUT", 7 * nm, 0, "NET_3V3"),
            Pin("GND", 0, 5 * nm, "GND"),
            Pin("EN", 0, -5 * nm, "NET_EN"),
        ],
    )

    r1 = Footprint(
        ref="R1",
        width=8 * nm,
        height=3 * nm,
        color="#55BB55",
        pins=[
            Pin("A", -4 * nm, 0, "VCC"),
            Pin("B", 4 * nm, 0, "NET_LED_A"),
        ],
    )

    r2 = Footprint(
        ref="R2",
        width=8 * nm,
        height=3 * nm,
        color="#22AA99",
        pins=[
            Pin("A", -4 * nm, 0, "NET_SPI_CLK"),
            Pin("B", 4 * nm, 0, "NET_SPI_MOSI"),
        ],
    )

    r3 = Footprint(
        ref="R3",
        width=8 * nm,
        height=3 * nm,
        color="#FF9922",
        pins=[
            Pin("A", -4 * nm, 0, "NET_UART_TX"),
            Pin("B", 4 * nm, 0, "NET_UART_RX"),
        ],
    )

    r4 = Footprint(
        ref="R4",
        width=8 * nm,
        height=3 * nm,
        color="#FF5522",
        pins=[
            Pin("A", -4 * nm, 0, "NET_3V3"),
            Pin("B", 4 * nm, 0, "NET_EN"),
        ],
    )

    d1 = Footprint(
        ref="D1",
        width=5 * nm,
        height=5 * nm,
        color="#FFCC22",
        pins=[
            Pin("A", -2 * nm, 0, "NET_LED_A"),
            Pin("K", 2 * nm, 0, "GND"),
        ],
    )

    footprints = [ic1, u1, u2, r1, r2, r3, r4, d1]

    netlist: Dict[str, List[Tuple[str, str]]] = {}

    for fp in footprints:
        for pin in fp.pins:
            netlist.setdefault(pin.net_id, []).append((fp.ref, pin.pin_id))

    return footprints, netlist


def random_placement(
    footprints: List[Footprint],
    avoid_overlap: bool = False,
    max_attempts: int = 500,
) -> Genome:
    """
    Erzeugt ein zufälliges Layout.

    avoid_overlap=False ist bewusst der neue Standard:
    Dadurch werden am Anfang auch überlappende Layouts zugelassen.
    Das vergrößert den Suchraum in der frühen Explorationsphase.
    Overlaps werden anschließend über die Fitness bestraft.
    """
    placed: Genome = []

    for fp in footprints:
        placed_component = None

        attempts = max_attempts if avoid_overlap else 1

        for _ in range(attempts):
            rot = random.choice(VALID_ROTATIONS)

            if rot in (90, 270):
                ew, eh = fp.height, fp.width
            else:
                ew, eh = fp.width, fp.height

            half_w, half_h = ew // 2, eh // 2

            x = random.randint(half_w, BOARD_W - half_w)
            y = random.randint(half_h, BOARD_H - half_h)

            candidate = PlacedComponent(
                footprint=fp,
                x=x,
                y=y,
                rot=rot,
            )

            if not avoid_overlap or not any(candidate.overlaps(other) for other in placed):
                placed_component = candidate
                break

        if placed_component is None:
            placed_component = PlacedComponent(
                footprint=fp,
                x=BOARD_W // 2,
                y=BOARD_H // 2,
                rot=0,
            )

        placed.append(placed_component)

    return placed


# ---------------------------------------------------------------------------
# Dynamische Schedule: großer Suchraum am Anfang, stärkere Constraints am Ende
# ---------------------------------------------------------------------------

def _lerp(start: float, end: float, progress: float) -> float:
    return start + progress * (end - start)


def get_scheduled_params(
    generation: int,
    schedule_generations: int = SCHEDULE_GENERATIONS,
) -> Dict[str, float]:
    """
    Dynamische Parameter:

    progress = 0.0:
        starke Exploration
        hohe Mutation
        große Schrittweite
        geringere Overlap-Strafe

    progress = 1.0:
        stärkere Exploitation
        geringere Mutation
        kleinere Schrittweite
        hohe Overlap-Strafe
    """
    if schedule_generations <= 0:
        progress = 1.0
    else:
        progress = max(0.0, min(1.0, generation / schedule_generations))

    overlap_weight = _lerp(
        START_OVERLAP_WEIGHT,
        END_OVERLAP_WEIGHT,
        progress,
    )

    trace_weight = 1.0 - overlap_weight

    mutation_rate = _lerp(
        START_MUTATION_RATE,
        END_MUTATION_RATE,
        progress,
    )

    sigma_xy_mm = _lerp(
        START_SIGMA_XY_MM,
        END_SIGMA_XY_MM,
        progress,
    )

    return {
        "progress": progress,
        "trace_weight": trace_weight,
        "overlap_weight": overlap_weight,
        "mutation_rate": mutation_rate,
        "sigma_xy": sigma_xy_mm * MM_TO_NM,
        "sigma_xy_mm": sigma_xy_mm,
    }


# ---------------------------------------------------------------------------
# Fitness-Komponenten
# ---------------------------------------------------------------------------

def compute_tracelength_fitness(
    genome: Genome,
    netlist: Dict[str, List[Tuple[str, str]]],
) -> float:
    """
    Gibt die geschätzte gesamte Verbindungslänge zurück.
    Kleinere Werte sind besser.

    Wenn ein Bauteil außerhalb der Platine liegt, wird inf zurückgegeben.
    """
    for comp in genome:
        if not comp.is_within_board():
            return float("inf")

    pin_positions = {}

    for comp in genome:
        for pin_id, _, abs_x, abs_y in comp.get_pin_positions():
            pin_positions[(comp.footprint.ref, pin_id)] = (abs_x, abs_y)

    total_length = 0.0

    for net_id, connections in netlist.items():
        net_pins = [
            pin_positions[key]
            for ref, pid in connections
            if (key := (ref, pid)) in pin_positions
        ]

        for i in range(len(net_pins)):
            for j in range(i + 1, len(net_pins)):
                total_length += math.hypot(
                    net_pins[i][0] - net_pins[j][0],
                    net_pins[i][1] - net_pins[j][1],
                )

    return total_length


def compute_overlap_penalty(genome: Genome) -> float:
    """
    Gibt die Summe aller paarweisen Überlappungen zurück.

    0.0 = keine Überlappung
    größerer Wert = schlechter
    """
    total = 0.0

    for i in range(len(genome)):
        for j in range(i + 1, len(genome)):
            total += genome[i].overlaps(genome[j])

    return total


def is_feasible(genome: Genome) -> bool:
    """
    Feasible bedeutet hier:

    - alle Bauteile liegen innerhalb der Platine
    - keine Bauteile überlappen sich
    """
    return all(comp.is_within_board() for comp in genome) and compute_overlap_penalty(genome) == 0.0


def evaluate_population(
    population: List[Genome],
    netlist: Dict[str, List[Tuple[str, str]]],
    trace_weight: float = TRACE_WEIGHT,
    overlap_weight: float = OVERLAP_WEIGHT,
) -> Dict[str, List[float]]:
    """
    Berechnet getrennte und kombinierte Fitnesswerte.

    Raw-Werte:
        trace_raw:
            Leiterbahnlänge, kleiner = besser
        overlap_raw:
            Überlappungsstrafe, kleiner = besser

    Scores:
        trace_score:
            normalisiert auf [0, 1], größer = besser
        overlap_score:
            normalisiert auf [0, 1], größer = besser
        combined:
            gewichtete Gesamtfitness, größer = besser
    """
    weight_sum = trace_weight + overlap_weight

    if weight_sum <= 0:
        trace_weight = TRACE_WEIGHT
        overlap_weight = OVERLAP_WEIGHT
        weight_sum = trace_weight + overlap_weight

    trace_weight /= weight_sum
    overlap_weight /= weight_sum

    traces = [
        compute_tracelength_fitness(g, netlist)
        for g in population
    ]

    overlaps = [
        compute_overlap_penalty(g)
        for g in population
    ]

    # Trace normalisieren
    finite_traces = [
        t for t in traces
        if math.isfinite(t)
    ]

    if finite_traces:
        t_min = min(finite_traces)
        t_max = max(finite_traces)

        if t_max != t_min:
            trace_scores = [
                0.0 if not math.isfinite(t)
                else 1.0 - (t - t_min) / (t_max - t_min)
                for t in traces
            ]
        else:
            trace_scores = [
                0.0 if not math.isfinite(t)
                else 1.0
                for t in traces
            ]
    else:
        trace_scores = [0.0] * len(population)

    # Overlap normalisieren
    o_max = max(overlaps) if overlaps else 0.0

    if o_max > 0.0:
        overlap_scores = [
            1.0 - (o / o_max)
            for o in overlaps
        ]
    else:
        overlap_scores = [1.0] * len(population)

    combined = [
        trace_weight * ts + overlap_weight * os
        for ts, os in zip(trace_scores, overlap_scores)
    ]

    return {
        "trace_raw": traces,
        "overlap_raw": overlaps,
        "trace_score": trace_scores,
        "overlap_score": overlap_scores,
        "combined": combined,
    }


def normalize_population_fitness(
    population: List[Genome],
    netlist: Dict[str, List[Tuple[str, str]]],
    trace_weight: float = TRACE_WEIGHT,
    overlap_weight: float = OVERLAP_WEIGHT,
) -> List[float]:
    """
    Kompatibilitätsfunktion.

    Gibt nur die kombinierte Fitness zurück.
    """
    return evaluate_population(
        population,
        netlist,
        trace_weight,
        overlap_weight,
    )["combined"]


# ---------------------------------------------------------------------------
# Diversitätsmessung
# ---------------------------------------------------------------------------

def genome_distance(g1: Genome, g2: Genome) -> float:
    """
    Genotypische Distanz zweier Layouts in Millimetern.

    Berücksichtigt:
    - Positionen gleicher Komponenten
    - Rotationsunterschiede
    """
    if not g1 or not g2:
        return 0.0

    by_ref_2 = {
        c.footprint.ref: c
        for c in g2
    }

    total_nm = 0.0
    count = 0

    for c1 in g1:
        c2 = by_ref_2.get(c1.footprint.ref)

        if c2 is None:
            continue

        pos_dist = math.hypot(
            c1.x - c2.x,
            c1.y - c2.y,
        )

        rot_steps = min(
            (c1.rot - c2.rot) % 360,
            (c2.rot - c1.rot) % 360,
        ) / 90

        # Eine 90°-Änderung zählt hier wie ungefähr 5 mm.
        rot_penalty_nm = rot_steps * 5 * MM_TO_NM

        total_nm += pos_dist + rot_penalty_nm
        count += 1

    if count == 0:
        return 0.0

    return (total_nm / count) * NM_TO_MM


def population_diversity(population: List[Genome]) -> float:
    """
    Durchschnittliche paarweise genotypische Distanz einer Population in mm.
    """
    if len(population) < 2:
        return 0.0

    total = 0.0
    pairs = 0

    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            total += genome_distance(
                population[i],
                population[j],
            )
            pairs += 1

    if pairs == 0:
        return 0.0

    return total / pairs


# ---------------------------------------------------------------------------
# Logging / Run-History
# ---------------------------------------------------------------------------

def _avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _avg_finite(values: List[float]) -> float:
    finite = [
        v for v in values
        if math.isfinite(v)
    ]

    return _avg(finite)


def collect_generation_stats(
    population: List[Genome],
    fitness_vals: List[float],
    netlist: Dict[str, List[Tuple[str, str]]],
    generation: int,
    trace_weight: float = TRACE_WEIGHT,
    overlap_weight: float = OVERLAP_WEIGHT,
    mutation_rate: float = MUTATION_RATE,
    sigma_xy: float = SIGMA_XY,
) -> Dict[str, Any]:
    """
    Sammelt alle Kennzahlen aus der Checkliste für eine Generation.
    """
    details = evaluate_population(
        population,
        netlist,
        trace_weight,
        overlap_weight,
    )

    feasible_flags = [
        is_feasible(g)
        for g in population
    ]

    feasible_population = [
        g for g, ok in zip(population, feasible_flags)
        if ok
    ]

    infeasible_population = [
        g for g, ok in zip(population, feasible_flags)
        if not ok
    ]

    combined = fitness_vals if fitness_vals else details["combined"]

    if combined:
        best_idx = max(
            range(len(combined)),
            key=lambda i: combined[i],
        )
    else:
        best_idx = 0

    trace_raw_mm = [
        v * NM_TO_MM if math.isfinite(v) else float("inf")
        for v in details["trace_raw"]
    ]

    sigma_xy_mm = sigma_xy * NM_TO_MM

    feasible_count = sum(feasible_flags)
    infeasible_count = len(population) - feasible_count

    return {
        "generation": generation,
        "population_size": len(population),

        "best_total_fitness": combined[best_idx] if combined else 0.0,
        "avg_total_fitness": _avg(combined),

        "best_trace_score": details["trace_score"][best_idx] if combined else 0.0,
        "avg_trace_score": _avg(details["trace_score"]),

        "best_overlap_score": details["overlap_score"][best_idx] if combined else 0.0,
        "avg_overlap_score": _avg(details["overlap_score"]),

        "best_trace_length_mm": trace_raw_mm[best_idx] if combined else 0.0,
        "avg_trace_length_mm": _avg_finite(trace_raw_mm),

        "best_overlap_penalty": details["overlap_raw"][best_idx] if combined else 0.0,
        "avg_overlap_penalty": _avg(details["overlap_raw"]),
        "min_overlap_penalty": min(details["overlap_raw"]) if details["overlap_raw"] else 0.0,

        "feasible_count": feasible_count,
        "infeasible_count": infeasible_count,
        "feasible_ratio": feasible_count / len(population) if population else 0.0,

        "diversity_all_mm": population_diversity(population),
        "diversity_feasible_mm": population_diversity(feasible_population),
        "diversity_infeasible_mm": population_diversity(infeasible_population),

        "trace_weight": trace_weight,
        "overlap_weight": overlap_weight,
        "mutation_rate": mutation_rate,
        "sigma_xy_mm": sigma_xy_mm,
    }


def save_history_csv(
    history: List[Dict[str, Any]],
    filename: str = DEFAULT_HISTORY_CSV,
) -> str:
    """
    Speichert die Run-History als CSV.
    """
    if not history:
        return filename

    fieldnames = list(history[0].keys())

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(history)

    return filename


def save_history_plot(
    history: List[Dict[str, Any]],
    filename: str = DEFAULT_HISTORY_PLOT,
) -> str:
    """
    Speichert Diagramme für:

    - Gesamtfitness
    - Overlap-Penalty
    - Feasible-Ratio
    - Diversität
    """
    if not history:
        return filename

    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    generations = [
        row["generation"]
        for row in history
    ]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 9),
        sharex=True,
    )

    axes[0].plot(
        generations,
        [row["best_total_fitness"] for row in history],
        label="Beste Gesamtfitness",
    )

    axes[0].plot(
        generations,
        [row["avg_total_fitness"] for row in history],
        label="Durchschnittliche Gesamtfitness",
    )

    axes[0].set_ylabel("Fitness")
    axes[0].legend()

    axes[1].plot(
        generations,
        [row["avg_overlap_penalty"] for row in history],
        label="Ø Overlap-Penalty",
    )

    axes[1].plot(
        generations,
        [row["min_overlap_penalty"] for row in history],
        label="Min. Overlap-Penalty",
    )

    axes[1].set_ylabel("Overlap")
    axes[1].legend()

    axes[2].plot(
        generations,
        [row["feasible_ratio"] for row in history],
        label="Feasible Ratio",
    )

    axes[2].plot(
        generations,
        [row["diversity_all_mm"] for row in history],
        label="Diversität gesamt [mm]",
    )

    axes[2].set_xlabel("Generation")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.close(fig)

    return filename


# ---------------------------------------------------------------------------
# GA-Operatoren
# ---------------------------------------------------------------------------

def tournament_selection(
    population: List[Genome],
    fitness_vals: List[float],
    k: int,
) -> Genome:
    """
    Tournament Selection.

    Höhere Fitness ist besser.
    """
    k = min(k, len(population))

    candidates = random.sample(
        range(len(population)),
        k,
    )

    best = max(
        candidates,
        key=lambda idx: fitness_vals[idx],
    )

    return copy.deepcopy(population[best])


def uniform_crossover(
    parent1: Genome,
    parent2: Genome,
) -> Tuple[Genome, Genome]:
    """
    Uniform Crossover.

    Für jedes Bauteil wird zufällig entschieden,
    ob Kind 1 die Platzierung von Parent 1 oder Parent 2 bekommt.
    """
    child1 = []
    child2 = []

    for c1, c2 in zip(parent1, parent2):
        if random.random() < 0.5:
            child1.append(copy.deepcopy(c1))
            child2.append(copy.deepcopy(c2))
        else:
            child1.append(copy.deepcopy(c2))
            child2.append(copy.deepcopy(c1))

    return child1, child2


def mutate(
    genome: Genome,
    sigma: float,
    mutation_rate: float,
    rot_mutate_p: float = ROT_MUTATE_P,
) -> Genome:
    """
    Mutation:

    - verschiebt Bauteile zufällig
    - ändert teilweise die Rotation
    - begrenzt Bauteile anschließend wieder auf die Platine
    """
    for comp in genome:
        if random.random() < mutation_rate:
            comp.x += int(random.gauss(0, sigma))
            comp.y += int(random.gauss(0, sigma))

            if random.random() < rot_mutate_p:
                comp.rot = random.choice(VALID_ROTATIONS)

            w, h = comp._rotated_dims()

            comp.x = max(
                w // 2,
                min(BOARD_W - w // 2, comp.x),
            )

            comp.y = max(
                h // 2,
                min(BOARD_H - h // 2, comp.y),
            )

    return genome


def evolve_one_generation(
    population: List[Genome],
    fitness_vals: List[float],
    netlist: Dict[str, List[Tuple[str, str]]],
    generation: int = 0,
    use_dynamic_schedule: bool = USE_DYNAMIC_SCHEDULE,
    trace_weight: Optional[float] = None,
    overlap_weight: Optional[float] = None,
    mutation_rate: Optional[float] = None,
    sigma_xy: Optional[float] = None,
    pop_size: int = POP_SIZE,
) -> Tuple[List[Genome], List[float]]:
    """
    Erzeugt eine neue Generation.

    Wenn use_dynamic_schedule=True ist:
    - am Anfang größere Exploration
    - später härtere Overlap-Bestrafung und kleinere Mutationsschritte
    """
    if use_dynamic_schedule:
        params = get_scheduled_params(generation)

        trace_weight = (
            params["trace_weight"]
            if trace_weight is None
            else trace_weight
        )

        overlap_weight = (
            params["overlap_weight"]
            if overlap_weight is None
            else overlap_weight
        )

        mutation_rate = (
            params["mutation_rate"]
            if mutation_rate is None
            else mutation_rate
        )

        sigma_xy = (
            params["sigma_xy"]
            if sigma_xy is None
            else sigma_xy
        )

    else:
        trace_weight = (
            TRACE_WEIGHT
            if trace_weight is None
            else trace_weight
        )

        overlap_weight = (
            OVERLAP_WEIGHT
            if overlap_weight is None
            else overlap_weight
        )

        mutation_rate = (
            MUTATION_RATE
            if mutation_rate is None
            else mutation_rate
        )

        sigma_xy = (
            SIGMA_XY
            if sigma_xy is None
            else sigma_xy
        )

    new_population: List[Genome] = []

    # Elitismus: bestes Individuum unverändert übernehmen
    best_idx = max(
        range(len(fitness_vals)),
        key=lambda i: fitness_vals[i],
    )

    new_population.append(
        copy.deepcopy(population[best_idx])
    )

    while len(new_population) < pop_size:
        p1 = tournament_selection(
            population,
            fitness_vals,
            TOURNAMENT_K,
        )

        p2 = tournament_selection(
            population,
            fitness_vals,
            TOURNAMENT_K,
        )

        if random.random() < CROSSOVER_RATE:
            c1, c2 = uniform_crossover(p1, p2)
        else:
            c1 = copy.deepcopy(p1)
            c2 = copy.deepcopy(p2)

        new_population.append(
            mutate(
                c1,
                sigma_xy,
                mutation_rate,
            )
        )

        if len(new_population) < pop_size:
            new_population.append(
                mutate(
                    c2,
                    sigma_xy,
                    mutation_rate,
                )
            )

    # Fitness der neuen Generation berechnen.
    # Bei dynamischer Schedule schon mit den Gewichten der nächsten Generation.
    if use_dynamic_schedule:
        next_params = get_scheduled_params(generation + 1)
        trace_weight = next_params["trace_weight"]
        overlap_weight = next_params["overlap_weight"]

    new_fitness = normalize_population_fitness(
        new_population,
        netlist,
        trace_weight,
        overlap_weight,
    )

    return new_population, new_fitness