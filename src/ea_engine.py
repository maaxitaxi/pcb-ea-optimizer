# ea_engine.py
import random
import copy
import math
from typing import List, Dict, Tuple
from config import BOARD_W, BOARD_H, POP_SIZE, TOURNAMENT_K, CROSSOVER_RATE, MUTATION_RATE, SIGMA_XY, ROT_MUTATE_P, VALID_ROTATIONS, MM_TO_NM
from models import Footprint, PlacedComponent, Genome, Pin

# Weights for the combined fitness score (must sum to 1.0)
TRACE_WEIGHT   = 0.4
OVERLAP_WEIGHT = 1 - TRACE_WEIGHT

def create_scenario() -> Tuple[List[Footprint], Dict[str, List[Tuple[str, str]]]]:
    nm = MM_TO_NM
    ic1 = Footprint(ref="IC1", width=30*nm, height=30*nm, color="#FF5555",
        pins=[
            Pin("P1", -15*nm, 0, "VCC"), Pin("P2", -15*nm, -8*nm, "GND"),
            Pin("P3", -15*nm, 8*nm, "NET_SPI_CLK"), Pin("P4", 15*nm, 0, "NET_SPI_MOSI"),
            Pin("P5", 15*nm, -8*nm, "NET_SPI_MISO"), Pin("P6", 15*nm, 8*nm, "NET_UART_TX"),
            Pin("P7", 0, -15*nm, "NET_UART_RX"), Pin("P8", 0, 15*nm, "GND"),
        ])
    u1 = Footprint(ref="U1", width=15*nm, height=10*nm, color="#BB55BB",
        pins=[Pin("IN", -7*nm, 0, "VIN"), Pin("OUT", 7*nm, 0, "VCC"), Pin("GND", 0, 5*nm, "GND"), Pin("EN", 0, -5*nm, "NET_EN")])
    u2 = Footprint(ref="U2", width=15*nm, height=10*nm, color="#7755CC",
        pins=[Pin("IN", -7*nm, 0, "VIN"), Pin("OUT", 7*nm, 0, "NET_3V3"), Pin("GND", 0, 5*nm, "GND"), Pin("EN", 0, -5*nm, "NET_EN")])
    r1 = Footprint(ref="R1", width=8*nm, height=3*nm, color="#55BB55", pins=[Pin("A", -4*nm, 0, "VCC"), Pin("B", 4*nm, 0, "NET_LED_A")])
    r2 = Footprint(ref="R2", width=8*nm, height=3*nm, color="#22AA99", pins=[Pin("A", -4*nm, 0, "NET_SPI_CLK"), Pin("B", 4*nm, 0, "NET_SPI_MOSI")])
    r3 = Footprint(ref="R3", width=8*nm, height=3*nm, color="#FF9922", pins=[Pin("A", -4*nm, 0, "NET_UART_TX"), Pin("B", 4*nm, 0, "NET_UART_RX")])
    r4 = Footprint(ref="R4", width=8*nm, height=3*nm, color="#FF5522", pins=[Pin("A", -4*nm, 0, "NET_3V3"), Pin("B", 4*nm, 0, "NET_EN")])
    d1 = Footprint(ref="D1", width=5*nm, height=5*nm, color="#FFCC22", pins=[Pin("A", -2*nm, 0, "NET_LED_A"), Pin("K", 2*nm, 0, "GND")])

    footprints = [ic1, u1, u2, r1, r2, r3, r4, d1]
    netlist = {}
    for fp in footprints:
        for pin in fp.pins:
            netlist.setdefault(pin.net_id, []).append((fp.ref, pin.pin_id))
    return footprints, netlist


def random_placement(footprints: List[Footprint]) -> Genome:
    placed: Genome = []
    for fp in footprints:
        for _ in range(500):
            rot = random.choice(VALID_ROTATIONS)
            ew, eh = (fp.height, fp.width) if rot in (90, 270) else (fp.width, fp.height)
            half_w, half_h = ew // 2, eh // 2
            x = random.randint(half_w, BOARD_W - half_w)
            y = random.randint(half_h, BOARD_H - half_h)
            candidate = PlacedComponent(footprint=fp, x=x, y=y, rot=rot)
            if not any(candidate.overlaps(other) for other in placed):
                placed.append(candidate)
                break
        else:
            placed.append(PlacedComponent(footprint=fp, x=BOARD_W//2, y=BOARD_H//2, rot=0))
    return placed


# ---------------------------------------------------------------------------
# Raw fitness components (both return lower = worse quality)
# ---------------------------------------------------------------------------

def compute_tracelength_fitness(genome: Genome, netlist: Dict[str, List[Tuple[str, str]]]) -> float:
    """Returns total estimated wire length in board units, or inf if any component is off-board."""
    for comp in genome:
        if not comp.is_within_board():
            return float('inf')

    pin_positions = {}
    for comp in genome:
        for pin_id, _, abs_x, abs_y in comp.get_pin_positions():
            pin_positions[(comp.footprint.ref, pin_id)] = (abs_x, abs_y)

    total_length = 0.0
    for net_id, connections in netlist.items():
        net_pins = [pin_positions[key] for ref, pid in connections if (key := (ref, pid)) in pin_positions]
        for i in range(len(net_pins)):
            for j in range(i + 1, len(net_pins)):
                total_length += math.hypot(net_pins[i][0] - net_pins[j][0], net_pins[i][1] - net_pins[j][1])
    return total_length


def compute_overlap_penalty(genome: Genome) -> float:
    """Returns sum of pairwise overlap ratios. 0.0 = no overlaps, higher = worse."""
    total = 0.0
    for i in range(len(genome)):
        for j in range(i + 1, len(genome)):
            total += genome[i].overlaps(genome[j])
    return total


# ---------------------------------------------------------------------------
# Combined normalised fitness over the whole population
# ---------------------------------------------------------------------------

def normalize_population_fitness(
    population: List[Genome],
    netlist: Dict[str, List[Tuple[str, str]]],
    trace_weight: float = TRACE_WEIGHT,
    overlap_weight: float = OVERLAP_WEIGHT,
) -> List[float]:
    """
    Computes a combined, normalised fitness score in [0, 1] for every genome.
    Higher score = better individual.
    Off-board / fully invalid genomes receive 0.0.
    """
    traces   = [compute_tracelength_fitness(g, netlist) for g in population]
    overlaps = [compute_overlap_penalty(g) for g in population]

    # --- Normalise trace lengths (ignore inf) ---
    finite_traces = [t for t in traces if t != float('inf')]
    if finite_traces:
        t_min, t_max = min(finite_traces), max(finite_traces)
        if t_max != t_min:
            norm_traces = [
                0.0 if t == float('inf')
                else 1.0 - (t - t_min) / (t_max - t_min)
                for t in traces
            ]
        else:
            # All finite traces identical — give full score to valid, 0 to invalid
            norm_traces = [0.0 if t == float('inf') else 1.0 for t in traces]
    else:
        # Every genome is off-board
        norm_traces = [0.0] * len(traces)

    # --- Normalise overlap penalties ---
    o_max = max(overlaps) if overlaps else 0.0
    if o_max > 0.0:
        norm_overlaps = [1.0 - (o / o_max) for o in overlaps]
    else:
        norm_overlaps = [1.0] * len(overlaps)  # no overlaps anywhere → perfect

    # --- Weighted sum → combined score in [0, 1] ---
    combined = [
        trace_weight * nt + overlap_weight * no
        for nt, no in zip(norm_traces, norm_overlaps)
    ]
    return combined


# ---------------------------------------------------------------------------
# GA operators
# ---------------------------------------------------------------------------

def tournament_selection(population: List[Genome], fitness_vals: List[float], k: int) -> Genome:
    """Higher fitness_vals = better, so we take the max."""
    candidates = random.sample(range(len(population)), k)
    best = max(candidates, key=lambda idx: fitness_vals[idx])
    return copy.deepcopy(population[best])


def uniform_crossover(parent1: Genome, parent2: Genome) -> Tuple[Genome, Genome]:
    child1, child2 = [], []
    for c1, c2 in zip(parent1, parent2):
        if random.random() < 0.5:
            child1.append(copy.deepcopy(c1)); child2.append(copy.deepcopy(c2))
        else:
            child1.append(copy.deepcopy(c2)); child2.append(copy.deepcopy(c1))
    return child1, child2


def mutate(genome: Genome, sigma: float, mutation_rate: float) -> Genome:
    for comp in genome:
        if random.random() < mutation_rate:
            comp.x += int(random.gauss(0, sigma))
            comp.y += int(random.gauss(0, sigma))
            if random.random() < ROT_MUTATE_P:
                comp.rot = random.choice(VALID_ROTATIONS)
            w, h = comp._rotated_dims()
            comp.x = max(w // 2, min(BOARD_W - w // 2, comp.x))
            comp.y = max(h // 2, min(BOARD_H - h // 2, comp.y))
    return genome


def evolve_one_generation(
    population: List[Genome],
    fitness_vals: List[float],
    netlist: Dict[str, List[Tuple[str, str]]],
) -> Tuple[List[Genome], List[float]]:
    new_population: List[Genome] = []

    # Elitism — carry over the best individual unchanged (higher = better now)
    best_idx = max(range(len(fitness_vals)), key=lambda i: fitness_vals[i])
    new_population.append(copy.deepcopy(population[best_idx]))

    while len(new_population) < POP_SIZE:
        p1 = tournament_selection(population, fitness_vals, TOURNAMENT_K)
        p2 = tournament_selection(population, fitness_vals, TOURNAMENT_K)
        if random.random() < CROSSOVER_RATE:
            c1, c2 = uniform_crossover(p1, p2)
        else:
            c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
        new_population.append(mutate(c1, SIGMA_XY, MUTATION_RATE))
        if len(new_population) < POP_SIZE:
            new_population.append(mutate(c2, SIGMA_XY, MUTATION_RATE))

    new_fitness = normalize_population_fitness(new_population, netlist)
    return new_population, new_fitness