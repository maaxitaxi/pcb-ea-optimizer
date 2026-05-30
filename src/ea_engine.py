# ea_engine.py
import random
import copy
import math
from typing import List, Dict, Tuple
from config import BOARD_W, BOARD_H, POP_SIZE, TOURNAMENT_K, CROSSOVER_RATE, MUTATION_RATE, SIGMA_XY, ROT_MUTATE_P, VALID_ROTATIONS, MM_TO_NM
from models import Footprint, PlacedComponent, Genome, Pin

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

def compute_fitness(genome: Genome, netlist: Dict[str, List[Tuple[str, str]]]) -> float:
    for comp in genome:
        if not comp.is_within_board(): return float('inf')
    for i in range(len(genome)):
        for j in range(i + 1, len(genome)):
            if genome[i].overlaps(genome[j]): return float('inf')

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

def tournament_selection(population: List[Genome], fitness_vals: List[float], k: int) -> Genome:
    candidates = random.sample(range(len(population)), k)
    best = min(candidates, key=lambda idx: fitness_vals[idx])
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

def evolve_one_generation(population: List[Genome], fitness_vals: List[float], netlist: Dict[str, List[Tuple[str, str]]]) -> Tuple[List[Genome], List[float]]:
    new_population: List[Genome] = []
    best_idx = min(range(len(fitness_vals)), key=lambda i: fitness_vals[i])
    new_population.append(copy.deepcopy(population[best_idx])) # Elitismus

    while len(new_population) < POP_SIZE:
        p1 = tournament_selection(population, fitness_vals, TOURNAMENT_K)
        p2 = tournament_selection(population, fitness_vals, TOURNAMENT_K)
        c1, c2 = uniform_crossover(p1, p2) if random.random() < CROSSOVER_RATE else (copy.deepcopy(p1), copy.deepcopy(p2))
        new_population.append(mutate(c1, SIGMA_XY, MUTATION_RATE))
        if len(new_population) < POP_SIZE:
            new_population.append(mutate(c2, SIGMA_XY, MUTATION_RATE))

    return new_population, [compute_fitness(g, netlist) for g in new_population]