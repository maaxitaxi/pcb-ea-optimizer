# ea_engine.py
import atexit
import math
import multiprocessing
import os
import random
from functools import lru_cache
from typing import List, Dict, Tuple
import config
from config import (
    POP_SIZE, TOURNAMENT_K, CROSSOVER_RATE, ROT_MUTATE_P,
    MUTATION_RATE, MUTATION_MIN_PARTS, SIGMA_START_FRAC, SIGMA_MIN, ANNEAL_GENS, MUTATION_TRIES, SWAP_P, COMPACT_P,
    VALID_ROTATIONS, MM_TO_NM, TRACE_WEIGHT, OVERLAP_WEIGHT, BBOX_WEIGHT,
    POWER_NET_KEYWORDS, POWER_NET_WEIGHT, DATA_NET_WEIGHT,
    PARALLEL_MIN_WORK, PARALLEL_WORKERS,
)
from models import Footprint, PlacedComponent, Genome, Pin


@lru_cache(maxsize=None)
def get_net_weight(net_id: str) -> float:
    upper = net_id.upper()
    if any(keyword in upper for keyword in POWER_NET_KEYWORDS):
        return POWER_NET_WEIGHT
    return DATA_NET_WEIGHT

def create_scenario() -> Tuple[List[Footprint], Dict[str, List[Tuple[str, str]]]]:
    nm = MM_TO_NM
    ic1 = Footprint(ref="IC1", width=30*nm, height=30*nm, color="#FF5555",
        pins=[
            Pin("P1", -13*nm, 0, "VCC"), Pin("P2", -13*nm, -8*nm, "GND"),
            Pin("P3", -13*nm, 8*nm, "NET_SPI_CLK"), Pin("P4", 13*nm, 0, "NET_SPI_MOSI"),
            Pin("P5", 13*nm, -8*nm, "NET_SPI_MISO"), Pin("P6", 13*nm, 8*nm, "NET_UART_TX"),
            Pin("P7", 0, -13*nm, "NET_UART_RX"), Pin("P8", 0, 13*nm, "GND"),
        ])
    u1 = Footprint(ref="U1", width=15*nm, height=10*nm, color="#BB55BB",
        pins=[Pin("IN", -6*nm, 0, "VIN"), Pin("OUT", 6*nm, 0, "VCC"), Pin("GND", 0, 4*nm, "GND"), Pin("EN", 0, -4*nm, "NET_EN")])
    u2 = Footprint(ref="U2", width=15*nm, height=10*nm, color="#7755CC",
        pins=[Pin("IN", -6*nm, 0, "VIN"), Pin("OUT", 6*nm, 0, "NET_3V3"), Pin("GND", 0, 4*nm, "GND"), Pin("EN", 0, -4*nm, "NET_EN")])
    r1 = Footprint(ref="R1", width=8*nm, height=3*nm, color="#55BB55", pins=[Pin("A", -3*nm, 0, "VCC"), Pin("B", 3*nm, 0, "NET_LED_A")])
    r2 = Footprint(ref="R2", width=8*nm, height=3*nm, color="#22AA99", pins=[Pin("A", -3*nm, 0, "NET_SPI_CLK"), Pin("B", 3*nm, 0, "NET_SPI_MOSI")])
    r3 = Footprint(ref="R3", width=8*nm, height=3*nm, color="#FF9922", pins=[Pin("A", -3*nm, 0, "NET_UART_TX"), Pin("B", 3*nm, 0, "NET_UART_RX")])
    r4 = Footprint(ref="R4", width=8*nm, height=3*nm, color="#FF5522", pins=[Pin("A", -3*nm, 0, "NET_3V3"), Pin("B", 3*nm, 0, "NET_EN")])
    d1 = Footprint(ref="D1", width=5*nm, height=5*nm, color="#FFCC22", pins=[Pin("A", -1.8*nm, 0, "NET_LED_A"), Pin("K", 1.8*nm, 0, "GND")])

    footprints = [ic1, u1, u2, r1, r2, r3, r4, d1]
    netlist = {}
    for fp in footprints:
        for pin in fp.pins:
            netlist.setdefault(pin.net_id, []).append((fp.ref, pin.pin_id))
    return footprints, netlist


def copy_genome(genome: Genome) -> Genome:
    # Footprints werden geteilt, nur Position/Drehung kopiert (deutlich schneller als deepcopy)
    return [PlacedComponent(c.footprint, c.x, c.y, c.rot) for c in genome]


def random_placement(footprints: List[Footprint]) -> Genome:
    margin = config.COURTYARD_MARGIN
    fixed_bboxes = [PlacedComponent(fp, *fp.fixed_placement).get_bbox() for fp in footprints if fp.fixed_placement is not None]
    placed: Genome = []
    placed_bboxes = []
    for fp in footprints:
        if fp.fixed_placement is not None:
            placed.append(PlacedComponent(fp, *fp.fixed_placement))
            placed_bboxes.append(placed[-1].get_bbox())
            continue
        for _ in range(500):
            rot = random.choice(VALID_ROTATIONS)
            ew, eh = (fp.height, fp.width) if rot in (90, 270) else (fp.width, fp.height)
            half_w, half_h = ew // 2, eh // 2
            x = random.randint(half_w, max(half_w, config.BOARD_W - half_w))
            y = random.randint(half_h, max(half_h, config.BOARD_H - half_h))
            ax1, ay1, ax2, ay2 = x - half_w - margin, y - half_h - margin, x + half_w + margin, y + half_h + margin
            if not any(min(ax2, bx2) > max(ax1, bx1) and min(ay2, by2) > max(ay1, by1)
                       for bx1, by1, bx2, by2 in placed_bboxes + fixed_bboxes):
                placed.append(PlacedComponent(footprint=fp, x=x, y=y, rot=rot))
                placed_bboxes.append((ax1, ay1, ax2, ay2))
                break
        else:
            placed.append(PlacedComponent(footprint=fp, x=config.BOARD_W//2, y=config.BOARD_H//2, rot=0))
            placed_bboxes.append(placed[-1].get_bbox())
    return placed


# --- Netz-Geometrie -----------------------------------------------------------

Point = Tuple[int, int]
Segment = Tuple[float, Point, Point]  # (Netzgewicht, Start, Ende)


def _pin_positions(genome: Genome) -> Dict[Tuple[str, str], Point]:
    positions = {}
    for comp in genome:
        ref = comp.footprint.ref
        for pin_id, _, abs_x, abs_y in comp.get_pin_positions():
            positions[(ref, pin_id)] = (abs_x, abs_y)
    return positions


def _mst_edges(points: List[Point]) -> List[Tuple[Point, Point]]:
    """Minimaler Spannbaum (Prim, O(n²)): verbindet n Pins mit n-1 Segmenten –
    so wie ein Router ein Netz verdrahtet, statt jeden Pin mit jedem zu verbinden."""
    n = len(points)
    if n < 2:
        return []
    x0, y0 = points[0]
    best = [(px - x0) ** 2 + (py - y0) ** 2 for px, py in points]
    parent = [0] * n
    remaining = list(range(1, n))
    edges = []
    while remaining:
        k = min(remaining, key=best.__getitem__)
        remaining.remove(k)
        edges.append((points[parent[k]], points[k]))
        kx, ky = points[k]
        for i in remaining:
            px, py = points[i]
            d = (px - kx) ** 2 + (py - ky) ** 2
            if d < best[i]:
                best[i] = d
                parent[i] = k
    return edges


def net_segments(genome: Genome, netlist: Dict[str, List[Tuple[str, str]]]) -> List[Segment]:
    """Luftlinien (Ratsnest) aller Netze als minimale Spannbäume."""
    pin_positions = _pin_positions(genome)
    segments = []
    for net_id, connections in netlist.items():
        net_pins = [pin_positions[key] for key in connections if key in pin_positions]
        weight = get_net_weight(net_id)
        for a, b in _mst_edges(net_pins):
            segments.append((weight, a, b))
    return segments


def _count_crossings(segments: List[Segment]) -> int:
    # Sweep über x: nur Segmente mit überlappenden Bounding-Boxen werden exakt geprüft
    boxes = sorted(
        (min(a[0], b[0]), max(a[0], b[0]), min(a[1], b[1]), max(a[1], b[1]), a, b)
        for _, a, b in segments
    )
    crossings = 0
    for i, (_, x2, y1, y2, A, B) in enumerate(boxes):
        ax, ay = A
        bx, by = B
        for j in range(i + 1, len(boxes)):
            ox1, _, oy1, oy2, C, D = boxes[j]
            if ox1 > x2:
                break
            if oy1 > y2 or oy2 < y1:
                continue
            # Segmente mit gemeinsamem Endpunkt kreuzen sich nicht
            if C == A or C == B or D == A or D == B:
                continue
            cx, cy = C
            dx, dy = D
            # ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
            if ((((dy - ay) * (cx - ax) > (cy - ay) * (dx - ax)) != ((dy - by) * (cx - bx) > (cy - by) * (dx - bx)))
                    and (((cy - ay) * (bx - ax) > (by - ay) * (cx - ax)) != ((dy - ay) * (bx - ax) > (by - ay) * (dx - ax)))):
                crossings += 1
    return crossings


def compute_crossing_penalty(genome: Genome, netlist: Dict[str, List[Tuple[str, str]]]) -> float:
    return float(_count_crossings(net_segments(genome, netlist)))


CROSSING_PENALTY = 50_000_000  # 50mm penalty per crossed net


def _tracelength_from_segments(segments: List[Segment]) -> float:
    total_length = sum(w * math.hypot(a[0] - b[0], a[1] - b[1]) for w, a, b in segments)
    return total_length + _count_crossings(segments) * CROSSING_PENALTY


def compute_tracelength_fitness(genome: Genome, netlist: Dict[str, List[Tuple[str, str]]]) -> float:
    if not all(comp.is_within_board() for comp in genome):
        return float('inf')
    return _tracelength_from_segments(net_segments(genome, netlist))


def _overlap_from_bboxes(genome: Genome, bboxes: List[Tuple[int, int, int, int]]) -> float:
    # Sweep über x: nur Paare mit überlappendem x-Bereich werden weiter geprüft
    order = sorted(range(len(genome)), key=lambda i: bboxes[i][0])
    total = 0.0
    for n, i in enumerate(order):
        ax1, ay1, ax2, ay2 = bboxes[i]
        i_fixed = genome[i].footprint.fixed_placement is not None
        for j in order[n + 1:]:
            bx1, by1, bx2, by2 = bboxes[j]
            if bx1 >= ax2:
                break
            # Zwei gesperrte Bauteile kann der EA nicht auseinanderbewegen
            if i_fixed and genome[j].footprint.fixed_placement is not None:
                continue
            inter_w = min(ax2, bx2) - max(ax1, bx1)
            inter_h = min(ay2, by2) - max(ay1, by1)
            if inter_w <= 0 or inter_h <= 0:
                continue
            smaller_area = min((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1))
            total += inter_w * inter_h / smaller_area
    return total


def compute_overlap_penalty(genome: Genome) -> float:
    return _overlap_from_bboxes(genome, [comp.get_bbox() for comp in genome])


def _bbox_area_from_bboxes(bboxes: List[Tuple[int, int, int, int]]) -> float:
    x_min = min(b[0] for b in bboxes)
    y_min = min(b[1] for b in bboxes)
    x_max = max(b[2] for b in bboxes)
    y_max = max(b[3] for b in bboxes)
    return (x_max - x_min) * (y_max - y_min)


def compute_bbox_area(genome: Genome) -> float:
    if not all(comp.is_within_board() for comp in genome):
        return float('inf')
    return _bbox_area_from_bboxes([comp.get_bbox() for comp in genome])


def _packing_density_from_bboxes(genome: Genome, bboxes: List[Tuple[int, int, int, int]]) -> float:
    # Gesperrte Bauteile (z.B. Stecker am Rand) zählen nicht – sie würden das Rechteck über die ganze Platine ziehen
    boxes = [b for c, b in zip(genome, bboxes) if c.footprint.fixed_placement is None] or bboxes
    width = max(b[2] for b in boxes) - min(b[0] for b in boxes)
    height = max(b[3] for b in boxes) - min(b[1] for b in boxes)
    # kleinstes Rechteck mit dem Seitenverhältnis der Platine, das alle Bauteile umschließt
    aspect = config.BOARD_W / config.BOARD_H
    rect_w = max(width, height * aspect)
    rect_h = rect_w / aspect
    used = sum((b[2] - b[0]) * (b[3] - b[1]) for b in boxes)
    return min(1.0, used / (rect_w * rect_h))


def compute_packing_density(genome: Genome) -> float:
    """Anteil der Fläche, den die Bauteile (inkl. Abstand) im umschließenden Rechteck mit dem
    Seitenverhältnis der Platine belegen: 1.0 = lückenlos gepackt, klein = viel ungenutzter Platz."""
    if not all(comp.is_within_board() for comp in genome):
        return 0.0
    return _packing_density_from_bboxes(genome, [comp.get_bbox() for comp in genome])


def evaluate_genome(genome: Genome, netlist: Dict[str, List[Tuple[str, str]]]) -> Tuple[float, float, float]:
    """(Trace-Score, Overlap-Penalty, Packungsdichte) in einem Durchlauf."""
    bboxes = [comp.get_bbox() for comp in genome]
    overlap = _overlap_from_bboxes(genome, bboxes)
    if not all(comp.is_within_board() for comp in genome):
        return float('inf'), overlap, 0.0
    trace = _tracelength_from_segments(net_segments(genome, netlist))
    return trace, overlap, _packing_density_from_bboxes(genome, bboxes)


# --- Parallele Bewertung ------------------------------------------------------
# Die Worker-Prozesse bekommen Footprints und Netzliste einmalig beim Start;
# pro Generation werden nur (ref, x, y, rot) und die aktuellen config-Werte übertragen.

_pool = None
_pool_key = None
_pool_refs = None  # hält Netzliste/Footprints am Leben, damit ihre id() eindeutig bleibt
_worker_footprints: Dict[str, Footprint] = {}
_worker_netlist: Dict[str, List[Tuple[str, str]]] = {}


def _worker_init(footprints: Dict[str, Footprint], netlist: Dict[str, List[Tuple[str, str]]]) -> None:
    global _worker_footprints, _worker_netlist
    _worker_footprints, _worker_netlist = footprints, netlist


def _worker_evaluate(task) -> List[Tuple[float, float, float]]:
    board_w, board_h, margin, genomes = task
    config.set_board_size(board_w, board_h)
    config.COURTYARD_MARGIN = margin
    fps = _worker_footprints
    return [evaluate_genome([PlacedComponent(fps[ref], x, y, rot) for ref, x, y, rot in g], _worker_netlist)
            for g in genomes]


def shutdown_pool() -> None:
    global _pool, _pool_key, _pool_refs
    if _pool is not None:
        _pool.terminate()
    _pool = _pool_key = _pool_refs = None


atexit.register(shutdown_pool)


def _worker_count() -> int:
    return PARALLEL_WORKERS or max(1, (os.cpu_count() or 1) - 1)


def _get_pool(population: List[Genome], netlist: Dict[str, List[Tuple[str, str]]]):
    global _pool, _pool_key, _pool_refs
    footprints = [c.footprint for c in population[0]]
    key = (id(netlist), tuple(id(fp) for fp in footprints))
    if _pool is not None and key == _pool_key:
        return _pool
    shutdown_pool()
    _pool = multiprocessing.Pool(_worker_count(), initializer=_worker_init,
                                 initargs=({fp.ref: fp for fp in footprints}, netlist))
    _pool_key, _pool_refs = key, (netlist, footprints)
    return _pool


def _use_parallel(population: List[Genome], netlist: Dict[str, List[Tuple[str, str]]]) -> bool:
    if _worker_count() < 2 or not population:
        return False
    # Grobe Aufwandsschätzung: Pins pro Individuum × Populationsgröße
    pins = sum(len(connections) for connections in netlist.values())
    return pins * len(population) >= PARALLEL_MIN_WORK


def evaluate_population(population: List[Genome], netlist: Dict[str, List[Tuple[str, str]]]) -> List[Tuple[float, float, float]]:
    if _use_parallel(population, netlist):
        try:
            pool = _get_pool(population, netlist)
            chunk = math.ceil(len(population) / _worker_count())
            compact = [[(c.footprint.ref, c.x, c.y, c.rot) for c in g] for g in population]
            tasks = [(config.BOARD_W, config.BOARD_H, config.COURTYARD_MARGIN, compact[i:i + chunk])
                     for i in range(0, len(compact), chunk)]
            return [scores for part in pool.map(_worker_evaluate, tasks) for scores in part]
        except Exception as e:  # z.B. Prozesse lassen sich nicht starten – seriell weiterrechnen
            print(f"Parallele Bewertung fehlgeschlagen ({e}), rechne seriell weiter")
            shutdown_pool()
    return [evaluate_genome(g, netlist) for g in population]


# --- EA -------------------------------------------------------------------------

def _normalize_lower_is_better(values: List[float]) -> List[float]:
    # Verhältnis zum besten Wert statt Min-Max: 5 % schlechter ergibt 0.95. Min-Max würde
    # winzige Unterschiede (z.B. der Bounding-Box auf einer vollen Platine) auf 0…1 aufblähen.
    finite = [v for v in values if v != float('inf')]
    if not finite:
        return [0.0] * len(values)
    v_min = min(finite)
    return [0.0 if v == float('inf') else (1.0 if v <= 0 else v_min / v) for v in values]


def normalize_population_fitness(
    population: List[Genome],
    netlist: Dict[str, List[Tuple[str, str]]],
    trace_weight: float = TRACE_WEIGHT,
    overlap_weight: float = OVERLAP_WEIGHT,
    bbox_weight: float = BBOX_WEIGHT,
) -> List[float]:
    scores    = evaluate_population(population, netlist)
    traces    = [s[0] for s in scores]
    overlaps  = [s[1] for s in scores]
    densities = [s[2] for s in scores]  # bereits absolut 0…1, keine Normierung nötig

    norm_traces = _normalize_lower_is_better(traces)

    combined = []
    for nt, o, density in zip(norm_traces, overlaps, densities):
        if o > 0.0:
            score = 0.01 / (1.0 + o)
        else:
            score = 1.0 + (trace_weight * nt + bbox_weight * density)
        combined.append(score)

    return combined


def tournament_selection(population: List[Genome], fitness_vals: List[float], k: int) -> Genome:
    candidates = random.sample(range(len(population)), k)
    best = max(candidates, key=lambda idx: fitness_vals[idx])
    return copy_genome(population[best])


def uniform_crossover(parent1: Genome, parent2: Genome) -> Tuple[Genome, Genome]:
    # Die Eltern sind bereits Kopien aus der Turnierselektion, ihre Gene können direkt übernommen werden
    child1, child2 = [], []
    for c1, c2 in zip(parent1, parent2):
        if random.random() < 0.5:
            child1.append(c1); child2.append(c2)
        else:
            child1.append(c2); child2.append(c1)
    return child1, child2


def mutation_schedule(generation: int, movable_parts: int) -> Tuple[float, float]:
    """(Schrittweite in nm, Mutationsrate pro Bauteil) für diese Generation:
    anfangs große Sprünge vieler Bauteile, dann exponentielle Abkühlung zur Feinjustierung."""
    cooling = math.exp(-generation / ANNEAL_GENS)
    sigma_start = SIGMA_START_FRAC * max(config.BOARD_W, config.BOARD_H)
    sigma = SIGMA_MIN + max(0.0, sigma_start - SIGMA_MIN) * cooling
    rate_end = min(1.0, MUTATION_MIN_PARTS / max(1, movable_parts))
    rate = rate_end + max(0.0, MUTATION_RATE - rate_end) * cooling
    return sigma, rate


def _collisions(bbox: Tuple[int, int, int, int], bboxes: List[Tuple[int, int, int, int]], skip: int) -> int:
    ax1, ay1, ax2, ay2 = bbox
    hits = 0
    for j, (bx1, by1, bx2, by2) in enumerate(bboxes):
        # gleichbedeutend mit min(ax2, bx2) > max(ax1, bx1) usw., aber ohne min/max-Aufrufe
        if bx1 < ax2 and bx2 > ax1 and by1 < ay2 and by2 > ay1 and j != skip:
            hits += 1
    return hits


def _colliding_indices(bboxes: List[Tuple[int, int, int, int]]) -> set:
    order = sorted(range(len(bboxes)), key=lambda i: bboxes[i][0])
    hits = set()
    for n, i in enumerate(order):
        ax1, ay1, ax2, ay2 = bboxes[i]
        for j in order[n + 1:]:
            bx1, by1, bx2, by2 = bboxes[j]
            if bx1 >= ax2:
                break
            if min(ay2, by2) > max(ay1, by1):
                hits.update((i, j))
    return hits


def _clamp_to_board(comp: PlacedComponent) -> None:
    w, h = comp._rotated_dims()
    comp.x = max(w // 2, min(config.BOARD_W - w // 2, comp.x))
    comp.y = max(h // 2, min(config.BOARD_H - h // 2, comp.y))


def _slide_towards(comp: PlacedComponent, i: int, bboxes: List[Tuple[int, int, int, int]], axis: int, target: int) -> None:
    """Schiebt ein Bauteil entlang einer Achse Richtung `target`, bis es den nächsten Nachbarn berührt."""
    lo, hi = axis, axis + 2                  # Koordinaten entlang der Bewegungsachse
    cross_lo, cross_hi = 1 - axis, 3 - axis  # Koordinaten quer dazu
    box = bboxes[i]
    center = comp.x if axis == 0 else comp.y
    distance = abs(target - center)
    direction = 1 if target > center else -1
    for j, other in enumerate(bboxes):
        if j == i or distance == 0:
            continue
        if min(box[cross_hi], other[cross_hi]) <= max(box[cross_lo], other[cross_lo]):
            continue  # liegt quer versetzt, kann nicht im Weg sein
        gap = other[lo] - box[hi] if direction > 0 else box[lo] - other[hi]
        if gap >= 0:
            distance = min(distance, gap)
    if distance <= 0:
        return
    if axis == 0:
        comp.x += direction * distance
    else:
        comp.y += direction * distance
    _clamp_to_board(comp)  # schiebt höchstens auf dem eben freien Weg zurück
    bboxes[i] = comp.get_bbox()


def mutate(genome: Genome, sigma: float, mutation_rate: float) -> Genome:
    """Gauß-Verschiebung mit Reparatur: ein Zug, der neue Überlappungen erzeugt, wird neu gewürfelt
    und nach MUTATION_TRIES Fehlversuchen verworfen. Bereits überlappende Bauteile (z.B. nach dem
    Crossover) werden immer bewegt und dürfen sich dabei nur verbessern."""
    bboxes = [c.get_bbox() for c in genome]
    colliding = _colliding_indices(bboxes)
    movable = [i for i, c in enumerate(genome) if c.footprint.fixed_placement is None]

    for i in movable:
        if i not in colliding and random.random() >= mutation_rate:
            continue
        comp = genome[i]
        old_x, old_y, old_rot = comp.x, comp.y, comp.rot
        before = _collisions(bboxes[i], bboxes, i) if i in colliding else 0
        for _ in range(MUTATION_TRIES):
            comp.x = old_x + int(random.gauss(0, sigma))
            comp.y = old_y + int(random.gauss(0, sigma))
            comp.rot = random.choice(VALID_ROTATIONS) if random.random() < ROT_MUTATE_P else old_rot
            _clamp_to_board(comp)
            bbox = comp.get_bbox()
            after = _collisions(bbox, bboxes, i)
            if after == 0 or after < before:
                bboxes[i] = bbox
                break
        else:
            comp.x, comp.y, comp.rot = old_x, old_y, old_rot

    # Verdichten: Bauteile Richtung Mitte der Gruppe ziehen – so weit, wie es überlappungsfrei geht
    if movable and random.random() < COMPACT_P:
        boxes = [bboxes[i] for i in movable]
        center_x = (min(b[0] for b in boxes) + max(b[2] for b in boxes)) // 2
        center_y = (min(b[1] for b in boxes) + max(b[3] for b in boxes)) // 2
        # innere Bauteile zuerst, damit die äußeren in die frei werdenden Lücken nachrücken können
        order = sorted(movable, key=lambda k: (genome[k].x - center_x) ** 2 + (genome[k].y - center_y) ** 2)
        for i in order:
            if i in colliding:
                continue
            axes = (0, 1) if random.random() < 0.5 else (1, 0)
            for axis in axes:
                _slide_towards(genome[i], i, bboxes, axis, center_x if axis == 0 else center_y)

    # Platztausch zweier Bauteile: große Umordnung, auch wenn die Platine voll ist
    if len(movable) >= 2 and random.random() < SWAP_P:
        i, j = random.sample(movable, 2)
        a, b = genome[i], genome[j]
        old_a, old_b = (a.x, a.y), (b.x, b.y)
        old_bbox_a, old_bbox_b = bboxes[i], bboxes[j]
        (a.x, a.y), (b.x, b.y) = old_b, old_a
        _clamp_to_board(a)
        _clamp_to_board(b)
        bboxes[i], bboxes[j] = a.get_bbox(), b.get_bbox()
        if _collisions(bboxes[i], bboxes, i) or _collisions(bboxes[j], bboxes, j):
            (a.x, a.y), (b.x, b.y) = old_a, old_b
            bboxes[i], bboxes[j] = old_bbox_a, old_bbox_b
    return genome


def evolve_one_generation(
    population: List[Genome],
    fitness_vals: List[float],
    netlist: Dict[str, List[Tuple[str, str]]],
    generation: int = 0,
) -> Tuple[List[Genome], List[float]]:
    new_population: List[Genome] = []

    best_idx = max(range(len(fitness_vals)), key=lambda i: fitness_vals[i])
    new_population.append(copy_genome(population[best_idx]))

    movable_parts = sum(1 for c in population[0] if c.footprint.fixed_placement is None)
    sigma, mutation_rate = mutation_schedule(generation, movable_parts)

    while len(new_population) < POP_SIZE:
        p1 = tournament_selection(population, fitness_vals, TOURNAMENT_K)
        p2 = tournament_selection(population, fitness_vals, TOURNAMENT_K)
        if random.random() < CROSSOVER_RATE:
            c1, c2 = uniform_crossover(p1, p2)
        else:
            c1, c2 = p1, p2
        new_population.append(mutate(c1, sigma, mutation_rate))
        if len(new_population) < POP_SIZE:
            new_population.append(mutate(c2, sigma, mutation_rate))

    new_fitness = normalize_population_fitness(new_population, netlist)
    return new_population, new_fitness