# dsn_parser.py
"""Liest Specctra-DSN-Dateien (KiCad: Datei → Exportieren → Specctra DSN) in das
EA-Datenmodell ein und schreibt eine optimierte Platzierung wieder zurück.

Koordinaten: Intern wird im DSN-Koordinatensystem gerechnet (y nach oben, Drehung
gegen den Uhrzeigersinn), nur um die linke untere Ecke des Platinenumrisses
verschoben, sodass die Platine bei (0, 0) beginnt. Die Bauteil-Position (x, y) im
EA ist der Mittelpunkt der Footprint-Bounding-Box – in der DSN steht dagegen der
Footprint-Ursprung, der bei vielen Footprints (z.B. Stiftleisten) nicht mittig liegt.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from models import Footprint, Genome, PlacedComponent, Pin

NM_PER_UNIT = {"inch": 25_400_000, "mil": 25_400, "cm": 10_000_000, "mm": 1_000_000, "um": 1_000}

PALETTE = ["#FF5555", "#BB55BB", "#7755CC", "#55BB55", "#22AA99", "#FF9922", "#FF5522", "#FFCC22", "#4FC3F7"]


class DsnParseError(ValueError):
    pass


# ---------------------------------------------------------------------------
# S-Expression-Parser (mit Textpositionen, damit wir gezielt zurückschreiben können)
# ---------------------------------------------------------------------------

@dataclass
class Atom:
    text: str   # Rohtext, ggf. mit Anführungszeichen
    value: str  # Text ohne Anführungszeichen
    start: int
    end: int


@dataclass
class Node:
    items: list
    start: int
    end: int = -1

    @property
    def head(self) -> str:
        return self.items[0].value if self.items and isinstance(self.items[0], Atom) else ""

    def atoms(self) -> List[Atom]:
        return [item for item in self.items[1:] if isinstance(item, Atom)]

    def children(self) -> List["Node"]:
        return [item for item in self.items if isinstance(item, Node)]

    def find(self, name: str) -> Optional["Node"]:
        return next((c for c in self.children() if c.head == name), None)

    def find_all(self, name: str) -> List["Node"]:
        return [c for c in self.children() if c.head == name]


def parse_sexpr(text: str) -> Node:
    quote = '"'
    stack: List[Node] = []
    root: Optional[Node] = None
    expect_quote_char = False
    i, n = 0, len(text)

    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if expect_quote_char:
            # "(string_quote ")" – das nächste Zeichen IST das Anführungszeichen
            quote = ch
            stack[-1].items.append(Atom(ch, ch, i, i + 1))
            expect_quote_char = False
            i += 1
            continue
        if ch == "(":
            stack.append(Node([], i))
            i += 1
            continue
        if ch == ")":
            if not stack:
                raise DsnParseError(f"Unerwartete ')' an Position {i}")
            node = stack.pop()
            node.end = i + 1
            if stack:
                stack[-1].items.append(node)
            elif root is None:
                root = node
            i += 1
            continue

        start = i
        while i < n and not text[i].isspace() and text[i] not in "()":
            if text[i] == quote:
                close = text.find(quote, i + 1)
                if close == -1:
                    raise DsnParseError(f"Nicht geschlossenes Anführungszeichen an Position {i}")
                i = close + 1
            else:
                i += 1
        if not stack:
            raise DsnParseError(f"Text außerhalb einer Klammer an Position {start}")
        raw = text[start:i]
        stack[-1].items.append(Atom(raw, raw.replace(quote, ""), start, i))
        if raw == "string_quote" and len(stack[-1].items) == 1:
            expect_quote_char = True

    if stack:
        raise DsnParseError("Datei endet mit offenen Klammern")
    if root is None:
        raise DsnParseError("Keine S-Expression gefunden")
    return root


# ---------------------------------------------------------------------------
# DSN-Auswertung
# ---------------------------------------------------------------------------

@dataclass
class DsnComponent:
    ref: str
    image: str
    side: str                    # "front" | "back"
    locked: bool
    place: Node                  # Original-(place …)-Knoten, dessen Zahlen beim Export ersetzt werden
    center: Tuple[int, int]      # Bbox-Mittelpunkt relativ zum Footprint-Ursprung (nm, ggf. gespiegelt)


@dataclass
class DsnDesign:
    text: str
    board_origin: Tuple[int, int]            # linke untere Ecke des Umrisses (nm, DSN-Koordinaten)
    board_size: Tuple[int, int]              # Breite, Höhe (nm)
    footprints: List[Footprint]
    netlist: Dict[str, List[Tuple[str, str]]]
    components: Dict[str, DsnComponent]
    initial_genome: Genome                   # aktuelle Platzierung aus KiCad
    placement_unit_nm: float
    resolution_nm: float
    warnings: List[str]


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _unit_nm(node: Optional[Node], default: float) -> float:
    unit = node.find("unit") if node is not None else None
    if unit is None or not unit.atoms():
        return default
    name = unit.atoms()[0].value.lower()
    if name not in NM_PER_UNIT:
        raise DsnParseError(f"Unbekannte Einheit '{name}'")
    return NM_PER_UNIT[name]


def _shape_points(shape: Node) -> List[Tuple[float, float]]:
    """Eckpunkte der Bounding-Box einer Specctra-Form (rect, circle, path, polygon)."""
    nums = [float(a.value) for a in shape.atoms()[1:] if _is_number(a.value)]  # [0] ist der Layer
    kind = shape.head
    if kind == "rect" and len(nums) >= 4:
        return [(nums[0], nums[1]), (nums[2], nums[3])]
    if kind == "circle" and nums:
        r = nums[0] / 2
        cx, cy = (nums[1], nums[2]) if len(nums) >= 3 else (0.0, 0.0)
        return [(cx - r, cy - r), (cx + r, cy + r)]
    if kind in ("path", "polygon", "polyline_path") and len(nums) >= 3:
        hw = nums[0] / 2
        coords = nums[1:]
        points = []
        for x, y in zip(coords[0::2], coords[1::2]):
            points += [(x - hw, y - hw), (x + hw, y + hw)]
        return points
    return []


def _rotate(x: float, y: float, rot: int) -> Tuple[float, float]:
    """Drehung gegen den Uhrzeigersinn um ein Vielfaches von 90° (wie PlacedComponent.get_pin_positions)."""
    return [(x, y), (-y, x), (-x, -y), (y, -x)][(rot // 90) % 4]


def _snap_rotation(rot: float) -> int:
    return int(round(rot / 90.0)) % 4 * 90


def _split_pin_ref(pin_ref: str, refs_longest_first: List[str]) -> Optional[Tuple[str, str]]:
    """'U1-7' → ('U1', '7'). Referenz über bekannte Bauteile, da auch Pins '-' enthalten dürfen."""
    for ref in refs_longest_first:
        if pin_ref.startswith(ref + "-"):
            return ref, pin_ref[len(ref) + 1:]
    return None


def load_dsn(path: str) -> DsnDesign:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    root = parse_sexpr(text)
    if root.head != "pcb":
        raise DsnParseError(f"Keine Specctra-DSN-Datei (Wurzelelement '{root.head}' statt 'pcb')")

    warnings: List[str] = []

    resolution = root.find("resolution")
    res_unit_nm = NM_PER_UNIT["inch"]
    resolution_nm = 1.0
    if resolution is not None and len(resolution.atoms()) >= 2:
        res_unit_nm = NM_PER_UNIT[resolution.atoms()[0].value.lower()]
        resolution_nm = res_unit_nm / float(resolution.atoms()[1].value)
    unit_nm = _unit_nm(root, res_unit_nm)

    structure = root.find("structure")
    placement = root.find("placement")
    library = root.find("library")
    network = root.find("network")
    for name, node in (("structure", structure), ("placement", placement), ("library", library), ("network", network)):
        if node is None:
            raise DsnParseError(f"Abschnitt '({name} …)' fehlt")

    # --- Platinenumriss ----------------------------------------------------
    structure_unit = _unit_nm(structure, unit_nm)
    pcb_points, other_points = [], []
    for boundary in structure.find_all("boundary"):
        for shape in boundary.children():
            layer = shape.atoms()[0].value if shape.atoms() else ""
            points = [(x * structure_unit, y * structure_unit) for x, y in _shape_points(shape)]
            (pcb_points if layer == "pcb" else other_points).extend(points)
    boundary_points = pcb_points or other_points
    if not boundary_points:
        raise DsnParseError("Kein Platinenumriss (boundary) gefunden – ist Edge.Cuts in KiCad geschlossen?")
    bx1 = int(round(min(p[0] for p in boundary_points)))
    by1 = int(round(min(p[1] for p in boundary_points)))
    bx2 = int(round(max(p[0] for p in boundary_points)))
    by2 = int(round(max(p[1] for p in boundary_points)))

    # --- Padstacks und Footprint-Bilder -------------------------------------
    library_unit = _unit_nm(library, unit_nm)
    pad_extents: Dict[str, Tuple[float, float, float, float]] = {}
    for padstack in library.find_all("padstack"):
        points = []
        for shape_node in padstack.find_all("shape"):
            for shape in shape_node.children():
                points += _shape_points(shape)
        if points:
            pad_extents[padstack.atoms()[0].value] = (
                min(p[0] for p in points), min(p[1] for p in points),
                max(p[0] for p in points), max(p[1] for p in points),
            )

    images: Dict[str, Tuple[List[Tuple[str, float, float]], List[Tuple[float, float]]]] = {}
    for image in library.find_all("image"):
        pins: List[Tuple[str, float, float]] = []
        points: List[Tuple[float, float]] = []
        for pin in image.find_all("pin"):
            atoms = pin.atoms()  # [padstack, pin_id, x, y] – (rotate …) ist ein Unterknoten
            if len(atoms) < 4:
                continue
            padstack, pin_id = atoms[0].value, atoms[1].value
            px, py = float(atoms[2].value), float(atoms[3].value)
            pins.append((pin_id, px, py))
            ext = pad_extents.get(padstack, (0.0, 0.0, 0.0, 0.0))
            rotate = pin.find("rotate")
            pad_rot = _snap_rotation(float(rotate.atoms()[0].value)) if rotate is not None and rotate.atoms() else 0
            for ex, ey in ((ext[0], ext[1]), (ext[2], ext[3])):
                rx, ry = _rotate(ex, ey, pad_rot)
                points.append((px + rx, py + ry))
        for container in image.find_all("outline") + image.find_all("keepout"):
            for shape in container.children():
                points += _shape_points(shape)
        images[image.atoms()[0].value] = (pins, points)

    # --- Platzierung ---------------------------------------------------------
    placement_unit = _unit_nm(placement, unit_nm)
    raw_places = []  # (ref, image, place_node, origin|None, side, rot, locked)
    for comp_node in placement.find_all("component"):
        image_name = comp_node.atoms()[0].value
        for place in comp_node.find_all("place"):
            atoms = place.atoms()
            ref = atoms[0].value
            if len(atoms) >= 5:
                origin = (float(atoms[1].value) * placement_unit, float(atoms[2].value) * placement_unit)
                side, rot = atoms[3].value, float(atoms[4].value)
            else:
                origin, side, rot = None, "front", 0.0
            raw_places.append((ref, image_name, place, origin, side, rot, place.find("lock_type") is not None))

    # --- Netzliste -----------------------------------------------------------
    refs_longest_first = sorted((p[0] for p in raw_places), key=len, reverse=True)
    pin_to_net: Dict[Tuple[str, str], str] = {}
    for net in network.find_all("net"):
        net_name = net.atoms()[0].value
        for pins_node in net.find_all("pins"):
            for atom in pins_node.atoms():
                split = _split_pin_ref(atom.value, refs_longest_first)
                if split is None:
                    warnings.append(f"Pin '{atom.value}' in Netz '{net_name}' gehört zu keinem platzierten Bauteil")
                    continue
                pin_to_net[split] = net_name

    # --- In das EA-Modell übersetzen -----------------------------------------
    footprints: List[Footprint] = []
    components: Dict[str, DsnComponent] = {}
    initial_genome: Genome = []
    netlist: Dict[str, List[Tuple[str, str]]] = {}

    for index, (ref, image_name, place, origin, side, rot, locked) in enumerate(raw_places):
        if image_name not in images:
            raise DsnParseError(f"Bauteil {ref}: Footprint-Bild '{image_name}' fehlt in (library …)")
        pins, points = images[image_name]
        mirror = -1 if side == "back" else 1  # Unterseite: Bild an der y-Achse gespiegelt
        points = [(mirror * x * library_unit, y * library_unit) for x, y in points]
        if not points:
            points = [(-500_000, -500_000), (500_000, 500_000)]
            warnings.append(f"Bauteil {ref}: keine Geometrie gefunden, verwende 1×1 mm")
        lx1, ly1 = min(p[0] for p in points), min(p[1] for p in points)
        lx2, ly2 = max(p[0] for p in points), max(p[1] for p in points)
        cx, cy = (lx1 + lx2) / 2, (ly1 + ly2) / 2

        fp_pins = []
        for pin_id, px, py in pins:
            net_name = pin_to_net.get((ref, pin_id))
            if net_name is None:
                continue  # unbeschalteter Pin: zählt nur für die Bauteilgröße
            fp_pins.append(Pin(pin_id, int(round(mirror * px * library_unit - cx)), int(round(py * library_unit - cy)), net_name))
            netlist.setdefault(net_name, []).append((ref, pin_id))

        rot90 = _snap_rotation(rot)
        if origin is None:
            x, y = (bx2 - bx1) // 2, (by2 - by1) // 2
            if locked:
                warnings.append(f"Bauteil {ref} ist gesperrt, hat aber keine Position – wird frei platziert")
                locked = False
        else:
            ccx, ccy = _rotate(cx, cy, rot90)
            x, y = int(round(origin[0] + ccx - bx1)), int(round(origin[1] + ccy - by1))
        if locked and rot != rot90:
            warnings.append(f"Bauteil {ref}: Drehung {rot}° wird für die Kollisionsprüfung auf {rot90}° gerundet")

        footprint = Footprint(
            ref=ref,
            width=max(int(round(lx2 - lx1)), 1),
            height=max(int(round(ly2 - ly1)), 1),
            pins=fp_pins,
            color=PALETTE[index % len(PALETTE)],
            fixed_placement=(x, y, rot90) if locked else None,
        )
        footprints.append(footprint)
        initial_genome.append(PlacedComponent(footprint, x, y, rot90))
        components[ref] = DsnComponent(ref, image_name, side, locked, place, (int(round(cx)), int(round(cy))))

    for (ref, pin_id), net_name in pin_to_net.items():
        if ref in components and not any(image_pin[0] == pin_id for image_pin in images[components[ref].image][0]):
            warnings.append(f"Pin {ref}-{pin_id} (Netz '{net_name}') fehlt im Footprint-Bild")

    return DsnDesign(
        text=text,
        board_origin=(bx1, by1),
        board_size=(bx2 - bx1, by2 - by1),
        footprints=footprints,
        netlist=netlist,
        components=components,
        initial_genome=initial_genome,
        placement_unit_nm=placement_unit,
        resolution_nm=resolution_nm,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Zurückschreiben
# ---------------------------------------------------------------------------

def _format_coord(value_nm: float, design: DsnDesign) -> str:
    snapped = round(value_nm / design.resolution_nm) * design.resolution_nm
    text = f"{snapped / design.placement_unit_nm:.6f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def render_dsn(design: DsnDesign, genome: Genome) -> str:
    """Original-DSN mit der Platzierung aus `genome`. Nur die Zahlen in den
    (place …)-Einträgen werden ersetzt, alles andere bleibt unverändert."""
    edits: List[Tuple[int, int, str]] = []
    for comp in genome:
        meta = design.components[comp.footprint.ref]
        if meta.locked:
            continue
        ccx, ccy = _rotate(*meta.center, comp.rot)
        x_txt = _format_coord(comp.x + design.board_origin[0] - ccx, design)
        y_txt = _format_coord(comp.y + design.board_origin[1] - ccy, design)
        atoms = meta.place.atoms()
        if len(atoms) >= 5:
            edits += [
                (atoms[1].start, atoms[1].end, x_txt),
                (atoms[2].start, atoms[2].end, y_txt),
                (atoms[4].start, atoms[4].end, str(comp.rot)),
            ]
        else:
            edits.append((atoms[0].end, atoms[0].end, f" {x_txt} {y_txt} front {comp.rot}"))

    text = design.text
    for start, end, replacement in sorted(edits, reverse=True):
        text = text[:start] + replacement + text[end:]
    return text


def write_dsn(design: DsnDesign, genome: Genome, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_dsn(design, genome))
