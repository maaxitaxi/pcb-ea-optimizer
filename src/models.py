# models.py
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import config

@dataclass
class Pin:
    pin_id: str          
    rel_x: int           
    rel_y: int           
    net_id: str          

@dataclass
class Footprint:
    ref: str             
    width: int           
    height: int          
    pins: List[Pin] = field(default_factory=list)
    color: str = "#4FC3F7"
    fixed_placement: Optional[Tuple[int, int, int]] = None  # (x, y, rot) gesperrter Bauteile – der EA bewegt sie nicht

# (cos, sin) je Drehwinkel – erspart math.cos/sin bei jeder Pin-Berechnung
_ROT_COS_SIN = {0: (1, 0), 90: (0, 1), 180: (-1, 0), 270: (0, -1)}


@dataclass
class PlacedComponent:
    footprint: Footprint
    x: int               
    y: int               
    rot: int             

    def _rotated_dims(self) -> Tuple[int, int]:
        if self.rot in (90, 270):
            return self.footprint.height, self.footprint.width
        return self.footprint.width, self.footprint.height

    def get_bbox(self) -> Tuple[int, int, int, int]:
        w, h = self._rotated_dims()
        margin = config.COURTYARD_MARGIN
        x_min = self.x - w // 2 - margin
        y_min = self.y - h // 2 - margin
        x_max = self.x + w // 2 + margin
        y_max = self.y + h // 2 + margin
        return x_min, y_min, x_max, y_max

    def get_pin_positions(self) -> List[Tuple[str, str, int, int]]:
        cos_a, sin_a = _ROT_COS_SIN[self.rot]
        x, y = self.x, self.y
        return [(pin.pin_id, pin.net_id,
                 x + int(pin.rel_x * cos_a - pin.rel_y * sin_a),
                 y + int(pin.rel_x * sin_a + pin.rel_y * cos_a))
                for pin in self.footprint.pins]

    def is_within_board(self) -> bool:
        # Gesperrte Bauteile (z.B. Stecker am Platinenrand) dürfen überstehen
        if self.footprint.fixed_placement is not None:
            return True
        w, h = self._rotated_dims()
        x_min = self.x - w // 2
        y_min = self.y - h // 2
        x_max = x_min + w
        y_max = y_min + h
        return x_min >= 0 and y_min >= 0 and x_max <= config.BOARD_W and y_max <= config.BOARD_H

    def overlaps(self, other: 'PlacedComponent') -> float:
        ax1, ay1, ax2, ay2 = self.get_bbox()
        bx1, by1, bx2, by2 = other.get_bbox()

        inter_w = min(ax2, bx2) - max(ax1, bx1)
        inter_h = min(ay2, by2) - max(ay1, by1)

        if inter_w <= 0 or inter_h <= 0:
            return 0.0

        inter_area = inter_w * inter_h
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        smaller_area = min(area_a, area_b)

        return inter_area / smaller_area

    def __deepcopy__(self, memo) -> 'PlacedComponent':
        # Der Footprint ändert sich während des EA nie – teilen statt mitkopieren
        return PlacedComponent(self.footprint, self.x, self.y, self.rot)

Genome = List[PlacedComponent]