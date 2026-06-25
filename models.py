# models.py
import math
from dataclasses import dataclass, field
from typing import List, Tuple

from config import BOARD_W, BOARD_H


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

        x_min = self.x - w // 2
        y_min = self.y - h // 2
        x_max = x_min + w
        y_max = y_min + h

        return x_min, y_min, x_max, y_max

    def get_pin_positions(self) -> List[Tuple[str, str, int, int]]:
        results = []

        angle_rad = math.radians(self.rot)
        cos_a = round(math.cos(angle_rad))
        sin_a = round(math.sin(angle_rad))

        for pin in self.footprint.pins:
            rx, ry = pin.rel_x, pin.rel_y

            rot_x = int(rx * cos_a + ry * sin_a)
            rot_y = int(-rx * sin_a + ry * cos_a)

            abs_x = self.x + rot_x
            abs_y = self.y + rot_y

            results.append((pin.pin_id, pin.net_id, abs_x, abs_y))

        return results

    def is_within_board(self) -> bool:
        x_min, y_min, x_max, y_max = self.get_bbox()
        return x_min >= 0 and y_min >= 0 and x_max <= BOARD_W and y_max <= BOARD_H

    def overlaps(self, other: "PlacedComponent") -> float:
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


Genome = List[PlacedComponent]