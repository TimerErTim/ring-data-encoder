import abc
import math
import random
from typing import List, Tuple

try:
    from hilbertcurve.hilbertcurve import HilbertCurve
    HILBERT_AVAILABLE = True
except ImportError:
    HILBERT_AVAILABLE = False


class PathGenerator(abc.ABC):
    """Abstract base class for path generators."""

    @abc.abstractmethod
    def generate(self, width: float, height: float, num_points: int, margin: float = 0.0) -> List[Tuple[float, float]]:
        """
        Generates a list of (x, y) coordinates.

        :param width: The width of the area.
        :param height: The height of the area.
        :param num_points: The number of points to generate.
        :param margin: The margin to leave around the path.
        :return: A list of (x, y) tuples.
        """
        pass


class SnakePathGenerator(PathGenerator):
    """Generates points in a snake-like (zig-zag) pattern."""

    def generate(self, width: float, height: float, num_points: int, margin: float = 0.0) -> List[Tuple[float, float]]:
        if num_points == 0:
            return []

        padded_width = width - 2 * margin
        padded_height = height - 2 * margin

        if num_points == 1:
            return [(width / 2, height / 2)]

        if padded_width <= 0 or padded_height <= 0:
            return []

        cols = int(math.sqrt(num_points * padded_width / padded_height)) if padded_height > 0 else num_points
        if cols == 0:
            cols = 1
        rows = math.ceil(num_points / cols)

        x_spacing_if_fill = padded_width / (cols + 1) if cols > 0 else padded_width
        y_spacing_if_fill = padded_height / (rows + 1) if rows > 0 else padded_height
        spacing = min(x_spacing_if_fill, y_spacing_if_fill)

        grid_width = (cols + 1) * spacing
        grid_height = (rows + 1) * spacing
        x_offset = margin + (padded_width - grid_width) / 2
        y_offset = margin + (padded_height - grid_height) / 2


        points = []
        for r in range(int(rows)):
            row_points = []
            for c in range(cols):
                point_index = r * cols + c
                if point_index < num_points:
                    x = x_offset + (c + 1) * spacing
                    y = y_offset + (r + 1) * spacing
                    row_points.append((x, y))
            if r % 2 == 1:
                row_points.reverse()
            points.extend(row_points)
        return points


class HilbertPathGenerator(PathGenerator):
    """Generates points along a Hilbert curve."""

    def generate(self, width: float, height: float, num_points: int, margin: float = 0.0) -> List[Tuple[float, float]]:
        if not HILBERT_AVAILABLE:
            raise RuntimeError("The 'hilbertcurve' library is required for the Hilbert path. Please install it.")
        if num_points == 0:
            return []
        
        padded_width = width - 2 * margin
        padded_height = height - 2 * margin
        
        if padded_width <= 0 or padded_height <= 0:
            return []

        draw_area_size = min(padded_width, padded_height)
        x_offset = margin + (padded_width - draw_area_size) / 2
        y_offset = margin + (padded_height - draw_area_size) / 2

        p = math.ceil(math.log2(num_points) / 2) if num_points > 0 else 0
        n = 2
        hilbert_curve = HilbertCurve(p, n)

        points = []
        side_length = 2**p
        scale = draw_area_size / (side_length - 1) if side_length > 1 else draw_area_size
        
        for i in range(num_points):
            coords = hilbert_curve.point_from_distance(i)
            x = x_offset + coords[0] * scale
            y = y_offset + coords[1] * scale
            points.append((x, y))
        return points


class _IntegerGridPathGenerator(PathGenerator):
    """
    Abstract base class for path generators that operate on an integer grid
    and then scale the points to the desired float dimensions.
    """

    @abc.abstractmethod
    def _generate_integer_points(self, width: int, height: int, num_points: int) -> List[Tuple[int, int]]:
        """
        Generates a list of (x, y) integer coordinates.
        :param width: The width of the integer grid.
        :param height: The height of the integer grid.
        :param num_points: The number of points to generate.
        :return: A list of (x, y) integer tuples.
        """
        pass

    def generate(self, width: float, height: float, num_points: int, margin: float = 0.0) -> List[Tuple[float, float]]:
        if num_points == 0:
            return []

        if num_points == 1:
            return [(width / 2, height / 2)]

        if height <= 0 or width <= 0:
            return []
        
        padded_width = width - 2 * margin
        padded_height = height - 2 * margin

        if padded_width <= 0 or padded_height <= 0:
            return []

        # Find the smallest w_int, h_int such that w_int * h_int >= num_points
        # and w_int/h_int is close to width/height.
        ratio = width / height if height > 0 else float('inf')
        w_int = 1
        if ratio > 0 and num_points > 0 and ratio != float('inf'):
            w_int = int(math.sqrt(num_points * ratio))
        
        if w_int == 0:
            w_int = 1

        h_int = (num_points + w_int - 1) // w_int
        
        # Seed randomness for reproducible paths based on dimensions
        random.seed(f"{width}-{height}")

        coords = self._generate_integer_points(w_int, h_int, num_points)

        if not coords:
            return []

        # Find bounds of generated integer coordinates to scale them properly
        min_x = min(p[0] for p in coords)
        max_x = max(p[0] for p in coords)
        min_y = min(p[1] for p in coords)
        max_y = max(p[1] for p in coords)

        int_width = max_x - min_x
        int_height = max_y - min_y

        scaled_points = []
        for x, y in coords:
            # Shift to origin
            shifted_x = x - min_x
            shifted_y = y - min_y

            # Scale points to fit the padded area
            # Handle single-point wide/high cases by centering
            if int_width == 0:
                scaled_x = width / 2
            else:
                scaled_x = margin + ((shifted_x + 0.5) / (int_width + 1)) * padded_width

            if int_height == 0:
                scaled_y = height / 2
            else:
                scaled_y = margin + ((shifted_y + 0.5) / (int_height + 1)) * padded_height
            
            scaled_points.append((scaled_x, scaled_y))

        return scaled_points[:num_points]


class RectangularHilbertPathGenerator(_IntegerGridPathGenerator):
    """
    Generates points along a generalized Hilbert curve for rectangular domains.
    Adapted from: https://github.com/jakubcerveny/gilbert
    """
    def _generate_integer_points(self, width: int, height: int, num_points: int) -> List[Tuple[int, int]]:
        coords = self._gilbert2d(width, height)
        return coords[:num_points]

    def _sign(self, n: int) -> int:
        if n > 0: return 1
        if n < 0: return -1
        return 0

    def _gilbert2d(self, width: int, height: int) -> List[Tuple[int, int]]:
        coords: List[Tuple[int, int]] = []
        if width >= height:
            self._gilbert2d_recursive(0, 0, width, 0, 0, height, coords)
        else:
            self._gilbert2d_recursive(0, 0, 0, height, width, 0, coords)
        return coords

    def _gilbert2d_recursive(self, x: int, y: int, ax: int, ay: int, bx: int, by: int, coords: List[Tuple[int, int]]):
        w = abs(ax + ay)
        h = abs(bx + by)

        dax = self._sign(ax)
        day = self._sign(ay)
        dbx = self._sign(bx)
        dby = self._sign(by)

        if h == 1:
            for i in range(w):
                coords.append((x + i * dax, y + i * day))
            return

        if w == 1:
            for i in range(h):
                coords.append((x + i * dbx, y + i * dby))
            return

        ax2 = ax // 2
        ay2 = ay // 2
        bx2 = bx // 2
        by2 = by // 2

        w2 = abs(ax2 + ay2)
        h2 = abs(bx2 + by2)

        if 2 * w > 3 * h:
            if (w2 % 2) and (w > 2):
                ax2 += dax
                ay2 += day
            
            self._gilbert2d_recursive(x, y, ax2, ay2, bx, by, coords)
            self._gilbert2d_recursive(x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by, coords)
        else:
            if (h2 % 2) and (h > 2):
                bx2 += dbx
                by2 += dby

            self._gilbert2d_recursive(x, y, bx2, by2, ax2, ay2, coords)
            self._gilbert2d_recursive(x + bx2, y + by2, ax, ay, bx - bx2, by - by2, coords)
            self._gilbert2d_recursive(x + (ax - dax) + (bx2 - dbx), y + (ay - day) + (by2 - dby),
                                 -bx2, -by2, -(ax - ax2), -(ay - ay2), coords)


class RandomWalkPathGenerator(_IntegerGridPathGenerator):
    """
    Generates points by performing a self-avoiding random walk on an integer grid.
    If the walk gets stuck, it jumps to a random unvisited point and continues.
    """
    def _generate_integer_points(self, width: int, height: int, num_points: int) -> List[Tuple[int, int]]:
        if width <= 0 or height <= 0:
            return []

        points = []
        occupied = set()

        possible_points = [(x, y) for x in range(width) for y in range(height)]
        if not possible_points:
            return []
        
        current_pos = random.choice(possible_points)
        points.append(current_pos)
        occupied.add(current_pos)

        while len(points) < num_points:
            x, y = current_pos
            neighbors = [(x + dx, y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
            
            valid_neighbors = [
                (nx, ny) for nx, ny in neighbors
                if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in occupied
            ]

            if valid_neighbors:
                current_pos = random.choice(valid_neighbors)
                points.append(current_pos)
                occupied.add(current_pos)
            else:
                remaining_points = [p for p in possible_points if p not in occupied]
                if not remaining_points:
                    break
                current_pos = random.choice(remaining_points)
                if current_pos in occupied:
                    continue
                points.append(current_pos)
                occupied.add(current_pos)

        return points


PATH_GENERATORS = {
    "snake": SnakePathGenerator,
    "hilbert": HilbertPathGenerator,
    "gilbert": RectangularHilbertPathGenerator,
    "random_walk": RandomWalkPathGenerator,
} 
