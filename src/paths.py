import abc
import math
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

        cols = int(math.sqrt(num_points * padded_width / padded_height))
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


class RectangularHilbertPathGenerator(PathGenerator):
    """
    Generates points along a generalized Hilbert curve for rectangular domains.
    Adapted from: https://github.com/jakubcerveny/gilbert
    """
    def generate(self, width: float, height: float, num_points: int, margin: float = 0.0) -> List[Tuple[float, float]]:
        
        # The underlying algorithm works with integer coordinates, so we'll scale to our float dimensions at the end.
        # To preserve the path's integrity, we need to determine the smallest integer grid that can contain `num_points`.
        
        # Find the smallest w_int, h_int such that w_int * h_int >= num_points
        # and w_int/h_int is close to width/height.
        if height <= 0 or width <=0:
            return []
            
        ratio = width / height
        w_int = int(math.sqrt(num_points * ratio))
        h_int = 0
        if w_int > 0:
            h_int = num_points // w_int
        
        while w_int > 0 and w_int * h_int < num_points:
            h_int +=1

        if w_int == 0: # handle num_points > 0 and width being very small
             w_int=1
             h_int=num_points

        coords = self._gilbert2d(w_int, h_int)
        
        if not coords:
            return []

        # Find bounds of generated integer coordinates to scale them properly
        min_x = min(p[0] for p in coords)
        max_x = max(p[0] for p in coords)
        min_y = min(p[1] for p in coords)
        max_y = max(p[1] for p in coords)

        int_width = max_x - min_x + 1
        int_height = max_y - min_y + 1

        # Scale and translate to fit the padded drawing area
        padded_width = width - 2 * margin
        padded_height = height - 2 * margin

        scaled_points = []
        for x, y in coords:
            # Shift to origin
            shifted_x = x - min_x + 0.5
            shifted_y = y - min_y + 0.5
            
            # Scale points to fit the padded area
            scaled_x = margin + (shifted_x / int_width) * padded_width if int_width > 1 else width / 2
            scaled_y = margin + (shifted_y / int_height) * padded_height if int_height > 1 else height / 2
            scaled_points.append((scaled_x, scaled_y))

        return scaled_points[:num_points]


    def _sign(self, n):
        if n > 0: return 1
        if n < 0: return -1
        return 0

    def _gilbert2d(self, width, height):
        coords = []
        if width >= height:
            self._gilbert2d_recursive(0, 0, width, 0, 0, height, coords)
        else:
            self._gilbert2d_recursive(0, 0, 0, height, width, 0, coords)
        return coords

    def _gilbert2d_recursive(self, x, y, ax, ay, bx, by, coords):
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


PATH_GENERATORS = {
    "snake": SnakePathGenerator,
    "hilbert": HilbertPathGenerator,
    "gilbert": RectangularHilbertPathGenerator,
} 
