import abc
import math
import random
import sys
from typing import List, Tuple

try:
    from hilbertcurve.hilbertcurve import HilbertCurve
    HILBERT_AVAILABLE = True
except ImportError:
    HILBERT_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


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
        
        if ratio > 0 and num_points > 0 and ratio != float('inf'):
            # Calculate optimal dimensions based on ratio
            # We want h_int to span full height, so calculate w_int accordingly
            h_int = int(math.sqrt(num_points / ratio))
            if h_int == 0:
                h_int = 1
            
            w_int = (num_points + h_int - 1) // h_int
            
            # Ensure we have enough points
            while w_int * h_int < num_points:
                h_int += 1
                w_int = (num_points + h_int - 1) // h_int
        else:
            # Fallback for edge cases
            w_int = int(math.sqrt(num_points))
            if w_int == 0:
                w_int = 1
            h_int = (num_points + w_int - 1) // w_int

        print(f"Generating {num_points} points in {w_int}x{h_int} grid", file=sys.stderr)
        
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

        point_spacing = min(padded_width / (int_width + 1), padded_height / (int_height + 1))

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
                scaled_x = margin + (shifted_x + 0.5) * point_spacing

            if int_height == 0:
                scaled_y = height / 2
            else:
                centering_margin_y = (padded_height - int_height * point_spacing) / 2
                scaled_y = margin + centering_margin_y + (shifted_y) * point_spacing
            
            scaled_points.append((scaled_x, scaled_y))

        if len(scaled_points) < num_points:
            raise ValueError(f"Generated {len(scaled_points)} points, but {num_points} were requested")

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


class WFCPathGenerator(_IntegerGridPathGenerator):
    """
    Generates a seamless path by filling 4x4 subgrids in an order determined
    by a Rectangular Hilbert Curve. A recursive, multi-subgrid backtracking solver
    ensures each subgrid's path ends at a point that borders the next subgrid in the sequence,
    guaranteeing a continuous path. Diagonal connections across subgrid boundaries
    only allow non-sharp outgoing connections.
    """
    subgrid_size = 4

    def _get_subgrid_bounds(self, sg_coords, width, height):
        if sg_coords is None:
            return None
        sg_x, sg_y = sg_coords
        sg_min_x = sg_x * self.subgrid_size
        sg_min_y = sg_y * self.subgrid_size
        sg_max_x = min(sg_min_x + self.subgrid_size, width)
        sg_max_y = min(sg_min_y + self.subgrid_size, height)
        return (sg_min_x, sg_min_y, sg_max_x, sg_max_y)

    def _has_unoccupied_neighbor_in_bounds(self, point, bounds, occupied, width, height):
        if not bounds:
            return False
        x, y = point
        b_min_x, b_min_y, b_max_x, b_max_y = bounds
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if (nx, ny) not in occupied and b_min_x <= nx < b_max_x and b_min_y <= ny < b_max_y:
                    return True
        return False

    def _solve_subgrid(self, bounds, start_pos, target_len, occupied, next_sg_bounds, width, height, entry_dir=None):
        """
        Backtracking solver for a single subgrid.
        entry_dir: (dx, dy) if entering from a diagonal, else None.
        """
        path = [start_pos]
        occupied_local = set(occupied)
        occupied_local.add(start_pos)
        return self._solve_subgrid_recursive(bounds, path, occupied_local, target_len, next_sg_bounds, width, height, entry_dir)

    def _solve_subgrid_recursive(self, bounds, path, occupied, target_len, next_sg_bounds, width, height, entry_dir):
        if len(path) == target_len:
            if next_sg_bounds is None:
                return path.copy()
            last_point = path[-1]
            if self._has_unoccupied_neighbor_in_bounds(last_point, next_sg_bounds, occupied, width, height):
                return path.copy()
            return []

        sg_min_x, sg_min_y, sg_max_x, sg_max_y = bounds
        x, y = path[-1]
        # Only orthogonal moves inside subgrid
        neighbors = [(x + dx, y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
        random.shuffle(neighbors)

        for nx, ny in neighbors:
            if not (sg_min_x <= nx < sg_max_x and sg_min_y <= ny < sg_max_y and (nx, ny) not in occupied):
                continue
            # If this is the first move in the subgrid and entry_dir is set (diagonal entry), enforce no sharp turn
            if len(path) == 1 and entry_dir is not None:
                move_dir = (nx - x, ny - y)
                # Only allow if move_dir is not sharp compared to entry_dir
                if move_dir[0] != entry_dir[0] and move_dir[1] != entry_dir[1]:
                    continue
            path.append((nx, ny))
            occupied.add((nx, ny))
            solution = self._solve_subgrid_recursive(bounds, path, occupied, target_len, next_sg_bounds, width, height, None)
            if solution:
                return solution
            occupied.remove(path.pop())
        return []

    def _find_connection_point(self, last_point, next_sg_bounds, occupied, width, height):
        x, y = last_point
        b_min_x, b_min_y, b_max_x, b_max_y = next_sg_bounds
        # Try all 8 directions
        candidates = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if (nx, ny) not in occupied and b_min_x <= nx < b_max_x and b_min_y <= ny < b_max_y:
                    candidates.append(((nx, ny), (dx, dy)))
        random.shuffle(candidates)
        return candidates

    def _solve_all_subgrids(self, subgrid_order, bounds_list, width, height, num_points, pbar, path, occupied, sg_idx, entry_dir=None):
        if len(path) >= num_points or sg_idx >= len(subgrid_order):
            return path[:num_points] if len(path) >= num_points else []
        current_bounds = bounds_list[sg_idx]
        next_bounds = bounds_list[sg_idx + 1] if sg_idx + 1 < len(bounds_list) else None
        points_in_subgrid_now = {p for p in occupied if current_bounds[0] <= p[0] < current_bounds[2] and current_bounds[1] <= p[1] < current_bounds[3]}
        subgrid_capacity = (current_bounds[2] - current_bounds[0]) * (current_bounds[3] - current_bounds[1])
        points_to_generate = min(num_points - len(path), subgrid_capacity - len(points_in_subgrid_now))
        if points_to_generate <= 0:
            # Move to next subgrid
            return self._solve_all_subgrids(subgrid_order, bounds_list, width, height, num_points, pbar, path, occupied, sg_idx + 1, None)
        target_len_in_sg = len(points_in_subgrid_now) + points_to_generate
        start_pos = path[-1]
        # Try all possible valid subgrid fillings and connections
        sub_path_local = self._solve_subgrid(current_bounds, start_pos, target_len_in_sg, occupied, next_bounds, width, height, entry_dir)
        if not sub_path_local:
            return []
        newly_added = [p for p in sub_path_local if p not in occupied]
        path.extend(newly_added)
        occupied.update(newly_added)
        if pbar:
            pbar.n = len(path)
            pbar.refresh()
        if next_bounds and len(path) < num_points:
            # Try all possible valid connections
            for (connection, dir_vec) in self._find_connection_point(path[-1], next_bounds, occupied, width, height):
                path.append(connection)
                occupied.add(connection)
                if pbar: pbar.update(1)
                # Recurse into next subgrid, enforcing the diagonal connection rule
                result = self._solve_all_subgrids(subgrid_order, bounds_list, width, height, num_points, pbar, path, occupied, sg_idx + 1, dir_vec if dir_vec[0] != 0 and dir_vec[1] != 0 else None)
                if result:
                    return result
                # Backtrack connection
                occupied.remove(path.pop())
            # If no connection worked, backtrack subgrid
            for p in newly_added[::-1]:
                occupied.remove(path.pop())
            return []
        else:
            # No more subgrids, done
            return path[:num_points]

    def _generate_integer_points(self, width: int, height: int, num_points: int) -> List[Tuple[int, int]]:
        if width <= 0 or height <= 0 or num_points == 0:
            return []
        if num_points > width * height:
            return []
        sg_width = (width + self.subgrid_size - 1) // self.subgrid_size
        sg_height = (height + self.subgrid_size - 1) // self.subgrid_size
        if sg_height * sg_width > 1:
            subgrid_order = self._generate_integer_points(sg_width, sg_height, sg_width * sg_height)
        else:
            subgrid_order = [(0, 0)]

        bounds_list = [self._get_subgrid_bounds(sg, width, height) for sg in subgrid_order]
        path = [(0, 0)]
        occupied = {(0, 0)}
        pbar = None
        if TQDM_AVAILABLE:
            pbar = tqdm(total=num_points, desc="Generating WFC path", file=sys.stderr, initial=1, leave=False)
        try:
            result = self._solve_all_subgrids(subgrid_order, bounds_list, width, height, num_points, pbar, path, occupied, 0, None)
        finally:
            if pbar:
                pbar.close()
        return result


PATH_GENERATORS = {
    "snake": SnakePathGenerator,
    "hilbert": HilbertPathGenerator,
    "gilbert": RectangularHilbertPathGenerator,
    "wfc": WFCPathGenerator,
} 
