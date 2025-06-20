import abc
import html
import re

from .paths import PATH_GENERATORS


class Encoder(abc.ABC):
    """Abstract base class for data encoders."""

    @abc.abstractmethod
    def encode(self, data: str, width: float, height: float) -> str:
        """
        Encodes data into an SVG string.

        :param data: The data to encode.
        :param width: The width of the output area in mm.
        :param height: The height of the output area in mm.
        :return: An SVG string.
        """
        pass


class FitTextEncoder(Encoder):
    """Encodes data by fitting text onto a rectangle."""

    def encode(self, data: str, width: float, height: float) -> str:
        """
        Fits text into the given width and height.
        Each line of input data becomes a line of text.
        """
        escaped_data = html.escape(data)
        lines = escaped_data.strip().split('\n')
        num_lines = len(lines)

        if num_lines == 0:
            return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}mm" height="{height}mm" viewBox="0 0 {width} {height}"></svg>'

        line_height = height / num_lines
        font_size = line_height * 0.8  # Use 80% of the line height for the font.

        text_elements = ""
        for i, line in enumerate(lines):
            y = (i + 0.5) * line_height
            text_elements += f'  <text x="0" y="{y}" font-size="{font_size}" textLength="{width}" lengthAdjust="spacingAndGlyphs">{line}</text>\n'

        svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg"
     width="{width}mm"
     height="{height}mm"
     viewBox="0 0 {width} {height}">
  <style>
    text {{
      font-family: monospace;
      dominant-baseline: middle;
      text-anchor: start;
      fill: black;
    }}
  </style>
{text_elements}
</svg>
"""
        return svg_content


class BitsToCirclesEncoder(Encoder):
    """Encodes a bit string into circles along a specified path."""

    def __init__(self, path_type: str = "snake"):
        if path_type not in PATH_GENERATORS:
            raise ValueError(f"Unknown path type: {path_type}. Available: {list(PATH_GENERATORS.keys())}")
        self.path_generator = PATH_GENERATORS[path_type]()

    def encode(self, data: str, width: float, height: float) -> str:
        """
        Encodes a string's bits as circles on a path.
        """
        bits = "".join(re.findall(r"[01]", data))
        num_bits = len(bits)

        if num_bits == 0:
            return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}mm" height="{height}mm" viewBox="0 0 {width} {height}"></svg>'

        margin = min(width, height) * 0.05  # 5% margin
        points = self.path_generator.generate(width, height, num_bits, margin=margin)

        if not points:
            return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}mm" height="{height}mm" viewBox="0 0 {width} {height}"></svg>'

        min_dist = float('inf')
        if len(points) > 1:
            for i in range(len(points) - 1):
                p1 = points[i]
                p2 = points[i+1]
                dist = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
                if dist > 0:
                    min_dist = min(min_dist, dist)
        
        radius = min_dist / 4 if len(points) > 1 and min_dist != float('inf') else min(width, height) / 4
        stroke_width = radius * 0.2
        
        path_data = "M " + " L ".join(f"{p[0]},{p[1]}" for p in points)
        path_element = f'<path d="{path_data}" fill="none" stroke="black" stroke-width="{stroke_width}" />'

        circle_elements = ""
        for i, bit in enumerate(bits):
            x, y = points[i]
            if bit == '1':
                circle_elements += f'  <circle cx="{x}" cy="{y}" r="{radius}" fill="black" />\n'
            else:
                circle_elements += f'  <circle cx="{x}" cy="{y}" r="{radius}" fill="white" stroke="black" stroke-width="{stroke_width}" />\n'

        svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg"
     width="{width}mm"
     height="{height}mm"
     viewBox="0 0 {width} {height}">
{path_element}
{circle_elements}
</svg>
"""
        return svg_content


ENCODERS = {
    "fit_text": FitTextEncoder,
    "bits_to_circles": BitsToCirclesEncoder,
} 