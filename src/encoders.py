import abc
import html


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


ENCODERS = {
    "fit_text": FitTextEncoder,
} 