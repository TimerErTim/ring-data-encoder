import argparse
import sys
import math

from . import encoders
from .paths import PATH_GENERATORS


def get_encoders():
    """Dynamically loads encoder classes from the 'encoders' module."""
    # This allows adding new encoders without changing this file.
    return encoders.ENCODERS

def debug_message(message):
    print(message, file=sys.stderr)


def main():
    encoders = get_encoders()
    parser = argparse.ArgumentParser(description="Encode data onto a ring surface.")
    parser.add_argument("--height", type=float, required=True, help="Height of the ring in mm.")
    parser.add_argument("--diameter", type=float, required=True, help="Diameter of the ring in mm.")
    parser.add_argument("--dpi", type=int, help="Dots per inch for raster output (PNG). If not given, outputs SVG.")
    parser.add_argument("--encoder", type=str, default="fit_text", choices=list(encoders.keys()),
                        help="The encoding method to use.")
    parser.add_argument("--path-type", type=str, default="snake", choices=list(PATH_GENERATORS.keys()),
                        help="The path type to use for the 'bits_to_circles' encoder.")

    args = parser.parse_args()

    # Log to stderr
    debug_message(f"Using encoder: {args.encoder}")
    debug_message(f"Ring dimensions: height={args.height}mm, diameter={args.diameter}mm")
    if args.dpi:
        debug_message(f"Output DPI: {args.dpi}")

    # Read data from stdin
    data_bytes = sys.stdin.buffer.read()
    if not data_bytes:
        debug_message("No data from stdin.")
        sys.exit(1)

    circumference = args.diameter * math.pi

    encoder_name = args.encoder
    encoder_class = encoders[encoder_name]

    encoder_args = {'width': circumference, 'height': args.height}

    if encoder_name == 'bits_to_circles':
        try:
            encoder = encoder_class(path_type=args.path_type)
            encoder_args['data'] = "".join(f"{byte:08b}" for byte in data_bytes)
        except RuntimeError as e:
            debug_message(f"Error initializing encoder: {e}")
            sys.exit(1)
    else:
        encoder = encoder_class()
        encoder_args['data'] = data_bytes.decode('utf-8', errors='replace')

    svg_data = encoder.encode(**encoder_args)

    if args.dpi:
        try:
            import cairosvg
        except ImportError:
            debug_message("cairosvg is not installed. Please install it to use --dpi.")
            debug_message("pip install cairosvg")
            sys.exit(1)

        # Calculate output size in pixels
        width_px = int(circumference / 25.4 * args.dpi)
        height_px = int(args.height / 25.4 * args.dpi)

        debug_message(f"Output image size: {width_px}x{height_px} pixels")

        png_data = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'), output_width=width_px, output_height=height_px)
        sys.stdout.buffer.write(png_data)
    else:
        # Output SVG to stdout
        sys.stdout.write(svg_data)


if __name__ == "__main__":
    main()
