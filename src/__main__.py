import argparse
import sys
import math


def get_encoders():
    """Dynamically loads encoder classes from the 'encoders' module."""
    # This allows adding new encoders without changing this file.
    import encoders
    return encoders.ENCODERS

def debug_message(message):
    print(message, file=sys.stderr)


def main():
    encoders = get_encoders()
    parser = argparse.ArgumentParser(description="Encode data onto a ring surface.")
    parser.add_argument("--height", type=float, required=True, help="Height of the ring in mm.")
    parser.add_argument("--diameter", type=float, required=True, help="Diameter of the ring in mm.")
    parser.add_argument("--dpi", type=int, help="Dots per inch for raster output (PNG). If not given, outputs SVG.")
    parser.add_argument("--encoder", type=str, default="fit_text", choices=encoders.keys(),
                        help="The encoding method to use.")

    args = parser.parse_args()

    # Log to stderr
    debug_message(f"Using encoder: {args.encoder}")
    debug_message(f"Ring dimensions: height={args.height}mm, diameter={args.diameter}mm")
    if args.dpi:
        debug_message(f"Output DPI: {args.dpi}")

    # Read data from stdin
    data = sys.stdin.read()
    if not data:
        debug_message("No data from stdin.")
        sys.exit(1)

    circumference = args.diameter * math.pi

    encoder_class = encoders[args.encoder]
    encoder = encoder_class()

    svg_data = encoder.encode(data, width=circumference, height=args.height)

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
