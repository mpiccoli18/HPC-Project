import numpy as np
import pyvista as pv


def parse_points(file) -> np.ndarray:
    lines = file.read().strip().splitlines()
    lines = lines[1:]  # skip header

    data = []
    for line in lines:
        x, y, z, label = line.split(',')
        data.append([float(x), float(y), float(z), int(label)])

    return np.array(data)


def plot_points(points: np.ndarray):
    coords = points[:, :3].astype(float)
    labels = points[:, 3].astype(float)

    cloud = pv.PolyData(coords)
    cloud["labels"] = labels
    print(f"Coords shape: {coords.shape}") # Should be (Number, 3)

    sphere_geom = pv.Sphere(radius=0.15) # Adjust radius based on your data scale
    geom_points = cloud.glyph(geom=sphere_geom, scale=False)

    plotter = pv.Plotter(off_screen=True)
    plotter.set_background("white")
    plotter.add_mesh(
        geom_points,
        scalars="labels",
        cmap="jet",
        smooth_shading=True
    )

    plotter.add_axes()
    plotter.camera_set = True
    plotter.reset_camera()

    print(f"Data range: {coords.min(axis=0)} to {coords.max(axis=0)}")
    print(f"Plotter bounds: {plotter.bounds}")

    plotter.screenshot("plot2.png", transparent_background=True)
    plotter.close()


def main():
    from argparse import ArgumentParser

    args = ArgumentParser()
    args.add_argument(
        '-i', '--input-file',
        help='input csv file',
        required=True,
        dest='input_file',
        type=str
    )
    pargs = args.parse_args()

    with open(pargs.input_file, 'r') as file:
        points = parse_points(file)
        print("Opened file: %s", file)
        plot_points(points)

if __name__ == '__main__':
    main()
