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
    coords = points[:, :3]
    labels = points[:, 3]

    plotter = pv.Plotter()
    plotter.add_points(
        coords,
        scalars=labels,
        render_points_as_spheres=True,
        point_size=6
    )
    plotter.show()


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
        plot_points(points)


if __name__ == '__main__':
    main()
