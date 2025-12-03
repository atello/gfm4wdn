#
# Created on Thu Oct 17 2024
# Copyright (c) 2024 Huy Truong
# ------------------------------
# Purpose: Support create terrain for any wn
# ------------------------------
#


import matplotlib.pyplot as plt
import wntr
import numpy as np
import networkx as nx
import pandas as pd


def create_ds_terrain(heightmapWidth: int = 65, zrange: tuple[float, float] = (0, 256.0), randomness: float = 128) -> np.ndarray:
    # modified from: https://learn.64bitdragon.com/articles/computer-science/procedural-generation/the-diamond-square-algorithm

    assert heightmapWidth % 2 != 0, f"ERROR! expect heightmapWidth is an odd value, but get {heightmapWidth}"
    heightmap = np.zeros([heightmapWidth, heightmapWidth], dtype=float)
    rand = zrange[0] + np.random.rand() * (zrange[1] - zrange[0])
    heightmap[0, 0] = rand
    heightmap[heightmapWidth - 1, 0] = rand
    heightmap[0, heightmapWidth - 1] = rand
    heightmap[heightmapWidth - 1, heightmapWidth - 1] = rand

    # set the randomness bounds, higher values mean rougher landscapes
    tileWidth = heightmapWidth - 1
    # we make a pass over the heightmap
    # each time we decrease the side length by 2
    while tileWidth > 1:
        halfSide = tileWidth // 2

        random_list = -randomness + np.random.rand((heightmapWidth - 1) // tileWidth, (heightmapWidth - 1) // tileWidth) * (
            randomness - (-randomness)
        )
        random_list = random_list.flatten().tolist()

        # set the diamond values (the centers of each tile)
        for x in range(0, heightmapWidth - 1, tileWidth):
            for y in range(0, heightmapWidth - 1, tileWidth):
                cornerSum = (
                    heightmap[x][y]
                    + heightmap[x + tileWidth][y]
                    + heightmap[x][y + tileWidth]
                    + heightmap[x + tileWidth][y + tileWidth]
                )

                avg = cornerSum / 4
                avg += random_list.pop()

                heightmap[x + halfSide][y + halfSide] = avg

        random_list = (
            -randomness
            + np.random.rand(heightmapWidth - 1 // halfSide, heightmapWidth - 1 // tileWidth) * randomness
            - (-randomness)
        )
        random_list = random_list.flatten().tolist()

        # set the square values (the midpoints of the sides)
        for x in range(0, heightmapWidth - 1, halfSide):
            for y in range((x + halfSide) % tileWidth, heightmapWidth - 1, tileWidth):
                avg = (
                    heightmap[(x - halfSide + heightmapWidth - 1) % (heightmapWidth - 1), y]
                    + heightmap[(x + halfSide) % (heightmapWidth - 1), y]
                    + heightmap[x, (y + halfSide) % (heightmapWidth - 1)]
                    + heightmap[x, (y - halfSide + heightmapWidth - 1) % (heightmapWidth - 1)]
                )

                avg /= 4.0
                avg += random_list.pop()
                avg = min(max(avg, zrange[0]), zrange[1])
                heightmap[x][y] = avg

                # because the values wrap round, the left and right edges are equal, same with top and bottom
                if x == 0:
                    heightmap[heightmapWidth - 1, y] = avg
                if y == 0:
                    heightmap[x, heightmapWidth - 1] = avg

        # reduce the randomness in each pass, making sure it never gets to 0
        randomness = max(randomness // 2, 1)
        tileWidth //= 2
    return heightmap


def convert_multigraph_to_single_graph(G_multi):
    G_single = nx.Graph()  # Create a new Graph (not MultiGraph)

    # Iterate over all edges in the MultiGraph
    for u, v in G_multi.edges(keys=False):
        # Get all edges between the two nodes u, v
        edges_between = G_multi.get_edge_data(u, v)

        # Sample one edge randomly (you can apply other sampling strategies)
        sampled_edge_key = np.random.choice(list(edges_between.keys()))

        # Add the sampled edge to the new single Graph
        # You can also add edge attributes like weight if needed
        G_single.add_edge(u, v, **edges_between[sampled_edge_key])

    return G_single


def check_if_coords_exists(nodal_cords: list[tuple[float, float]], fast_check: bool = True) -> bool:
    tmp = nodal_cords[:2] if fast_check else nodal_cords
    tup_of_list = zip(*tmp)
    xs, ys = tup_of_list
    return not (all(xs[0] == x for x in xs) and all(ys[0] == x for x in xs))


def create_diamond_kernel(size: int, val: float = 1.0 / 4) -> np.ndarray:
    kernel = np.zeros([size, size])

    kernel[0, 0] = val
    kernel[size - 1, 0] = val
    kernel[0, size - 1] = val
    kernel[size - 1, size - 1] = val

    return kernel


def create_square_kernel(size: int, val: float = 1.0 / 3) -> np.ndarray:
    kernel = np.zeros([size, size])
    mid_point = size // 2

    kernel[0, 0] = val
    kernel[size - 1, 0] = val
    kernel[mid_point, mid_point] = val
    kernel[0, size - 1] = val
    kernel[size - 1, size - 1] = val

    return kernel


def plot_3d_scatter(
    some_z: np.ndarray,
    map: np.ndarray | None = None,
    lower_map_value: float = 0,
    zrange: tuple[float, float] | None = None,
):
    # show height map in 3d
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False  # type:ignore
    if map is not None:
        lower_map = map + np.full_like(map, lower_map_value)
        x, y = np.meshgrid(range(lower_map.shape[0]), range(lower_map.shape[1]))
        ax.plot_surface(x, y, lower_map, alpha=0.5, vmin=0)  # type:ignore
    # ax.scatter(some_z[:, 0], some_z[:, 1], some_z[:, 2], c=some_z[:, 2], cmap="hot")
    ax.scatter(some_z[:, 1], some_z[:, 0], some_z[:, 2], c=some_z[:, 2], cmap="viridis_r", vmin=0)

    if zrange is not None:
        ax.set_zlim(zrange[0], zrange[1])  # type:ignore

    zmin, zmax = np.min(some_z[:, -1]), np.max(some_z[:, -1])
    plt.title(f"zmin-{zmin} zmax-{zmax}")
    plt.show()


def normalize(x: np.ndarray, axis: int | None = 0, eps: float = 1e-8) -> np.ndarray:
    min_x, max_x = np.min(x, axis=axis), np.max(x, axis=axis)
    norm_x = (x - min_x) / (max_x - min_x + eps)
    return norm_x


def continuous_to_discrete(
    x: float, y: float, N: int, x_min: float = 0, x_max: float = 1.0, y_min: float = 0.0, y_max: float = 1.0
) -> tuple[int, int]:
    i = np.floor((x - x_min) * (N - 1) / (x_max - x_min)).astype(int)
    j = np.floor((y - y_min) * (N - 1) / (y_max - y_min)).astype(int)
    i = np.clip(i, 0, N - 1)  # Clamp to valid range
    j = np.clip(j, 0, N - 1)
    return i, j


def generate_elevation(
    node_names: list[str],
    wn: wntr.network.WaterNetworkModel,
    height_map_width: int = 129,
    zrange: tuple[float, float] | None = None,
    randomness: float = 0.1,
    verbose: bool = True,
    plot_fig: bool = False,
) -> np.ndarray:
    """Create a 2d map using diamond-square algorithm, then extract elevation from the map according to the nodal coords.
    If cooordinates are unavailable, we generate one using spring layout from networkx.

    Args:
        node_names (list[str]): list of node names in wn
        wn (wntr.network.WaterNetworkModel): water network model
        height_map_width (int, optional): map length. Bigger map is more sophisicated. Defaults to 129.
        zrange (tuple[float, float], optional): elevation bounds. If none, auto-pick the elevation range from wn. Defaults to None.
        randomness (float, optional): More noise, more mountain-like. Defaults to 0.1.
        verbose (bool, optional): for debug. Defaults to True.
        plot_fig (bool, optional): for visualization. Defaults to False.

    Returns:
        np.ndarray: _description_
    """
    if zrange is None:
        try:
            ele_series: pd.Series = wn.query_node_attribute("elevation")
            zrange = (float(ele_series.min()), float(ele_series.max()))  # type:ignore
        except Exception:
            zrange = (0, 256)

    try:
        node_coords: list[tuple[float, float]] = [tuple(wn.get_node(name).coordinates) for name in node_names]  # type: ignore
    except Exception:
        if verbose:
            print("WARN! coordinates cannot extracted in the wn! Use empty coordinate instead!")
        node_coords: list[tuple[float, float]] = [(0.0, 0.0) for name in node_names]

    if not check_if_coords_exists(node_coords, fast_check=True):
        g: nx.Graph = convert_multigraph_to_single_graph(wn.to_graph().to_undirected())
        pos = nx.spring_layout(g)
        node_coords = [tuple(pos[name].tolist()) for name in wn.junction_name_list]

    # first convert to numpy
    np_coords: np.ndarray = np.asarray(node_coords, dtype=float)  # type: ignore
    # normalize the coords
    normed_coords = normalize(np_coords, axis=0)
    zmap = create_ds_terrain(height_map_width, zrange=zrange, randomness=randomness)

    new_coords = np.apply_along_axis(lambda p: continuous_to_discrete(p[0], p[1], height_map_width), axis=1, arr=normed_coords)

    xs, ys = np.expand_dims(new_coords[:, 0].astype(int), axis=1), np.expand_dims(new_coords[:, 1].astype(int), axis=1)
    elevations = [zmap[xs[i], ys[i]] for i in range(xs.shape[0])]  # zmap[xs, ys]
    elevations = np.asarray(elevations).reshape(-1, 1)

    if plot_fig:
        pos = np.concatenate([xs, ys, elevations], axis=1)
        plot_3d_scatter(pos, zmap, zrange=zrange)
        plt.show()

    return elevations


if __name__ == "__main__":
    wn = wntr.network.WaterNetworkModel(inp_file_name=r"gigantic_dataset/inputs/public/Anytown.inp")

    for randomness in [4, 20, 128]:
        generate_elevation(
            node_names=wn.junction_name_list,
            wn=wn,
            height_map_width=17,
            zrange=None,
            randomness=randomness,
            verbose=True,
            plot_fig=True,
        )
