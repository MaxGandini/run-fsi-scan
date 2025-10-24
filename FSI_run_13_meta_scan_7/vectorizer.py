from init import Cursor, MainGrid, Grid3D
from parameters import STRICTNESS

def main_vectorizer(cluster_centers,corr_stack_list,inferred_parameters):
    print("running vectorizer")

    list_of_grids = []

    for ((i, (image, centers_per_image)), image_parameters) in zip(
    enumerate(zip(corr_stack_list, cluster_centers)),
    inferred_parameters):

        widths, median_intensity = image_parameters

        threshold = 130
        cursor = Cursor(threshold)
        grid = MainGrid()

        grid.create_crosses_from_points(centers_per_image)

        if len(grid.crosses) == 0:
            print(f"  Skipping grid {i} because no points were found")
            continue

        grid.compute_neighbors()
        grid.generate_connections(cursor, image)

        list_of_grids.append(grid)

    grid3d = Grid3D(list_of_grids)

    grid3d.project_cross_layers()

    grid3d.align_connection_endpoints_to_top_star()

    geojson_vectors = grid3d.get_connections_geojson()

    return geojson_vectors

