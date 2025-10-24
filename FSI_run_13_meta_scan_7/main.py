import importlib
from plots import render_gif_from_folder
from parameters import TEST_FOLDER, PROJ_ON_IMAGE_FOLDER, CROSSINGS_FOLDER, FROM_IMAGE, TO_IMAGE

import init
import scanner
import binarizer 
import vectorizer
import parameters 
import cv2

def reload_modules():
    for mod in [init, scanner, binarizer, vectorizer,parameters]:
        importlib.reload(mod)

def main():
    reload_modules()
    cluster_centers, filtered_images_list, inferred_parameters, corr_stack_list = scanner.main_scanner()

    # cluster_centers, filtered_images_list, inferred_parameters= binarizer.main_binarizer(corr_stack_list,inferred_parameters)
    out_vectors = vectorizer.main_vectorizer(cluster_centers,corr_stack_list,inferred_parameters)

    for (i, image), filt_image in zip(enumerate(corr_stack_list), filtered_images_list):
        centers = cluster_centers[i]
        img_copy = image.copy()

        if len(filt_image.shape) == 2:
            filt_image_bgr = cv2.cvtColor(filt_image, cv2.COLOR_GRAY2BGR)
        else:
            filt_image_bgr = filt_image.copy()

        for (x, y, z) in centers:
            cv2.circle(filt_image_bgr, (int(x), int(y)), radius=20, color=(255, 0, 0), thickness=20)

        output_path = TEST_FOLDER / "filtered_points" / f"scanned_{i}.png"
        out_filt_path = TEST_FOLDER / "filtered_img" / f"filtered_{i}.png"
        cv2.imwrite(str(output_path), img_copy)
        cv2.imwrite(str(out_filt_path), filt_image_bgr)

        render_gif_from_folder(CROSSINGS_FOLDER,(FROM_IMAGE,TO_IMAGE,"projection_with_selected_crosses_"))
        render_gif_from_folder(TEST_FOLDER / "depths",(FROM_IMAGE,TO_IMAGE,"intensity_vs_depth_"))

    plot_geojson_lines_3d(out_vectors)

import plotly.graph_objects as go
import json

def plot_geojson_lines_3d(geojson_data, output_html_path="plot.html", z_spacing=1.0, show=True):
    """
    Plot GeoJSON lines in 3D with Z scaling and inverted axes like plot_filtered.

    Parameters
    ----------
    geojson_data : dict or str
        GeoJSON dict or path to GeoJSON file.
    output_html_path : str
        Path to save HTML plot.
    z_spacing : float
        Factor to scale Z-axis values.
    show : bool
        Whether to display the figure in browser.
    """
    # Load from file if geojson_data is a path
    if isinstance(geojson_data, str):
        with open(geojson_data, "r") as f:
            geojson_data = json.load(f)

    fig = go.Figure()

    for feature in geojson_data.get("features", []):
        geom_type = feature["geometry"]["type"]
        coords = feature["geometry"]["coordinates"]

        if geom_type == "LineString":
            xs, ys, zs = [], [], []
            for coord in coords:
                if len(coord) == 3:
                    x, y, z = coord
                else:
                    x, y = coord
                    z = 0
                xs.append(-x)
                ys.append(y)
                zs.append(-z * z_spacing)
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=dict(color="green", width=5)
            ))

        elif geom_type == "MultiLineString":
            for line in coords:
                xs, ys, zs = [], [], []
                for coord in line:
                    if len(coord) == 3:
                        x, y, z = coord
                    else:
                        x, y = coord
                        z = 0
                    xs.append(-x)
                    ys.append(y)
                    zs.append(-z * z_spacing)
                fig.add_trace(go.Scatter3d(
                    x=xs, y=ys, z=zs,
                    mode="lines",
                    line=dict(color="green", width=5)
                ))

    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data"  # preserves aspect ratio
        ),
        width=800,
        height=600,
        showlegend=False
    )

    fig.write_html(output_html_path)
    if show:
        fig.show()
    print(f"Plot saved to {output_html_path}")

if __name__ == "__main__":
    main()
