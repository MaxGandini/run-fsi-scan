import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union
from parameters import STRICTNESS
from utilities import read_order_img

def plot_cutoff_differences(
    diffs: Union[np.ndarray, List[Tuple[float, float]]],
    projection_folder: Union[str, Path] = None,
    save_name: str = "cutoff_differences.png",
    show_axes: bool = True,
    marker_kw: dict = None,
    bins: int = 20,
):
    """
    Scatter plot Δx vs Δy + histograms of Δx and Δy separately.
    """

    # flatten input and filter only valid (dx, dy)
    flat = []
    if isinstance(diffs, list):
        for item in diffs:
            if item is None:
                continue
            # handle nested lists/tuples
            if isinstance(item, (list, tuple)):
                for sub in item:
                    if isinstance(sub, (list, tuple, np.ndarray)) and len(sub) == 2:
                        flat.append(sub)
            elif isinstance(item, (list, tuple, np.ndarray)) and len(item) == 2:
                flat.append(item)
    else:
        flat = diffs

    # convert to ndarray
    diffs_arr = np.array(flat, dtype=float)
    if diffs_arr.size == 0:
        diffs_arr = np.empty((0, 2))
    elif diffs_arr.ndim == 1 and diffs_arr.size == 2:
        diffs_arr = diffs_arr.reshape(1, 2)
    elif diffs_arr.shape[1] != 2:
        # remove any rows that don't have exactly 2 elements
        diffs_arr = np.array([row for row in diffs_arr if len(row) == 2], dtype=float)

    if diffs_arr.size == 0:
        return diffs_arr

    dx = diffs_arr[:, 0]
    dy = diffs_arr[:, 1]

    # scatter
    plt.figure(figsize=(6, 6))
    mk = {"c": "red", "marker": "o", "s": 30, "alpha": 0.8}
    if marker_kw:
        mk.update(marker_kw)
    plt.scatter(dx, dy, **mk)
    if show_axes:
        plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        plt.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("Δx (width difference in x)")
    plt.ylabel("Δy (width difference in y)")
    plt.title("Cutoff width differences (Δx vs Δy)")
    plt.tight_layout()

    if projection_folder:
        p = Path(projection_folder)
        p.mkdir(parents=True, exist_ok=True)
        plt.savefig(p / save_name, dpi=150)
        plt.close()
    else:
        plt.show()

    # histogram Δx
    plt.figure(figsize=(6, 4))
    plt.hist(dx, bins=bins, color="blue", alpha=0.7, edgecolor="black")
    plt.xlabel("Δx (width difference in x)")
    plt.ylabel("Count")
    plt.title("Histogram of Δx differences")
    plt.tight_layout()
    if projection_folder:
        plt.savefig(Path(projection_folder) / (Path(save_name).stem + "_hist_x.png"), dpi=150)
        plt.close()
    else:
        plt.show()

    # histogram Δy
    plt.figure(figsize=(6, 4))
    plt.hist(dy, bins=bins, color="green", alpha=0.7, edgecolor="black")
    plt.xlabel("Δy (width difference in y)")
    plt.ylabel("Count")
    plt.title("Histogram of Δy differences")
    plt.tight_layout()
    if projection_folder:
        plt.savefig(Path(projection_folder) / (Path(save_name).stem + "_hist_y.png"), dpi=150)
        plt.close()
    else:
        plt.show()

    return diffs_arr

def plot_dark_zero_lines_on_image(
    image,
    h_cutoffs=None,
    v_cutoffs=None,
    projection_folder=None,
    image_name="projection_with_dark_zeros",
    idx=0,
    dark_percentile=None,
    n_iter=0,
):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_rgb = image.copy()
    else:
        gray = image.copy()
        img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    h_values = []
    v_values = []

    # Horizontal
    if h_cutoffs is not None:
        for entry in h_cutoffs:
            zero = entry[1] if isinstance(entry, (list, tuple)) and len(entry) == 3 else entry
            y = int(zero[0])
            if 0 <= y < gray.shape[0]:
                mean_intensity = np.mean(gray[y, :])
                h_values.append((y, mean_intensity))

    # Vertical
    if v_cutoffs is not None:
        for entry in v_cutoffs:
            zero = entry[1] if isinstance(entry, (list, tuple)) and len(entry) == 3 else entry
            x = int(zero[0])
            if 0 <= x < gray.shape[1]:
                mean_intensity = np.mean(gray[:, x])
                v_values.append((x, mean_intensity))

    if h_values:
        h_threshold = np.percentile([v for _, v in h_values], (dark_percentile or 0.2) * 100)
        h_filtered = [y for y, v in h_values if v <= h_threshold]
    else:
        h_filtered = []

    if v_values:
        v_threshold = np.percentile([v for _, v in v_values], (dark_percentile or 0.2) * 100)
        v_filtered = [x for x, v in v_values if v <= v_threshold]
    else:
        v_filtered = []

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_rgb, cmap="gray")

    for y in h_filtered:
        ax.axhline(y=y, color="red", linestyle="--", linewidth=1)

    for x in v_filtered:
        ax.axvline(x=x, color="blue", linestyle="--", linewidth=1)

    ax.set_title(f"Darkest {(dark_percentile or 0.2)*100:.1f}% Zero Lines")
    ax.axis("off")

    if projection_folder is not None:
        iter_folder = Path(projection_folder) / f"iteration_{n_iter:02d}"
        iter_folder.mkdir(parents=True, exist_ok=True)
        plt.savefig(iter_folder / f"{image_name}_{idx}_darkzeros.png", bbox_inches="tight")

    plt.close(fig)

def plot_projection_results(
    h_profile, v_profile, h_derivative, v_derivative,
    x_line, y_line, projection_folder, image_name="projection",
    h_cutoffs=None, v_cutoffs=None, darkest_percentage=STRICTNESS
):
    """
    Plots projection profiles and their derivatives.
    Optionally overlays zero-crossings and cutoff coordinates (x, y).
    Optionally includes percentage of darkest targets in plot titles.

    Parameters:
        h_profile, v_profile: 1D projection arrays
        h_derivative, v_derivative: derivative arrays
        x_line, y_line: reference coordinates
        projection_folder: Path to save figures
        image_name: filename prefix
        h_cutoffs, v_cutoffs: list of ((x_left, y_left), (x0, y0), (x_right, y_right)) tuples
        darkest_percentage: float in [0,1], optional, fraction of darkest pixels
    """
    projection_folder.mkdir(parents=True, exist_ok=True)

    # title suffix
    perc_suffix = ""
    if darkest_percentage is not None:
        perc_suffix = f" | Darkest {darkest_percentage*100:.1f}%"

    # Horizontal projection
    (projection_folder / "horizontal_proj_gray").mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(h_profile, color='black')
    plt.title(f"Horizontal Projection (Y={y_line}){perc_suffix}")
    plt.xlabel("Y coordinate")
    plt.ylabel("Weighted Sum (Gray)")
    plt.tight_layout()
    plt.savefig(projection_folder / "horizontal_proj_gray" / f"{image_name}_horizontal.png")
    plt.close()

    # Vertical projection
    (projection_folder / "vertical_proj_gray").mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(v_profile, color='black')
    plt.title(f"Vertical Projection (X={x_line}){perc_suffix}")
    plt.xlabel("X coordinate")
    plt.ylabel("Weighted Sum (Gray)")
    plt.tight_layout()
    plt.savefig(projection_folder / "vertical_proj_gray" / f"{image_name}_vertical.png")
    plt.close()

    # Horizontal derivative
    (projection_folder / "horizontal_proj_gray_derivative").mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(h_derivative, color='gray')
    if h_cutoffs is not None:
        for left, zero, right in h_cutoffs:
            # width line
            plt.plot([left[0], right[0]], [left[1], right[1]], 'r-', alpha=0.5)
            # zero crossing vertical line
            plt.axvline(x=zero[0], color='red', linestyle='--', alpha=0.6)
            # zero crossing mark
            plt.scatter([zero[0]], [zero[1]], color='red', marker='x', zorder=5)
    plt.title(f"Horizontal Derivative Projection (Y={y_line}){perc_suffix}")
    plt.xlabel("Y coordinate")
    plt.ylabel("d/dy (Gray Projection)")
    plt.tight_layout()
    plt.savefig(projection_folder / "horizontal_proj_gray_derivative" / f"{image_name}_horizontal_deriv.png")
    plt.close()

    # Vertical derivative
    (projection_folder / "vertical_proj_gray_derivative").mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(v_derivative, color='gray')
    if v_cutoffs is not None:
        for left, zero, right in v_cutoffs:
            plt.plot([left[0], right[0]], [left[1], right[1]], 'r-', alpha=0.5)
            plt.axvline(x=zero[0], color='red', linestyle='--', alpha=0.6)
            plt.scatter([zero[0]], [zero[1]], color='red', marker='x', zorder=5)
    plt.title(f"Vertical Derivative Projection (X={x_line}){perc_suffix}")
    plt.xlabel("X coordinate")
    plt.ylabel("d/dx (Gray Projection)")
    plt.tight_layout()
    plt.savefig(projection_folder / "vertical_proj_gray_derivative" / f"{image_name}_vertical_deriv.png")
    plt.close()

def plot_selected_crossings_on_image(
    image,
    crossings,
    projection_folder=None,
    image_name="projection_with_selected_crosses",
    idx=0,
):
    """
    Plot scatter points for preselected crossings (x, y, intensity).

    Parameters:
        image: np.ndarray (grayscale or RGB)
        crossings: list of (x, y, intensity) tuples
        projection_folder: output directory
        image_name: base name for saved image
        idx: index for saving
    """
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Prepare image
    if image.ndim == 3:
        img_rgb = image.copy()
    else:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if not crossings:
        return

    # Extract coordinates and intensities
    xs = np.array([x for x, _, _ in crossings])
    ys = np.array([y for _, y, _ in crossings])
    intensities = np.array([z for _, _, z in crossings])

    # Normalize intensity for colormap scaling
    if len(intensities) > 0:
        norm_intensity = (intensities - np.min(intensities)) / (np.ptp(intensities) + 1e-9)
    else:
        norm_intensity = np.zeros_like(intensities)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_rgb, cmap="gray")
    scatter = ax.scatter(xs, ys, c=norm_intensity, cmap="plasma", s=20, marker="x", linewidths=1)

    ax.set_title("Selected Crossings (Intensity-Colored)")
    ax.axis("off")

    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Relative Intensity")

    if projection_folder is not None:
        Path(projection_folder).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(projection_folder) / f"{image_name}_{idx}_crossings.png", bbox_inches="tight")
    plt.close(fig)

def render_gif_from_folder(folder_path: Path, args,order_func=read_order_img,duration: int = 300, loop: int = 0):

    if not folder_path.exists() or not folder_path.is_dir():
        raise ValueError(f"{folder_path} is not a valid directory.")

    from_img, to_img, prefix = args
    sorted_images = order_func(from_img,to_img,folder_path,prefix)
    if not sorted_images:
        raise ValueError("No images found by order_func.")

    frames = [Image.open(img_path) for img_path in sorted_images]

    output_path = folder_path / f"{folder_path.name}_animated.gif"

    frames[0].save(
        output_path,
        format='GIF',
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop
    )

def plot_image_intensity_vs_depth(dark_medians, whole_medians, save_path, image_idx):
    depths = list(range(1, len(dark_medians) + 1))

    # Compute the difference
    differences = np.array(whole_medians) - np.array(dark_medians)

    plt.figure(figsize=(8, 5))

    plt.plot(depths, dark_medians, label='Dark Median', marker='o')
    plt.plot(depths, whole_medians, label='Whole Image Median', marker='x')

    ax = plt.gca()
    ax2 = ax.twinx()
    ax2.plot(depths, differences, label='Difference (Whole - Dark)', color='tab:red', linestyle='--', marker='s')

    ax.set_xlabel('Correlation Window Depth')
    ax.set_ylabel('Median Intensity')
    ax2.set_ylabel('Intensity Difference')
    plt.title(f'Median Intensities vs Depth (Image {image_idx})')

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')

    ax.grid(True, linestyle='--', alpha=0.6)

    # Save figure
    save_path.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path / f"intensity_vs_depth_{image_idx}.png")
    plt.close()
