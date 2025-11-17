import os
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from skimage import exposure


def _center_bbox(
    img: np.ndarray,
    bb_coords: tuple[float, float],
    bb_dimn: tuple[float, float],
    img_dimn: tuple[int, int],
    padding_factor=1.1,
) -> Optional[np.ndarray]:
    """
    Helper function to center a bounding box around a particle with padding.
    """
    x, y = bb_coords
    w, h = bb_dimn
    img_height, img_width = img_dimn

    # check if on image edge
    is_on_left = x == 0
    is_on_top = y == 0
    is_on_right = x + w == img_width
    is_on_bottom = y + h == img_height

    if is_on_left or is_on_top or is_on_right or is_on_bottom:
        print("Skipping particle (on image edge).")
        return

    # new bbox center
    center_x = x + w // 2
    center_y = y + h // 2

    # square size - take max dimn
    size = int(max(w, h) * padding_factor)

    # bottom left corner of new bbox
    x1 = max(center_x - size // 2, 0)
    y1 = max(center_y - size // 2, 0)

    # top right corner of new bbox
    x2 = max(x1 + size, 0)
    y2 = max(y1 + size, 0)

    cropped_particle = img[y1:y2, x1:x2]

    # Remove empty crops
    if cropped_particle.size == 0:
        return

    return cropped_particle


def find_best_max_size(
    img_dir: str,
    csv_path: str,
    plot: Optional[bool] = False,
    plot_savepath: Optional[str] = None,
) -> float:
    """
    Function to find the best maximum size threshold for particle segregation.
    Parameters
    ----------
    img_dir : str
        Directory containing images of particles.
    csv_path : str
        Path to the CSV file containing results data.
    plot : Optional[bool], optional
        Whether to plot the CDF and exclusion line, by default False.
    plot_savepath : Optional[str], optional
        Path to save the plot if plot is True, by default None.
    """
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Directory {img_dir} does not exist.")
    elif not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file {csv_path} does not exist.")

    # get all jpg files in directory
    jpg_files = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
    areas = []
    cnts = []
    for jpg in jpg_files:
        # Read image
        img = cv2.imread(os.path.join(img_dir, jpg), cv2.IMREAD_GRAYSCALE)
        # Detect contours
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        _, thresh = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cnts.append(len(contours))
        for cnt in contours:
            areas.append(cv2.contourArea(cnt))

    # Read results CSV
    results_df = pd.read_csv(csv_path, encoding="ISO-8859-1", sep="\t")

    n_results = len(results_df)  # Number of particles with results
    n_contours = sum(cnts)  # Total number of detected contours

    # Generate CDF and find size threshold
    areas_sorted = np.sort(np.array(areas))
    p = np.linspace(0, 1, len(areas))
    excluded_frac = 1 - n_results / n_contours

    def cdf(a) -> float:
        return np.interp(a, p, areas_sorted).item()

    min_size = cdf(excluded_frac)

    if plot:
        # Plot CDF and exclusion line
        plt.plot(areas_sorted, p)
        plt.axhline(y=excluded_frac, color="r", linestyle="--")

        plt.xlabel("Particle Area", fontsize=14)
        plt.ylabel("CDF", fontsize=14)

        plt.savefig(plot_savepath) if plot_savepath else plt.show()
        plt.close()

    return min_size


def crop_particles(img_dir: str, max_size: float, target_size: int = 256) -> None:
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Directory {img_dir} does not exist.")

    # get all jpg files in directory
    jpg_files = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]

    particles_processed = 0
    particles_skipped = 0

    for jpg in jpg_files:
        # read image
        img = cv2.imread(os.path.join(img_dir, jpg), cv2.IMREAD_GRAYSCALE)
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        _, thresh = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            # filter small contours
            area = cv2.contourArea(cnt)
            if area <= max_size:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            img_height, img_width = img.shape[:2]

            centered_img = _center_bbox(
                img,
                bb_coords=(x, y),
                bb_dimn=(w, h),
                img_dimn=(img_height, img_width),
                padding_factor=1.2,
            )
            if centered_img is None:
                particles_skipped += 1
                continue
            else:
                particles_processed += 1

            scaled_img = cv2.resize(
                centered_img,
                dsize=(target_size, target_size),
                interpolation=cv2.INTER_CUBIC,
            )

            binarised_img = cv2.threshold(
                scaled_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1]

            blur_bin = cv2.GaussianBlur(
                binarised_img, (0, 0), sigmaX=3, sigmaY=3, borderType=cv2.BORDER_DEFAULT
            )
            rescaled_img = exposure.rescale_intensity(
                blur_bin, in_range=(127.5, 255), out_range=(0, 255)
            )

            if not os.path.isdir("cropped_particles"):
                os.makedirs("cropped_particles")
            basename = os.path.basename(img_dir)
            save_path = (
                f"cropped_particles/{basename}_particle_{particles_processed}.png"
            )
            cv2.imwrite(save_path, rescaled_img)
