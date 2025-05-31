# Adaptive Bounding Box Annotation for whole dataset

import os
import cv2
import csv
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_iou(boxA_coords, boxB_coords):
    xA = max(boxA_coords[0], boxB_coords[0])
    yA = max(boxA_coords[1], boxB_coords[1])
    xB = min(boxA_coords[2], boxB_coords[2])
    yB = min(boxA_coords[3], boxB_coords[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA_coords[2] - boxA_coords[0]) * (boxA_coords[3] - boxA_coords[1])
    boxBArea = (boxB_coords[2] - boxB_coords[0]) * (boxB_coords[3] - boxB_coords[1])
    unionArea = boxAArea + boxBArea - interArea
    iou = interArea / unionArea if unionArea > 0 else 0.0
    return iou


def _preprocess_single_image(
    image_path,
    dotted_image_path,
    image_annotations_df,
    box_size,
    min_allowable_rate,
    image_id_str,
    verbose=False,
):
    min_allowable_dimension = int(box_size * min_allowable_rate)
    current_image_final_box_details = []

    class_colors_preprocess = {
        0: (0, 255, 0),
        1: (255, 0, 0),
        2: (0, 0, 255),
        3: (0, 255, 255),
        4: (255, 0, 255),
    }
    default_color_preprocess = (255, 255, 255)

    image = cv2.imread(image_path)
    if image is None:
        if verbose:
            print(
                f"  Warning (preprocess): Could not read image {image_path}. Skipping."
            )
        return None, None

    if os.path.exists(dotted_image_path):
        dotted_image = cv2.imread(dotted_image_path)
        if dotted_image is not None:
            if image.shape == dotted_image.shape:
                black_pixels_mask = np.all(dotted_image == [0, 0, 0], axis=-1)
                image[black_pixels_mask] = [0, 0, 0]
            elif verbose:
                print(
                    f"  Warning (preprocess): Original image {image_id_str} and its dotted version have different dimensions. Skipping masking."
                )
        elif verbose:
            print(
                f"  Warning (preprocess): Could not read dotted image {dotted_image_path} for {image_id_str}. Masking skipped."
            )
    elif verbose:
        print(
            f"  Info (preprocess): No dotted image found at {dotted_image_path} for {image_id_str}."
        )

    if image_annotations_df.empty:
        if verbose:
            print(
                f"  Info (preprocess): No annotations found for image {image_id_str}."
            )
    else:
        initial_boxes_props = []
        for index, row_data in image_annotations_df.iterrows():
            center_x = int(row_data["col"])
            center_y = int(row_data["row"])
            cls = int(row_data["cls"])
            initial_boxes_props.append(
                {"center_x": center_x, "center_y": center_y, "cls": cls, "id": index}
            )

        num_boxes = len(initial_boxes_props)
        final_box_dimensions = [(box_size, box_size) for _ in range(num_boxes)]

        if num_boxes > 1:
            initial_coords_list_for_iou_decision = []
            for i_idx in range(num_boxes):
                props = initial_boxes_props[i_idx]
                h_bs = box_size // 2
                x1 = props["center_x"] - h_bs
                y1 = props["center_y"] - h_bs
                x2 = props["center_x"] + h_bs
                y2 = props["center_y"] + h_bs
                initial_coords_list_for_iou_decision.append((x1, y1, x2, y2))

            for i_idx in range(num_boxes):
                for j_idx in range(i_idx + 1, num_boxes):
                    coords_i_orig = initial_coords_list_for_iou_decision[i_idx]
                    coords_j_orig = initial_coords_list_for_iou_decision[j_idx]
                    iou = calculate_iou(coords_i_orig, coords_j_orig)
                    reduction_factor_primary = 1.0
                    if iou >= 0.5:
                        reduction_factor_primary = 0.70
                    elif iou >= 0.45:
                        reduction_factor_primary = 0.72
                    elif iou >= 0.4:
                        reduction_factor_primary = 0.74
                    elif iou >= 0.35:
                        reduction_factor_primary = 0.76
                    elif iou >= 0.3:
                        reduction_factor_primary = 0.78
                    elif iou >= 0.25:
                        reduction_factor_primary = 0.81
                    elif iou >= 0.2:
                        reduction_factor_primary = 0.84
                    elif iou >= 0.15:
                        reduction_factor_primary = 0.87
                    elif iou >= 0.1:
                        reduction_factor_primary = 0.90

                    if reduction_factor_primary < 1.0:
                        reduction_factor_secondary = (
                            1.0 - (1.0 - reduction_factor_primary) / 2.0
                        )
                        if verbose:
                            print(
                                f"  Image {image_id_str}: Adjusting boxes {initial_boxes_props[i_idx]['id']} and {initial_boxes_props[j_idx]['id']} with IOU {iou:.4f}, Factors: P={reduction_factor_primary:.2f}, S={reduction_factor_secondary:.2f}"
                            )
                        props_i = initial_boxes_props[i_idx]
                        props_j = initial_boxes_props[j_idx]
                        delta_x = abs(props_i["center_x"] - props_j["center_x"])
                        delta_y = abs(props_i["center_y"] - props_j["center_y"])
                        current_w_i, current_h_i = final_box_dimensions[i_idx]
                        current_w_j, current_h_j = final_box_dimensions[j_idx]

                        if delta_x >= delta_y:  # Adjust width more, height less
                            if (
                                current_w_i > min_allowable_dimension
                                or current_h_i > min_allowable_dimension
                            ):
                                target_new_w_i = int(
                                    current_w_i * reduction_factor_primary
                                )
                                actual_new_w_i = max(
                                    target_new_w_i, min_allowable_dimension
                                )
                                target_new_h_i = int(
                                    current_h_i * reduction_factor_secondary
                                )
                                actual_new_h_i = max(
                                    target_new_h_i, min_allowable_dimension
                                )
                                final_box_dimensions[i_idx] = (
                                    min(current_w_i, actual_new_w_i),
                                    min(current_h_i, actual_new_h_i),
                                )
                            if (
                                current_w_j > min_allowable_dimension
                                or current_h_j > min_allowable_dimension
                            ):
                                target_new_w_j = int(
                                    current_w_j * reduction_factor_primary
                                )
                                actual_new_w_j = max(
                                    target_new_w_j, min_allowable_dimension
                                )
                                target_new_h_j = int(
                                    current_h_j * reduction_factor_secondary
                                )
                                actual_new_h_j = max(
                                    target_new_h_j, min_allowable_dimension
                                )
                                final_box_dimensions[j_idx] = (
                                    min(current_w_j, actual_new_w_j),
                                    min(current_h_j, actual_new_h_j),
                                )
                        else:  # Adjust height more, width less
                            if (
                                current_h_i > min_allowable_dimension
                                or current_w_i > min_allowable_dimension
                            ):
                                target_new_h_i = int(
                                    current_h_i * reduction_factor_primary
                                )
                                actual_new_h_i = max(
                                    target_new_h_i, min_allowable_dimension
                                )
                                target_new_w_i = int(
                                    current_w_i * reduction_factor_secondary
                                )
                                actual_new_w_i = max(
                                    target_new_w_i, min_allowable_dimension
                                )
                                final_box_dimensions[i_idx] = (
                                    min(current_w_i, actual_new_w_i),
                                    min(current_h_i, actual_new_h_i),
                                )
                            if (
                                current_h_j > min_allowable_dimension
                                or current_w_j > min_allowable_dimension
                            ):
                                target_new_h_j = int(
                                    current_h_j * reduction_factor_primary
                                )
                                actual_new_h_j = max(
                                    target_new_h_j, min_allowable_dimension
                                )
                                target_new_w_j = int(
                                    current_w_j * reduction_factor_secondary
                                )
                                actual_new_w_j = max(
                                    target_new_w_j, min_allowable_dimension
                                )
                                final_box_dimensions[j_idx] = (
                                    min(current_w_j, actual_new_w_j),
                                    min(current_h_j, actual_new_h_j),
                                )

        for k_idx in range(num_boxes):
            props = initial_boxes_props[k_idx]
            current_center_x, current_center_y, current_cls = (
                props["center_x"],
                props["center_y"],
                props["cls"],
            )
            current_w_final, current_h_final = final_box_dimensions[k_idx]
            current_w_final = max(current_w_final, min_allowable_dimension)
            current_h_final = max(current_h_final, min_allowable_dimension)
            half_w, half_h = current_w_final // 2, current_h_final // 2
            x1_final, y1_final = current_center_x - half_w, current_center_y - half_h
            x2_final, y2_final = current_center_x + half_w, current_center_y + half_h
            current_image_final_box_details.append(
                (x1_final, y1_final, x2_final, y2_final, current_cls)
            )

        if verbose and not image_annotations_df.empty:
            img_to_draw_on = image.copy()
            for k_idx in range(len(current_image_final_box_details)):
                x1_f, y1_f, x2_f, y2_f, cls_f = current_image_final_box_details[k_idx]
                color = class_colors_preprocess.get(cls_f, default_color_preprocess)
                cv2.rectangle(img_to_draw_on, (x1_f, y1_f), (x2_f, y2_f), color, 2)

    return image.copy(), current_image_final_box_details  # Return a copy of the image


def _get_patches_for_single_image(
    processed_image_single,
    boxes_details_for_single_image,
    patch_size=960,
    max_rate_for_patches=0.30,
    min_select_num=4,
    max_select_num=8,
    num_grid_rows=8,
    num_grid_cols=6,
    black_pixel_threshold=0.4,
    drop_zero_bbox_rate=0.6,
    verbose=False,  # Verbose for this specific image's patch generation
):
    def _create_patch_and_adjust_boxes(
        image_to_crop, center_x, center_y, p_size, original_boxes_for_img
    ):
        img_h, img_w = image_to_crop.shape[:2]
        patch_origin_x = int(center_x - p_size / 2)
        patch_origin_y = int(center_y - p_size / 2)
        num_channels = image_to_crop.shape[2] if image_to_crop.ndim == 3 else 1
        current_patch_image = np.zeros(
            (p_size, p_size, num_channels) if num_channels > 1 else (p_size, p_size),
            dtype=image_to_crop.dtype,
        )
        src_x_start = max(0, patch_origin_x)
        src_y_start = max(0, patch_origin_y)
        src_x_end = min(img_w, patch_origin_x + p_size)
        src_y_end = min(img_h, patch_origin_y + p_size)
        dst_x_start = max(0, -patch_origin_x)
        dst_y_start = max(0, -patch_origin_y)
        copy_w = src_x_end - src_x_start
        copy_h = src_y_end - src_y_start

        if copy_w > 0 and copy_h > 0:
            if num_channels > 1:
                current_patch_image[
                    dst_y_start : dst_y_start + copy_h,
                    dst_x_start : dst_x_start + copy_w,
                    :,
                ] = image_to_crop[src_y_start:src_y_end, src_x_start:src_x_end, :]
            else:
                current_patch_image[
                    dst_y_start : dst_y_start + copy_h,
                    dst_x_start : dst_x_start + copy_w,
                ] = image_to_crop[src_y_start:src_y_end, src_x_start:src_x_end]

        if num_channels > 1:
            black_pixels = np.sum(np.all(current_patch_image == 0, axis=-1))
        else:
            black_pixels = np.sum(current_patch_image == 0)
        total_pixels = p_size * p_size
        black_pixel_ratio = black_pixels / total_pixels if total_pixels > 0 else 0

        adjusted_boxes_for_patch = []
        for orig_box in original_boxes_for_img:
            x1_o, y1_o, x2_o, y2_o, cls_o = orig_box
            if not (
                x2_o <= patch_origin_x
                or x1_o >= patch_origin_x + p_size
                or y2_o <= patch_origin_y
                or y1_o >= patch_origin_y + p_size
            ):
                x1_rel, y1_rel = x1_o - patch_origin_x, y1_o - patch_origin_y
                x2_rel, y2_rel = x2_o - patch_origin_x, y2_o - patch_origin_y
                x1_p, y1_p = max(0, x1_rel), max(0, y1_rel)
                x2_p, y2_p = min(p_size, x2_rel), min(p_size, y2_rel)
                orig_area = max(0, x2_o - x1_o) * max(0, y2_o - y1_o)
                clipped_area = max(0, x2_p - x1_p) * max(0, y2_p - y1_p)
                if x1_p < x2_p and y1_p < y2_p and orig_area > 0:
                    if clipped_area / orig_area >= 0.2:
                        adjusted_boxes_for_patch.append(
                            (int(x1_p), int(y1_p), int(x2_p), int(y2_p), cls_o)
                        )
        return current_patch_image, adjusted_boxes_for_patch, black_pixel_ratio

    patches_from_this_image = []
    boxes_for_patches_from_this_image = []

    if processed_image_single is None:
        if verbose:
            print("  Error (get_patches): Received None image for patch generation.")
        return patches_from_this_image, boxes_for_patches_from_this_image

    image = processed_image_single
    current_image_original_boxes = boxes_details_for_single_image
    img_height, img_width = image.shape[:2]

    # Overfitting: Patches centered on selected bounding boxes
    num_total_boxes = len(current_image_original_boxes)
    if num_total_boxes > 0:
        select_count_by_percent = int(num_total_boxes * max_rate_for_patches)
        num_to_select_centers = min(
            max(min_select_num, select_count_by_percent),
            max_select_num,
            num_total_boxes,
        )
        if verbose:
            print(
                f"    Method 1 (get_patches): Aiming for {num_to_select_centers} patches from {num_total_boxes} bbox candidates."
            )
        if num_to_select_centers > 0:
            actual_num_to_select = min(num_to_select_centers, num_total_boxes)
            selected_indices = np.random.choice(
                range(num_total_boxes), size=actual_num_to_select, replace=False
            )
            boxes_for_patch_centers = [
                current_image_original_boxes[idx] for idx in selected_indices
            ]
            for center_box_details in boxes_for_patch_centers:
                x1_cb, y1_cb, x2_cb, y2_cb, _ = center_box_details
                center_x_cb, center_y_cb = (x1_cb + x2_cb) / 2, (y1_cb + y2_cb) / 2
                patch_img, adjusted_boxes, black_ratio = _create_patch_and_adjust_boxes(
                    image,
                    center_x_cb,
                    center_y_cb,
                    patch_size,
                    current_image_original_boxes,
                )
                if black_ratio < black_pixel_threshold:
                    if len(adjusted_boxes) == 0:
                        if np.random.rand() < drop_zero_bbox_rate:
                            if verbose:
                                print("      Patch discarded (no bbox, random drop).")
                            continue
                    patches_from_this_image.append(patch_img)
                    boxes_for_patches_from_this_image.append(adjusted_boxes)
                elif verbose:
                    print(
                        f"      Patch (bbox center) discarded, black_ratio: {black_ratio:.2f}"
                    )

    # Tiling: Patches from a uniform grid
    if verbose:
        print(
            f"    Method 2 (get_patches): Grid sampling {num_grid_rows}x{num_grid_cols} cells."
        )
    margin = patch_size // 2
    valid_center_x_start, valid_center_x_end = margin, img_width - margin
    valid_center_y_start, valid_center_y_end = margin, img_height - margin

    stride = int(patch_size * 0.95)
    auto_num_grid_cols = max(1, int((img_width - patch_size) // stride + 1))
    auto_num_grid_rows = max(1, int((img_height - patch_size) // stride + 1))
    num_grid_cols = auto_num_grid_cols
    num_grid_rows = auto_num_grid_rows

    if (
        valid_center_x_end > valid_center_x_start
        and valid_center_y_end > valid_center_y_start
    ):
        for r_idx in range(num_grid_rows):
            for c_idx in range(num_grid_cols):
                center_x_grid = valid_center_x_start + c_idx * stride
                center_y_grid = valid_center_y_start + r_idx * stride

                patch_img, adjusted_boxes, black_ratio = _create_patch_and_adjust_boxes(
                    image,
                    center_x_grid,
                    center_y_grid,
                    patch_size,
                    current_image_original_boxes,
                )
                if black_ratio < black_pixel_threshold:
                    if len(adjusted_boxes) == 0:
                        if np.random.rand() < drop_zero_bbox_rate:
                            if verbose:
                                print("      Patch discarded (no bbox, random drop).")
                            continue
                    patches_from_this_image.append(patch_img)
                    boxes_for_patches_from_this_image.append(adjusted_boxes)
                elif verbose:
                    print(
                        f"      Patch (grid center) discarded, black_ratio: {black_ratio:.2f}"
                    )
    elif verbose:
        print(
            f"    Method 2 (get_patches): Skipped due to insufficient image size for grid sampling."
        )

    target_patch_size = 640
    if patch_size != target_patch_size:
        scale_factor_x = target_patch_size / patch_size
        scale_factor_y = target_patch_size / patch_size
        resized_patches = []
        resized_boxes = []
        for patch_img, patch_boxes in zip(
            patches_from_this_image, boxes_for_patches_from_this_image
        ):
            patch_img_resized = cv2.resize(
                patch_img,
                (target_patch_size, target_patch_size),
                interpolation=cv2.INTER_AREA,
            )
            patch_boxes_resized = []
            for box in patch_boxes:
                x1, y1, x2, y2, cls = box
                x1_new = int(round(x1 * scale_factor_x))
                y1_new = int(round(y1 * scale_factor_y))
                x2_new = int(round(x2 * scale_factor_x))
                y2_new = int(round(y2 * scale_factor_y))
                patch_boxes_resized.append((x1_new, y1_new, x2_new, y2_new, cls))
            resized_patches.append(patch_img_resized)
            resized_boxes.append(patch_boxes_resized)
        patches_from_this_image = resized_patches
        boxes_for_patches_from_this_image = resized_boxes

    if verbose:
        print(f"    Generated {len(patches_from_this_image)} patches from this image.")
    return patches_from_this_image, boxes_for_patches_from_this_image


def _xyxy_to_yolo_for_patches(
    patches_list_from_single_image,
    patch_boxes_list_from_single_image,
):
    all_yolo_annotations_for_these_patches = []
    if len(patches_list_from_single_image) != len(patch_boxes_list_from_single_image):
        print("Error (_xyxy_to_yolo): Mismatch between num patches and num box lists.")
        return []

    for i, patch_image in enumerate(patches_list_from_single_image):
        img_height, img_width = patch_image.shape[:2]
        if img_height == 0 or img_width == 0:  # Safeguard for empty patches
            print(f"Warning (_xyxy_to_yolo): Patch {i} has zero dimension. Skipping.")
            all_yolo_annotations_for_these_patches.append([])
            continue

        current_patch_yolo_annotations = []
        boxes_for_current_patch = patch_boxes_list_from_single_image[i]
        for box_detail in boxes_for_current_patch:
            x1, y1, x2, y2, class_id = box_detail
            box_width, box_height = float(x2 - x1), float(y2 - y1)
            if box_width <= 0 or box_height <= 0:
                continue

            x_center, y_center = float(x1) + box_width / 2, float(y1) + box_height / 2

            x_center_norm = np.clip(x_center / img_width, 0.0, 1.0)
            y_center_norm = np.clip(y_center / img_height, 0.0, 1.0)
            width_norm = np.clip(box_width / img_width, 0.0, 1.0)
            height_norm = np.clip(box_height / img_height, 0.0, 1.0)

            yolo_format_string = f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
            current_patch_yolo_annotations.append(yolo_format_string)
        all_yolo_annotations_for_these_patches.append(current_patch_yolo_annotations)
    return all_yolo_annotations_for_these_patches


def visualize_patches(patches_list, patch_boxes_list, grid_size=(4, 4)):
    num_patches_to_show = grid_size[0] * grid_size[1]

    if not patches_list:
        print("\nNo patches were generated/collected to visualize.")
        return

    if len(patches_list) < num_patches_to_show:
        print(
            f"\nWarning (visualize): Not enough patches ({len(patches_list)}) to fill a {grid_size[0]}x{grid_size[1]} grid. Showing {len(patches_list)} patches instead."
        )
        num_patches_to_show = len(patches_list)

    if num_patches_to_show == 0:
        print("\nNo patches to show for visualization.")
        return

    actual_rows = int(np.ceil(np.sqrt(num_patches_to_show)))
    actual_cols = int(np.ceil(num_patches_to_show / actual_rows))
    num_patches_to_show = min(num_patches_to_show, len(patches_list))

    vis_class_colors = {
        0: (0, 255, 0),
        1: (255, 0, 0),
        2: (0, 0, 255),
        3: (0, 255, 255),
        4: (255, 0, 255),
    }
    vis_default_color = (255, 255, 255)

    num_available_patches = len(patches_list)
    selected_indices = np.random.choice(
        num_available_patches,
        size=min(num_patches_to_show, num_available_patches),
        replace=False,
    )

    actual_num_really_shown = len(selected_indices)
    if actual_num_really_shown == 0:
        print("No patches selected to display in visualization.")
        return

    actual_rows = int(np.ceil(np.sqrt(actual_num_really_shown)))
    actual_cols = int(np.ceil(actual_num_really_shown / actual_rows))

    fig, axes = plt.subplots(
        actual_rows, actual_cols, figsize=(actual_cols * 5, actual_rows * 5)
    )
    if actual_num_really_shown == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    print(
        f"\nVisualizing {len(selected_indices)} random patches in a {actual_rows}x{actual_cols} grid."
    )

    for i, random_idx in enumerate(selected_indices):
        if i >= len(axes):
            break

        patch_to_visualize = patches_list[random_idx].copy()
        boxes_for_patch = patch_boxes_list[random_idx]
        ax = axes[i]

        for box in boxes_for_patch:
            x1, y1, x2, y2, cls = box
            color = vis_class_colors.get(cls, vis_default_color)
            cv2.rectangle(patch_to_visualize, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                patch_to_visualize,
                f"Cls: {cls}",
                (x1, y1 - 10 if y1 - 10 > 10 else y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        if patch_to_visualize.ndim == 2 or (
            patch_to_visualize.ndim == 3 and patch_to_visualize.shape[2] == 1
        ):
            ax.imshow(patch_to_visualize, cmap="gray")
        else:
            ax.imshow(cv2.cvtColor(patch_to_visualize, cv2.COLOR_BGR2RGB))

        ax.set_title(
            f"Sample Patch (Orig Idx: {random_idx}), {len(boxes_for_patch)} bxs"
        )
        ax.axis("off")

    for j in range(len(selected_indices), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def trans_to_yolo(
    image_dir=r"dataset/TrainSmall2/Train",
    dotted_image_dir=r"dataset/TrainSmall2/TrainDotted",
    csv_file_path=r"dataset/coords-threeplusone-v0.4.csv",
    mismatched_file_path=r"dataset/MismatchedTrainImages.txt",
    output_base_dir=r"yolo_dataset_sequential",
    box_size_preprocess=55,
    bbox_rate_min=0.4,
    patch_size_getpatches=960,
    max_rate_getpatches=0.2,
    min_select_getpatches=3,
    num_grid_rows_getpatches=10,
    num_grid_cols_getpatches=7,
    val_split_ratio=0.06,
    black_pixel_threshold_getpatches=0.5,
    visualize_final_samples=True,
    num_samples_to_visualize=16,
    main_verbose=False,
    preprocess_verbose=False,
    getpatches_verbose=False,
    drop_zero_bbox_rate=0.6,
    max_select_num=6,
):
    print("Starting YOLO data generation pipeline (sequential processing)...")

    mismatched_ids = set()
    if os.path.exists(mismatched_file_path):
        with open(mismatched_file_path, "r") as f:
            for line in f:
                img_id = line.strip()
                if img_id:
                    mismatched_ids.add(img_id)
    else:
        print(f"Warning: {mismatched_file_path} not found, no images will be skipped.")

    try:
        master_annotations_df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: Master CSV File {csv_file_path} not found. Exiting.")
        return None

    image_files = [
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not image_files:
        print(f"Error: No compatible image files found in {image_dir}. Exiting.")
        return None

    # Partition the dataset into training and validation sets
    image_files = sorted(image_files)
    num_images = len(image_files)
    # num_valid = max(1, int(num_images * val_split_ratio))
    num_valid = 0
    valid_indices = set(np.random.choice(num_images, num_valid, replace=False))
    train_indices = set(range(num_images)) - valid_indices

    output_images_train_dir = os.path.join(output_base_dir, "images", "train")
    output_labels_train_dir = os.path.join(output_base_dir, "labels", "train")
    output_images_valid_dir = os.path.join(output_base_dir, "images", "valid")
    output_labels_valid_dir = os.path.join(output_base_dir, "labels", "valid")
    os.makedirs(output_images_train_dir, exist_ok=True)
    os.makedirs(output_labels_train_dir, exist_ok=True)
    os.makedirs(output_images_valid_dir, exist_ok=True)
    os.makedirs(output_labels_valid_dir, exist_ok=True)

    global_patch_save_index_train = 0
    global_patch_save_index_valid = 0
    total_patches_successfully_saved_train = 0
    total_patches_successfully_saved_valid = 0

    sample_patches_for_vis = []
    sample_boxes_for_vis = []

    counts_csv_path = os.path.join(output_base_dir, "counts.csv")
    with open(counts_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "filename",
                "adult_males",
                "subadult_males",
                "adult_females",
                "juveniles",
                "pups",
            ]
        )

    for idx, image_filename in enumerate(
        tqdm.tqdm(image_files, desc="Processing Original Images", ncols=120)
    ):
        if main_verbose:
            print(f"\nProcessing original image: {image_filename}")
        try:
            image_id_str = os.path.splitext(image_filename)[0]
            image_id = int(image_id_str)
        except ValueError:
            if main_verbose:
                print(
                    f"  Warning (main): Filename {image_filename} not a parsable ID. Skipping."
                )
            continue

        current_image_path = os.path.join(image_dir, image_filename)
        current_dotted_path = os.path.join(dotted_image_dir, image_filename)

        image_annotations_df_single = master_annotations_df[
            master_annotations_df["tid"] == image_id
        ]

        # --- Stage 1: Preprocess single image ---
        if main_verbose:
            print(f"  Stage 1: Preprocessing {image_id_str}...")
        processed_img, final_boxes_for_img = _preprocess_single_image(
            current_image_path,
            current_dotted_path,
            image_annotations_df_single,
            box_size_preprocess,
            bbox_rate_min,
            image_id_str,
            preprocess_verbose,
        )
        if processed_img is None:
            if main_verbose:
                print(
                    f"  Skipping {image_filename} due to preprocessing error or no image data."
                )
            continue

        # --- Stage 2: Generate patches for this single processed image ---
        if main_verbose:
            print(f"  Stage 2: Generating patches for {image_id_str}...")
        patches_from_this_orig_img, patch_boxes_xyxy_this_orig_img = (
            _get_patches_for_single_image(
                processed_img,
                final_boxes_for_img,
                patch_size=patch_size_getpatches,
                max_rate_for_patches=max_rate_getpatches,
                min_select_num=min_select_getpatches,
                num_grid_rows=num_grid_rows_getpatches,
                drop_zero_bbox_rate=drop_zero_bbox_rate,
                max_select_num=max_select_num,
                num_grid_cols=num_grid_cols_getpatches,
                black_pixel_threshold=black_pixel_threshold_getpatches,
                verbose=getpatches_verbose,
            )
        )

        if not patches_from_this_orig_img:
            if main_verbose:
                print(
                    f"  No patches generated for {image_filename}. Skipping save for this image."
                )
            del processed_img, final_boxes_for_img  # Clean up
            continue
        if main_verbose:
            print(
                f"    Generated {len(patches_from_this_orig_img)} patches for {image_id_str}."
            )

        # --- Stage 3: Convert patch annotations to YOLO format ---
        if main_verbose:
            print(
                f"  Stage 3: Converting {len(patches_from_this_orig_img)} patch annotations to YOLO for {image_id_str}..."
            )
        yolo_annotations_for_these_patches = _xyxy_to_yolo_for_patches(
            patches_from_this_orig_img, patch_boxes_xyxy_this_orig_img
        )

        # --- Stage 4: Save these patches and their labels ---
        if main_verbose:
            print(
                f"  Stage 4: Saving {len(patches_from_this_orig_img)} patches and labels for {image_id_str}..."
            )
        is_valid = idx in valid_indices
        if is_valid:
            img_dir = output_images_valid_dir
            lbl_dir = output_labels_valid_dir
        else:
            img_dir = output_images_train_dir
            lbl_dir = output_labels_train_dir

        for patch_idx, (patch_img, patch_boxes, yolo_annos) in enumerate(
            zip(
                patches_from_this_orig_img,
                patch_boxes_xyxy_this_orig_img,
                yolo_annotations_for_these_patches,
            )
        ):
            patch_filename_with_extension = ""
            if is_valid:
                patch_filename_base = f"val_patch_{global_patch_save_index_valid:07d}"
                global_patch_save_index_valid += 1
            else:
                patch_filename_base = f"train_patch_{global_patch_save_index_train:07d}"
                global_patch_save_index_train += 1

            patch_filename_with_extension = f"{patch_filename_base}.png"
            img_save_path = os.path.join(img_dir, patch_filename_with_extension)
            lbl_save_path = os.path.join(lbl_dir, f"{patch_filename_base}.txt")

            try:
                write_status = cv2.imwrite(img_save_path, patch_img)
                if not write_status:
                    if main_verbose:
                        print(
                            f"    Error (main): Failed to write image {img_save_path}."
                        )
                    continue

                with open(lbl_save_path, "w") as f:
                    for line in yolo_annos:
                        f.write(f"{line}\n")

                class_counts = [0, 0, 0, 0, 0]
                for box in patch_boxes:
                    cls = int(box[4])
                    if 0 <= cls <= 4:
                        class_counts[cls] += 1

                with open(counts_csv_path, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(
                        [
                            patch_filename_with_extension,  # Use filename with extension
                            class_counts[0],
                            class_counts[1],
                            class_counts[2],
                            class_counts[3],
                            class_counts[4],
                        ]
                    )

                if is_valid:
                    total_patches_successfully_saved_valid += 1
                else:
                    total_patches_successfully_saved_train += 1

                if (
                    visualize_final_samples
                    and num_samples_to_visualize > 0
                    and len(sample_patches_for_vis) < num_samples_to_visualize
                ):
                    sample_patches_for_vis.append(patch_img.copy())
                    sample_boxes_for_vis.append(patch_boxes)

            except Exception as e:
                if main_verbose:
                    print(
                        f"    Error (main): Saving patch {patch_filename_base} or its label: {e}"
                    )
                if os.path.exists(img_save_path):
                    os.remove(img_save_path)
                if os.path.exists(lbl_save_path):
                    os.remove(lbl_save_path)
                continue

        del (
            processed_img,
            final_boxes_for_img,
            patches_from_this_orig_img,
            patch_boxes_xyxy_this_orig_img,
            yolo_annotations_for_these_patches,
        )
        if "patch_image_to_save" in locals():
            del patch_image_to_save

    print(
        f"\nPipeline finished. Successfully saved {total_patches_successfully_saved_train} training patches and {total_patches_successfully_saved_valid} validation patches."
    )
    print(f"Train images saved in: {output_images_train_dir}")
    print(f"Train labels saved in: {output_labels_train_dir}")
    print(f"Valid images saved in: {output_images_valid_dir}")
    print(f"Valid labels saved in: {output_labels_valid_dir}")
    print(f"Patch counts saved in: {counts_csv_path}")

    # --- Stage 5: Visualization of collected samples (if enabled) ---
    if visualize_final_samples and num_samples_to_visualize > 0:
        if sample_patches_for_vis:
            visualize_patches(sample_patches_for_vis, sample_boxes_for_vis)
        else:
            print("\nNo patches were collected/available for visualization.")

    return output_base_dir


if __name__ == "__main__":
    final_output_dir = trans_to_yolo(
        image_dir=r"SrcData/Train",  # Source of original images
        dotted_image_dir=r"SrcData/TrainDotted",  # Source of dotted images
        output_base_dir="yolo_dataset_800",
        # image_dir=r"dataset/TrainSmall2/Train",  # Source of original images
        # dotted_image_dir=r"dataset/TrainSmall2/TrainDotted",  # Source of dotted images
        # output_base_dir="yolo_dataset_small",
        csv_file_path=r"dataset/coords-threeplusone-v0.4.csv",  # Master annotations
        mismatched_file_path=r"dataset/MismatchedTrainImages.txt",
        box_size_preprocess=55,
        bbox_rate_min=0.5,
        val_split_ratio=0.05,  # Validation split ratio
        patch_size_getpatches=800,  # Patch size
        min_select_getpatches=1,  # Min patches from bboxes (per oversample image)
        max_select_num=3,  # Max patches from bboxes (per oversample image)
        drop_zero_bbox_rate=0.75,  # Rate to drop patches with no bboxes
        black_pixel_threshold_getpatches=0.5,  # Max black pixel ratio for a patch
        visualize_final_samples=False,  # Whether to show samples at the end
        num_samples_to_visualize=16,  # How many samples to show (if visualize_final_samples is True)
        main_verbose=False,  # Overall progress prints
        preprocess_verbose=False,  # Detailed prints during single image preprocessing
        getpatches_verbose=False,  # Detailed prints during single image patch generation
    )

    if final_output_dir:
        print(f"\nDataset generation complete. Output in: {final_output_dir}")
    else:
        print("\nDataset generation failed or was interrupted.")
