import html
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from IPython.display import HTML, display


def render_availability_table(experiments):
    """Render a simple availability table for notebook audiences."""
    rows = []
    for experiment_name, mapping in experiments:
        for item_name, value in mapping.items():
            if value is None:
                available = "N/A (optional)"
            else:
                available = "Yes" if value.exists() else "No"
            rows.append((experiment_name, item_name, available))

    header = """
<tr>
  <th style='text-align:left; padding:6px;'>Experiment</th>
  <th style='text-align:left; padding:6px;'>Data item</th>
  <th style='text-align:left; padding:6px;'>Available</th>
</tr>
"""

    body = ""
    for experiment, item, available in rows:
        body += (
            f"<tr><td style='padding:6px;'>{experiment}</td>"
            f"<td style='padding:6px;'>{item}</td>"
            f"<td style='padding:6px;'>{available}</td></tr>"
        )

    table_html = f"""
<table style='border-collapse:collapse; border:1px solid #ccc; font-size:14px;'>
{header}
{body}
</table>
"""

    display(HTML(table_html))


def labels_to_oar_names(label_values, roi_dict):
    """Convert numeric labels to readable OAR names."""
    names = []
    for value in label_values:
        label_id = int(value)
        if label_id == 0:
            continue
        names.append(roi_dict.get(label_id, f"Unknown label ({label_id})"))
    return names


def render_oar_name_list(names, title="Predicted OAR names", n_columns=2):
    """Render OAR names in a simple multi-column card layout."""
    safe_title = html.escape(title)
    if not names:
        display(
            HTML(
                f"""
<div style='border:1px solid #ddd; border-radius:8px; padding:10px; max-width:900px;'>
  <div style='font-weight:600; margin-bottom:6px;'>{safe_title}</div>
  <div style='color:#666;'>No OARs found in this view.</div>
</div>
"""
            )
        )
        return

    list_items = "".join(
        f"<li style='margin-bottom:4px;'>{html.escape(str(name))}</li>" for name in names
    )

    display(
        HTML(
            f"""
<div style='border:1px solid #ddd; border-radius:8px; padding:10px; max-width:900px;'>
  <div style='font-weight:600; margin-bottom:6px;'>{safe_title}</div>
  <ol style='margin:0; padding-left:18px; column-count:{n_columns}; column-gap:24px;'>
    {list_items}
  </ol>
</div>
"""
        )
    )


def _build_label_colormap(roi_dict):
    """Create a stable label->color map so colors do not change across slices."""
    max_label = max(roi_dict.keys())
    base = plt.cm.get_cmap("tab20", max_label)
    # Index 0 is background (transparent when masked)
    colors = [(0.0, 0.0, 0.0, 0.0)]
    for idx in range(1, max_label + 1):
        colors.append(base(idx - 1))
    cmap = ListedColormap(colors)
    bounds = np.arange(-0.5, max_label + 1.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm, max_label


def _apply_zoom_and_pan(image_slice, label_slice, zoom_factor, shift_x, shift_y):
    """Return a cropped view from full image using zoom and small pan offsets."""
    if zoom_factor <= 1.0:
        return image_slice, label_slice

    h, w = image_slice.shape
    crop_h = max(10, int(h / zoom_factor))
    crop_w = max(10, int(w / zoom_factor))

    center_y = h // 2 + int(shift_y)
    center_x = w // 2 + int(shift_x)

    center_y = max(crop_h // 2, min(h - crop_h // 2, center_y))
    center_x = max(crop_w // 2, min(w - crop_w // 2, center_x))

    r0 = max(0, center_y - crop_h // 2)
    r1 = min(h, r0 + crop_h)
    c0 = max(0, center_x - crop_w // 2)
    c1 = min(w, c0 + crop_w)

    return image_slice[r0:r1, c0:c1], label_slice[r0:r1, c0:c1]


def _rgba_to_hex(rgba):
    """Convert matplotlib RGBA tuple to hex color string."""
    r, g, b = rgba[:3]
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def _window_limits(level, width):
    """Convert CT window level/width to imshow vmin/vmax values."""
    width = max(1.0, float(width))
    level = float(level)
    return level - (width / 2.0), level + (width / 2.0)


def load_experiment2_data(metal_exp, nib, remove_slices=0):
    """Load Experiment 2 data and predictions for staged discovery workflow."""
    metal_img = nib.load(str(metal_exp["metal_case_image"])).get_fdata()[:, :, remove_slices:-remove_slices]
    no_metal_img = nib.load(str(metal_exp["no_metal_case_image"])).get_fdata()[:, :, remove_slices:-remove_slices]

    pred_model_a_case1 = (
        nib.load(str(metal_exp["no_metal_model_pred_metal"])).get_fdata()[:, :, remove_slices:-remove_slices]
        if Path(metal_exp["no_metal_model_pred_metal"]).exists()
        else None
    )
    pred_model_a_case2 = (
        nib.load(str(metal_exp["no_metal_model_pred_no_metal"])).get_fdata()[:, :, remove_slices:-remove_slices]
        if Path(metal_exp["no_metal_model_pred_no_metal"]).exists()
        else None
    )

    pred_model_b_case1 = nib.load(str(metal_exp["metal_model_pred_metal"])).get_fdata()[:, :, remove_slices:-remove_slices]
    pred_model_b_case2 = (
        nib.load(str(metal_exp["metal_model_pred_no_metal"])).get_fdata()
        if Path(metal_exp["metal_model_pred_no_metal"]).exists()
        else None
    )

    gt_case1 = (
        nib.load(str(metal_exp["gt_metal"])).get_fdata()[:, :, remove_slices:-remove_slices]
        if metal_exp.get("gt_metal") and Path(metal_exp["gt_metal"]).exists()
        else None
    )
    gt_case2 = (
        nib.load(str(metal_exp["gt_no_metal"])).get_fdata()[:, :, remove_slices:-remove_slices]
        if metal_exp.get("gt_no_metal") and Path(metal_exp["gt_no_metal"]).exists()
        else None
    )

    return {
        "metal_img": metal_img,
        "no_metal_img": no_metal_img,
        "pred_model_a_case1": pred_model_a_case1,
        "pred_model_a_case2": pred_model_a_case2,
        "pred_model_b_case1": pred_model_b_case1,
        "pred_model_b_case2": pred_model_b_case2,
        "gt_case1": gt_case1,
        "gt_case2": gt_case2,
    }


def launch_postop_viewer(postop, postop_pred, roi_dict, widgets, has_widgets):
    """Interactive post-surgery viewer with full-view start, zoom/pan, and stable legend."""
    if not has_widgets:
        print("ipywidgets is not available. Install it with: pip install ipywidgets")
        print("Then restart the kernel and run this cell again.")
        return

    max_slice = postop.shape[2] - 1
    label_counts = (postop_pred > 0).sum(axis=(0, 1))
    default_slice = int(np.argmax(label_counts)) if np.any(label_counts) else max_slice // 2
    cmap, norm, _ = _build_label_colormap(roi_dict)

    output = widgets.Output()
    legend_output = widgets.Output()
    slice_slider = widgets.IntSlider(
        value=default_slice,
        min=0,
        max=max_slice,
        step=1,
        description="Slice",
        continuous_update=False,
    )
    zoom_slider = widgets.FloatSlider(
        value=1.0,
        min=1.0,
        max=3.0,
        step=0.1,
        description="Zoom",
        readout_format=".1f",
        continuous_update=False,
    )
    shift_x_slider = widgets.IntSlider(
        value=0,
        min=-100,
        max=100,
        step=2,
        description="Move X",
        continuous_update=False,
    )
    shift_y_slider = widgets.IntSlider(
        value=0,
        min=-100,
        max=100,
        step=2,
        description="Move Y",
        continuous_update=False,
    )
    minus_button = widgets.Button(description="-", layout=widgets.Layout(width="40px"))
    plus_button = widgets.Button(description="+", layout=widgets.Layout(width="40px"))
    ct_preset = widgets.Dropdown(
        options=["Default (auto)", "Soft tissue", "Bone"],
        value="Soft tissue",
        description="CT preset",
    )
    level_slider = widgets.IntSlider(
        value=40,
        min=-300,
        max=500,
        step=5,
        description="Level",
        continuous_update=False,
    )
    width_slider = widgets.IntSlider(
        value=400,
        min=50,
        max=2500,
        step=10,
        description="Width",
        continuous_update=False,
    )

    def apply_ct_preset(preset_name):
        if preset_name == "Soft tissue":
            level_slider.value = 40
            width_slider.value = 400
        elif preset_name == "Bone":
            level_slider.value = 300
            width_slider.value = 1500

    def render_view():
        slice_idx = slice_slider.value
        image_slice = np.rot90(postop[:, :, slice_idx])
        label_slice = np.rot90(postop_pred[:, :, slice_idx]).astype(int)
        image_slice, label_slice = _apply_zoom_and_pan(
            image_slice,
            label_slice,
            zoom_slider.value,
            shift_x_slider.value,
            shift_y_slider.value,
        )

        if ct_preset.value == "Default (auto)":
            imshow_kwargs = {}
        else:
            vmin, vmax = _window_limits(level_slider.value, width_slider.value)
            imshow_kwargs = {"vmin": vmin, "vmax": vmax}

        present_ids = sorted(int(v) for v in np.unique(label_slice) if v > 0)

        with output:
            output.clear_output(wait=True)
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))

            ax[0].imshow(image_slice, cmap="gray", **imshow_kwargs)
            ax[0].set_title(f"CT image (slice {slice_idx})")
            ax[0].axis("off")

            label_masked = np.ma.masked_where(label_slice == 0, label_slice)
            ax[1].imshow(image_slice, cmap="gray", **imshow_kwargs)
            ax[1].imshow(label_masked, cmap=cmap, norm=norm, alpha=0.5)
            ax[1].set_title("Predicted OAR labels")
            ax[1].axis("off")

            if not present_ids:
                ax[1].text(
                    0.5,
                    0.05,
                    "No OARs in this slice",
                    transform=ax[1].transAxes,
                    ha="center",
                    va="bottom",
                    color="white",
                    bbox={"facecolor": "black", "alpha": 0.5, "pad": 3},
                )

            plt.tight_layout()
            plt.show()

    def on_minus(_):
        slice_slider.value = max(slice_slider.min, slice_slider.value - 1)

    def on_plus(_):
        slice_slider.value = min(slice_slider.max, slice_slider.value + 1)

    def on_view_change(_):
        render_view()

    def on_preset_change(change):
        if change["name"] == "value":
            apply_ct_preset(change["new"])
            render_view()

    minus_button.on_click(on_minus)
    plus_button.on_click(on_plus)
    slice_slider.observe(on_view_change, names="value")
    zoom_slider.observe(on_view_change, names="value")
    shift_x_slider.observe(on_view_change, names="value")
    shift_y_slider.observe(on_view_change, names="value")
    level_slider.observe(on_view_change, names="value")
    width_slider.observe(on_view_change, names="value")
    ct_preset.observe(on_preset_change, names="value")

    apply_ct_preset(ct_preset.value)

    # Show a stable legend with the same colors used in the overlay.
    with legend_output:
        legend_rows = ""
        for label_id in sorted(roi_dict.keys()):
            color_hex = _rgba_to_hex(cmap(label_id))
            name = html.escape(roi_dict[label_id])
            legend_rows += (
                "<div style='display:flex; align-items:center; gap:8px; margin:2px 0;'>"
                f"<span style='display:inline-block; width:14px; height:14px; border:1px solid #666; background:{color_hex};'></span>"
                f"<span>{name}</span>"
                "</div>"
            )

        display(
            HTML(
                f"""
<div style='border:1px solid #ddd; border-radius:8px; padding:10px; max-width:900px;'>
  <div style='font-weight:600; margin-bottom:6px;'>Legend: OAR color mapping</div>
  <div style='column-count:2; column-gap:24px;'>
    {legend_rows}
  </div>
</div>
"""
            )
        )

    display(
        widgets.VBox(
            [
                widgets.HBox([minus_button, plus_button, slice_slider, zoom_slider]),
                widgets.HBox([shift_x_slider, shift_y_slider]),
                widgets.HBox([ct_preset, level_slider, width_slider]),
            ]
        )
    )
    display(output)
    display(legend_output)
    render_view()


def launch_metal_viewer(
    case_dict,
    widgets,
    has_widgets,
    model_a_name="Metal-trained model",
    model_b_name="No-metal-trained model",
):
    """Interactive two-model viewer with hidden callbacks and optional neutral model labels."""
    if not has_widgets:
        print("ipywidgets is not available. Install it with: pip install ipywidgets")
        return

    case_names = list(case_dict.keys())
    if not case_names:
        print("No cases available for viewer.")
        return

    default_case = case_names[0]
    state = {"case_name": default_case}
    output = widgets.Output()
    case_selector = widgets.Dropdown(
        options=case_names,
        value=default_case,
        description="Case",
    )
    first_img = case_dict[state["case_name"]][0]
    slice_slider = widgets.IntSlider(
        value=first_img.shape[2] // 2,
        min=0,
        max=first_img.shape[2] - 1,
        step=1,
        description="Slice",
        continuous_update=False,
    )
    minus_button = widgets.Button(description="-", layout=widgets.Layout(width="40px"))
    plus_button = widgets.Button(description="+", layout=widgets.Layout(width="40px"))
    ct_preset = widgets.Dropdown(
        options=["Default (auto)", "Soft tissue", "Bone"],
        value="Soft tissue",
        description="CT preset",
    )
    level_slider = widgets.IntSlider(
        value=40,
        min=-300,
        max=500,
        step=5,
        description="Level",
        continuous_update=False,
    )
    width_slider = widgets.IntSlider(
        value=400,
        min=50,
        max=2500,
        step=10,
        description="Width",
        continuous_update=False,
    )

    def apply_ct_preset(preset_name):
        if preset_name == "Soft tissue":
            level_slider.value = 40
            width_slider.value = 400
        elif preset_name == "Bone":
            level_slider.value = 300
            width_slider.value = 1500

    def render_view():
        img, pred_metal, pred_no_metal = case_dict[state["case_name"]]
        slice_idx = min(slice_slider.value, img.shape[2] - 1)
        if slice_slider.value != slice_idx:
            slice_slider.value = slice_idx
            return
        image_slice = np.rot90(img[:, :, slice_idx])
        metal_slice = np.rot90(pred_metal[:, :, slice_idx])
        no_metal_slice = None if pred_no_metal is None else np.rot90(pred_no_metal[:, :, slice_idx])

        if ct_preset.value == "Default (auto)":
            imshow_kwargs = {}
        else:
            vmin, vmax = _window_limits(level_slider.value, width_slider.value)
            imshow_kwargs = {"vmin": vmin, "vmax": vmax}

        with output:
            output.clear_output(wait=True)
            fig, ax = plt.subplots(1, 3, figsize=(16, 5))

            ax[0].imshow(image_slice, cmap="gray", **imshow_kwargs)
            ax[0].set_title(f"Input ({state['case_name']})")
            ax[0].axis("off")

            ax[1].imshow(image_slice, cmap="gray", **imshow_kwargs)
            ax[1].imshow(metal_slice, cmap="jet", alpha=0.35)
            ax[1].set_title(model_a_name)
            ax[1].axis("off")

            ax[2].imshow(image_slice, cmap="gray", **imshow_kwargs)
            if no_metal_slice is not None:
                ax[2].imshow(no_metal_slice, cmap="jet", alpha=0.35)
                ax[2].set_title(model_b_name)
            else:
                ax[2].set_title(f"{model_b_name} (missing)")
            ax[2].axis("off")

            plt.tight_layout()
            plt.show()

    def on_case_change(change):
        if change["name"] == "value":
            state["case_name"] = change["new"]
            current_img = case_dict[state["case_name"]][0]
            slice_slider.max = current_img.shape[2] - 1
            slice_slider.value = current_img.shape[2] // 2

    def on_slice_change(_):
        render_view()

    def on_preset_change(change):
        if change["name"] == "value":
            apply_ct_preset(change["new"])
            render_view()

    def on_minus(_):
        slice_slider.value = max(slice_slider.min, slice_slider.value - 1)

    def on_plus(_):
        slice_slider.value = min(slice_slider.max, slice_slider.value + 1)

    case_selector.observe(on_case_change, names="value")
    slice_slider.observe(on_slice_change, names="value")
    level_slider.observe(on_slice_change, names="value")
    width_slider.observe(on_slice_change, names="value")
    ct_preset.observe(on_preset_change, names="value")
    minus_button.on_click(on_minus)
    plus_button.on_click(on_plus)

    apply_ct_preset(ct_preset.value)

    display(widgets.HBox([minus_button, plus_button, slice_slider, case_selector]))
    display(widgets.HBox([ct_preset, level_slider, width_slider]))
    display(output)
    render_view()


def _dice_binary(pred_mask, gt_mask):
    """Compute Dice for two boolean masks."""
    pred_sum = pred_mask.sum()
    gt_sum = gt_mask.sum()
    if pred_sum == 0 and gt_sum == 0:
        return None
    if pred_sum + gt_sum == 0:
        return 0.0
    inter = np.logical_and(pred_mask, gt_mask).sum()
    return (2.0 * inter) / (pred_sum + gt_sum)


def render_dice_table_per_oar(postop_pred, postop_gt, roi_dict):
    """Render a resident-friendly Dice table per OAR."""
    rows_html = ""
    for label_id in sorted(roi_dict.keys()):
        name = html.escape(roi_dict[label_id])
        pred_mask = postop_pred == label_id
        gt_mask = postop_gt == label_id
        dice = _dice_binary(pred_mask, gt_mask)

        pred_present = "Yes" if pred_mask.any() else "No"
        gt_present = "Yes" if gt_mask.any() else "No"
        if dice is None:
            dice_txt = "N/A"
            score_color = "#777"
        else:
            dice_txt = f"{dice:.3f}"
            score_color = "#0a7d34" if dice >= 0.8 else "#b26a00" if dice >= 0.5 else "#b42318"

        rows_html += (
            "<tr>"
            f"<td style='padding:6px; border-bottom:1px solid #eee;'>{name}</td>"
            f"<td style='padding:6px; border-bottom:1px solid #eee; text-align:center;'>{pred_present}</td>"
            f"<td style='padding:6px; border-bottom:1px solid #eee; text-align:center;'>{gt_present}</td>"
            f"<td style='padding:6px; border-bottom:1px solid #eee; text-align:center; color:{score_color}; font-weight:600;'>{dice_txt}</td>"
            "</tr>"
        )

    display(
        HTML(
            f"""
<div style='border:1px solid #ddd; border-radius:8px; padding:10px; max-width:980px;'>
  <div style='font-weight:600; margin-bottom:8px;'>Per-OAR Dice: predicted vs clinical contour</div>
  <table style='border-collapse:collapse; width:100%; font-size:14px;'>
    <tr>
      <th style='text-align:left; padding:6px; border-bottom:1px solid #ccc;'>OAR</th>
      <th style='text-align:center; padding:6px; border-bottom:1px solid #ccc;'>Pred present</th>
      <th style='text-align:center; padding:6px; border-bottom:1px solid #ccc;'>Clinical present</th>
      <th style='text-align:center; padding:6px; border-bottom:1px solid #ccc;'>Dice</th>
    </tr>
    {rows_html}
  </table>
</div>
"""
        )
    )


def render_dice_table_two_cases(
    pred_case1,
    gt_case1,
    pred_case2,
    gt_case2,
    roi_dict,
    case1_name="Case 1",
    case2_name="Case 2",
):
    """Render one combined Dice table for two cases (one row per OAR)."""

    def _format_score(dice_value):
        if dice_value is None:
            return "N/A", "#777"
        score_text = f"{dice_value:.3f}"
        score_color = "#0a7d34" if dice_value >= 0.8 else "#b26a00" if dice_value >= 0.5 else "#b42318"
        return score_text, score_color

    rows_html = ""
    for label_id in sorted(roi_dict.keys()):
        name = html.escape(roi_dict[label_id])

        dice_case1 = None
        if pred_case1 is not None and gt_case1 is not None:
            dice_case1 = _dice_binary(pred_case1 == label_id, gt_case1 == label_id)

        dice_case2 = None
        if pred_case2 is not None and gt_case2 is not None:
            dice_case2 = _dice_binary(pred_case2 == label_id, gt_case2 == label_id)

        score_1, color_1 = _format_score(dice_case1)
        score_2, color_2 = _format_score(dice_case2)

        rows_html += (
            "<tr>"
            f"<td style='padding:6px; border-bottom:1px solid #eee;'>{name}</td>"
            f"<td style='padding:6px; border-bottom:1px solid #eee; text-align:center; color:{color_1}; font-weight:600;'>{score_1}</td>"
            f"<td style='padding:6px; border-bottom:1px solid #eee; text-align:center; color:{color_2}; font-weight:600;'>{score_2}</td>"
            "</tr>"
        )

    display(
        HTML(
            f"""
<div style='border:1px solid #ddd; border-radius:8px; padding:10px; max-width:980px;'>
  <div style='font-weight:600; margin-bottom:8px;'>Per-OAR Dice comparison across two cases</div>
  <table style='border-collapse:collapse; width:100%; font-size:14px;'>
    <tr>
      <th style='text-align:left; padding:6px; border-bottom:1px solid #ccc;'>OAR</th>
      <th style='text-align:center; padding:6px; border-bottom:1px solid #ccc;'>{html.escape(case1_name)} Dice</th>
      <th style='text-align:center; padding:6px; border-bottom:1px solid #ccc;'>{html.escape(case2_name)} Dice</th>
    </tr>
    {rows_html}
  </table>
</div>
"""
        )
    )


def render_dice_table_two_models_two_cases(
    pred_model_a_case1,
    pred_model_a_case2,
    pred_model_b_case1,
    pred_model_b_case2,
    gt_case1,
    gt_case2,
    roi_dict,
    model_a_name="Model A",
    model_b_name="Model B",
    case1_name="Case 1",
    case2_name="Case 2",
):
    """Render one Dice table for two models across two cases."""

    def _format_score(dice_value):
        if dice_value is None:
            return "N/A", "#777"
        score_text = f"{dice_value:.3f}"
        score_color = "#0a7d34" if dice_value >= 0.8 else "#b26a00" if dice_value >= 0.5 else "#b42318"
        return score_text, score_color

    rows_html = ""
    for label_id in sorted(roi_dict.keys()):
        name = html.escape(roi_dict[label_id])

        dice_a_1 = None
        dice_a_2 = None
        dice_b_1 = None
        dice_b_2 = None

        if pred_model_a_case1 is not None and gt_case1 is not None:
            dice_a_1 = _dice_binary(pred_model_a_case1 == label_id, gt_case1 == label_id)
        if pred_model_a_case2 is not None and gt_case2 is not None:
            dice_a_2 = _dice_binary(pred_model_a_case2 == label_id, gt_case2 == label_id)
        if pred_model_b_case1 is not None and gt_case1 is not None:
            dice_b_1 = _dice_binary(pred_model_b_case1 == label_id, gt_case1 == label_id)
        if pred_model_b_case2 is not None and gt_case2 is not None:
            dice_b_2 = _dice_binary(pred_model_b_case2 == label_id, gt_case2 == label_id)

        score_a_1, color_a_1 = _format_score(dice_a_1)
        score_a_2, color_a_2 = _format_score(dice_a_2)
        score_b_1, color_b_1 = _format_score(dice_b_1)
        score_b_2, color_b_2 = _format_score(dice_b_2)

        rows_html += (
            "<tr>"
            f"<td style='padding:6px; border-bottom:1px solid #eee;'>{name}</td>"
            f"<td style='padding:6px; border-bottom:1px solid #eee; text-align:center; color:{color_a_1}; font-weight:600;'>{score_a_1}</td>"
            f"<td style='padding:6px; border-bottom:1px solid #eee; text-align:center; color:{color_a_2}; font-weight:600;'>{score_a_2}</td>"
            f"<td style='padding:6px; border-bottom:1px solid #eee; text-align:center; color:{color_b_1}; font-weight:600;'>{score_b_1}</td>"
            f"<td style='padding:6px; border-bottom:1px solid #eee; text-align:center; color:{color_b_2}; font-weight:600;'>{score_b_2}</td>"
            "</tr>"
        )

    display(
        HTML(
            f"""
<div style='border:1px solid #ddd; border-radius:8px; padding:10px; max-width:1100px;'>
  <div style='font-weight:600; margin-bottom:8px;'>Per-OAR Dice comparison across two models and two cases</div>
  <table style='border-collapse:collapse; width:100%; font-size:14px;'>
    <tr>
      <th style='text-align:left; padding:6px; border-bottom:1px solid #ccc;'>OAR</th>
      <th style='text-align:center; padding:6px; border-bottom:1px solid #ccc;'>{html.escape(model_a_name)} - {html.escape(case1_name)}</th>
      <th style='text-align:center; padding:6px; border-bottom:1px solid #ccc;'>{html.escape(model_a_name)} - {html.escape(case2_name)}</th>
      <th style='text-align:center; padding:6px; border-bottom:1px solid #ccc;'>{html.escape(model_b_name)} - {html.escape(case1_name)}</th>
      <th style='text-align:center; padding:6px; border-bottom:1px solid #ccc;'>{html.escape(model_b_name)} - {html.escape(case2_name)}</th>
    </tr>
    {rows_html}
  </table>
</div>
"""
        )
    )


def launch_pred_vs_clinical_viewer(postop, postop_pred, postop_gt, roi_dict, widgets, has_widgets):
    """Interactive visual comparison of prediction vs clinical contour."""
    if not has_widgets:
        print("ipywidgets is not available. Install it with: pip install ipywidgets")
        print("Then restart the kernel and run this cell again.")
        return

    max_slice = postop.shape[2] - 1
    label_counts = (postop_gt > 0).sum(axis=(0, 1))
    default_slice = int(np.argmax(label_counts)) if np.any(label_counts) else max_slice // 2
    cmap, norm, _ = _build_label_colormap(roi_dict)

    output = widgets.Output()
    slice_slider = widgets.IntSlider(
        value=default_slice,
        min=0,
        max=max_slice,
        step=1,
        description="Slice",
        continuous_update=False,
    )
    zoom_slider = widgets.FloatSlider(
        value=1.0,
        min=1.0,
        max=3.0,
        step=0.1,
        description="Zoom",
        readout_format=".1f",
        continuous_update=False,
    )
    shift_x_slider = widgets.IntSlider(
        value=0,
        min=-100,
        max=100,
        step=2,
        description="Move X",
        continuous_update=False,
    )
    shift_y_slider = widgets.IntSlider(
        value=0,
        min=-100,
        max=100,
        step=2,
        description="Move Y",
        continuous_update=False,
    )
    minus_button = widgets.Button(description="-", layout=widgets.Layout(width="40px"))
    plus_button = widgets.Button(description="+", layout=widgets.Layout(width="40px"))
    ct_preset = widgets.Dropdown(
        options=["Default (auto)", "Soft tissue", "Bone"],
        value="Soft tissue",
        description="CT preset",
    )
    level_slider = widgets.IntSlider(
        value=40,
        min=-300,
        max=500,
        step=5,
        description="Level",
        continuous_update=False,
    )
    width_slider = widgets.IntSlider(
        value=400,
        min=50,
        max=2500,
        step=10,
        description="Width",
        continuous_update=False,
    )

    def apply_ct_preset(preset_name):
        if preset_name == "Soft tissue":
            level_slider.value = 40
            width_slider.value = 400
        elif preset_name == "Bone":
            level_slider.value = 300
            width_slider.value = 1500

    def render_view():
        slice_idx = slice_slider.value
        image_raw = np.rot90(postop[:, :, slice_idx])
        pred_raw = np.rot90(postop_pred[:, :, slice_idx]).astype(int)
        gt_raw = np.rot90(postop_gt[:, :, slice_idx]).astype(int)

        image_slice, pred_slice = _apply_zoom_and_pan(
            image_raw,
            pred_raw,
            zoom_slider.value,
            shift_x_slider.value,
            shift_y_slider.value,
        )
        _, gt_slice = _apply_zoom_and_pan(
            image_raw,
            gt_raw,
            zoom_slider.value,
            shift_x_slider.value,
            shift_y_slider.value,
        )

        if ct_preset.value == "Default (auto)":
            imshow_kwargs = {}
        else:
            vmin, vmax = _window_limits(level_slider.value, width_slider.value)
            imshow_kwargs = {"vmin": vmin, "vmax": vmax}

        with output:
            output.clear_output(wait=True)
            fig, ax = plt.subplots(1, 3, figsize=(16, 6))

            ax[0].imshow(image_slice, cmap="gray", **imshow_kwargs)
            ax[0].set_title(f"CT image (slice {slice_idx})")
            ax[0].axis("off")

            pred_masked = np.ma.masked_where(pred_slice == 0, pred_slice)
            ax[1].imshow(image_slice, cmap="gray", **imshow_kwargs)
            ax[1].imshow(pred_masked, cmap=cmap, norm=norm, alpha=0.5)
            ax[1].set_title("Predicted segmentation")
            ax[1].axis("off")

            gt_masked = np.ma.masked_where(gt_slice == 0, gt_slice)
            ax[2].imshow(image_slice, cmap="gray", **imshow_kwargs)
            ax[2].imshow(gt_masked, cmap=cmap, norm=norm, alpha=0.5)
            ax[2].set_title("Clinical contour (reference)")
            ax[2].axis("off")

            plt.tight_layout()
            plt.show()

    def on_minus(_):
        slice_slider.value = max(slice_slider.min, slice_slider.value - 1)

    def on_plus(_):
        slice_slider.value = min(slice_slider.max, slice_slider.value + 1)

    def on_view_change(_):
        render_view()

    def on_preset_change(change):
        if change["name"] == "value":
            apply_ct_preset(change["new"])
            render_view()

    minus_button.on_click(on_minus)
    plus_button.on_click(on_plus)
    slice_slider.observe(on_view_change, names="value")
    zoom_slider.observe(on_view_change, names="value")
    shift_x_slider.observe(on_view_change, names="value")
    shift_y_slider.observe(on_view_change, names="value")
    level_slider.observe(on_view_change, names="value")
    width_slider.observe(on_view_change, names="value")
    ct_preset.observe(on_preset_change, names="value")

    apply_ct_preset(ct_preset.value)

    display(
        widgets.VBox(
            [
                widgets.HBox([minus_button, plus_button, slice_slider, zoom_slider]),
                widgets.HBox([shift_x_slider, shift_y_slider]),
                widgets.HBox([ct_preset, level_slider, width_slider]),
            ]
        )
    )
    display(output)
    render_view()
