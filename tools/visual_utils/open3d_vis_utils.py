"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import matplotlib
import matplotlib.colors
import numpy as np
import open3d
import torch

box_colormap: list[list[float]] = [
    [1, 0, 0], #red - car
    [0, 1, 0], #green - truck
    [0, 0, 1], #blue - construction_vehicle
    [1, 1, 0], #yellow - bus
    [1, 0, 1], #magenta - trailer
    [0, 1, 1], #cyan - barrier
    [1, 0.5, 0], #orange - motorcycle
    [0.5, 0, 1], #violet - bicycle
    [0.5, 0.5, 0.5], #grey - pedestrian
    [0.5, 0.5, 0], #olive - traffic_cone
    [0, 0.5, 0.5], # dark teal - unused
    [1, 1, 1] #white - unused
]


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True, class_names=None):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores, class_names=class_names)

    vis.run()
    vis.destroy_window()


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None, class_names=None):
    has_labels = hasattr(vis, 'add_3d_label')
    if score is not None and not has_labels:
        # Using a static variable to ensure the message is printed only once per session
        if not getattr(draw_box, 'has_warned_about_labels', False):
            print("\nWarning: Your Open3D version does not support 3D labels. Scores and class names will not be displayed in the visualizer.\n")
            draw_box.has_warned_about_labels = True

    for i in range(gt_boxes.shape[0]):
        scr = None
        if score is not None:
            scr = float(score[i].item()) if isinstance(score[i], torch.Tensor) else float(score[i])
        
        if scr is None or scr > 0.1:
            line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
            text = f'{scr:.2f}'

            if ref_labels is None:
                line_set.paint_uniform_color(color)
            else:
                label_idx = int(ref_labels[i].item()) if isinstance(ref_labels[i], torch.Tensor) else int(ref_labels[i])
                line_set.paint_uniform_color(box_colormap[label_idx - 1])
                if class_names is not None:
                    if 0 <= (label_idx - 1) < len(class_names):
                        cls_name = class_names[label_idx - 1]
                        text = f'{cls_name}: {scr:.2f}, {box_colormap[label_idx - 1]}, {label_idx}'

            vis.add_geometry(line_set)

            corners = box3d.get_box_points()
            
            print(text)
            # 4. Add the label at corner 5 (top-front-right usually)
            if has_labels:
                vis.add_3d_label(corners[5], text)
            
    return vis