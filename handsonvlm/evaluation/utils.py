import cv2
import numpy as np


def create_gradient_color(start_color, end_color, steps):
    gradient = []
    for i in range(steps):
        r = start_color[0] + (end_color[0] - start_color[0]) * i / (steps - 1)
        g = start_color[1] + (end_color[1] - start_color[1]) * i / (steps - 1)
        b = start_color[2] + (end_color[2] - start_color[2]) * i / (steps - 1)
        gradient.append((int(r), int(g), int(b)))
    return gradient


def vis_hand_traj(frame_vis, hand_traj, shape, style='arrow', circle_radius=4, circle_thickness=3,
                  line_thickness=2, gap=5, hand_rgb={"LEFT": (0, 90, 181), "RIGHT": (220, 50, 32)},
                  arrow_length=20, arrow_angle=30):
    width, height = shape
    output = frame_vis.copy()

    for side in hand_traj:
        traj = hand_traj[side].copy()
        traj[:, 0] = traj[:, 0] * width
        traj[:, 1] = traj[:, 1] * height

        if len(traj) > 0:
            if style == 'gradient':
                start_color = hand_rgb[side][::-1]
                end_color = tuple(int(c * 0.5) for c in start_color)
                colors = create_gradient_color(start_color, end_color, len(traj))
            else:
                colors = [hand_rgb[side][::-1]] * len(traj)

            output = vis_traj(output, traj, colors, style=style,
                              circle_radius=circle_radius,
                              circle_thickness=circle_thickness,
                              line_thickness=line_thickness,
                              gap=gap,
                              arrow_length=arrow_length,
                              arrow_angle=arrow_angle)

    return output


def draw_arrow(img, pt1, pt2, color, thickness, arrow_length=20, arrow_angle=30):
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    angle = np.arctan2(dy, dx)

    angle_left = angle + np.deg2rad(arrow_angle)
    angle_right = angle - np.deg2rad(arrow_angle)

    arrow_pt1 = (int(pt2[0] - arrow_length * np.cos(angle_left)),
                 int(pt2[1] - arrow_length * np.sin(angle_left)))
    arrow_pt2 = (int(pt2[0] - arrow_length * np.cos(angle_right)),
                 int(pt2[1] - arrow_length * np.sin(angle_right)))

    cv2.line(img, pt1, pt2, color, thickness)
    cv2.line(img, pt2, arrow_pt1, color, thickness)
    cv2.line(img, pt2, arrow_pt2, color, thickness)


def vis_traj(frame_vis, traj, colors, style='mixed', circle_radius=4, circle_thickness=3,
             line_thickness=2, gap=5, arrow_length=20, arrow_angle=30):
    overlay = frame_vis.copy()

    if len(traj) <= 0:
        return frame_vis

    # 绘制轨迹
    for idx, (x, y) in enumerate(traj):
        pt = (int(round(x)), int(round(y)))

        if idx > 0:
            pt1 = (int(round(traj[idx - 1][0])), int(round(traj[idx - 1][1])))
            pt2 = pt
            cv2.line(overlay, pt1, pt2, colors[idx], line_thickness)

    if len(traj) > 0:
        start_pt = (int(round(traj[0][0])), int(round(traj[0][1])))
        cv2.circle(overlay, start_pt, radius=circle_radius, color=colors[0], thickness=-1)

        if len(traj) > 1:
            end_pt = (int(round(traj[-1][0])), int(round(traj[-1][1])))
            prev_pt = (int(round(traj[-2][0])), int(round(traj[-2][1])))
            draw_arrow(overlay, prev_pt, end_pt, colors[-1], line_thickness,
                       arrow_length, arrow_angle)
    cv2.addWeighted(overlay, 1, frame_vis, 0, 0, frame_vis)
    return frame_vis


def create_trajectory_video(image_paths, pred_hand_trajectory, output_path):

    frame_list = []
    print(f"Processing {len(image_paths)} original frames...")
    last_original_frame = None
    for i, local_path in enumerate(image_paths):
        frame = cv2.imread(local_path)
        if frame is None:
            raise ValueError(f"Failed to load image: {local_path}")
        frame = cv2.resize(frame, (960, 540))
        if frame.shape[2] == 3:
            b, g, r = cv2.split(frame)
            alpha = np.ones(b.shape, dtype=b.dtype) * 255
            frame = cv2.merge((b, g, r, alpha))

        alpha = 0.5
        beta = 1 - alpha
        if i == len(image_paths) - 1:
            last_original_frame = frame.copy()
            white_bg = np.ones((540, 960, 3), dtype=np.uint8) * 255
            last_original_frame = cv2.addWeighted(frame[:, :, :3], alpha, white_bg, beta, 0)
            b, g, r = cv2.split(last_original_frame)
            alpha = np.ones(b.shape, dtype=b.dtype) * 255
            last_original_frame = cv2.merge((b, g, r, alpha))
        frame_with_text = frame.copy()

        frame_list.append(frame_with_text)

    for t in range(pred_hand_trajectory.shape[-2]):
        base_frame = last_original_frame.copy()
        current_pred = {
            "RIGHT": pred_hand_trajectory[0, 0, 0, :t + 1, :].cpu().numpy(),
            "LEFT": pred_hand_trajectory[0, 0, 1, :t + 1, :].cpu().numpy()
        }
        frame_with_traj = vis_hand_traj(
            base_frame.copy(),
            current_pred,
            (960, 540),
            style='arrow',
            circle_radius=16,
            circle_thickness=16,
            line_thickness=10,
            arrow_length=15,
            arrow_angle=30,
            hand_rgb={
                "LEFT": (0, 90, 181),
                "RIGHT": (220, 50, 32)
            }
        )
        frame_list.extend([frame_with_traj] * 2)    # duplicated for 0.5x speed

    print(f"Total frames to write: {len(frame_list)}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        output_path,
        fourcc,
        10.0,
        (960, 540)
    )

    for i, frame in enumerate(frame_list):
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to: {output_path}")
    return output_path