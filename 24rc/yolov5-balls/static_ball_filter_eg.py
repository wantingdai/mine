# Author: Ethan Lee
# 2024/6/26 上午0:16

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

np.random.seed()


def generate_data(num_frames, num_objects, noise_level, movement_prob):
    data = []
    for _ in range(num_frames):
        frame_data = []
        for obj in range(num_objects):
            if np.random.random() < movement_prob:
                x = np.random.uniform(0, 100)
                y = np.random.uniform(0, 100)
            else:
                x = obj * 10 + np.random.normal(0, noise_level)
                y = 50 + np.random.normal(0, noise_level)

            if np.random.random() < 0.8:  # 20% chance of missing detection
                frame_data.append((x, y))
        data.append(frame_data)
    return data


def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def associate_detections_to_trackers(detections, trackers, max_distance=5.0):
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 2), dtype=int)

    distance_matrix = np.zeros((len(detections), len(trackers)))
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            distance_matrix[d, t] = euclidean_distance(det, trk)

    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    matched_indices = np.column_stack((row_ind, col_ind))

    unmatched_detections = [d for d in range(len(detections)) if d not in matched_indices[:, 0]]
    unmatched_trackers = [t for t in range(len(trackers)) if t not in matched_indices[:, 1]]

    matches = [m for m in matched_indices if distance_matrix[m[0], m[1]] <= max_distance]

    unmatched_detections.extend([m[0] for m in matched_indices if distance_matrix[m[0], m[1]] > max_distance])
    unmatched_trackers.extend([m[1] for m in matched_indices if distance_matrix[m[0], m[1]] > max_distance])

    return np.array(matches), np.array(unmatched_detections), np.array(unmatched_trackers)


class SimpleTracker:
    def __init__(self, point):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1, 0, 1, 0],
                              [0, 1, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])
        self.kf.R *= 10
        self.kf.P *= 1000
        self.kf.Q = np.eye(4) * 0.1
        self.kf.x = np.array([point[0], point[1], 0, 0])
        self.history = [self.kf.x[:2].copy()]

    def update(self, point):
        self.kf.update(point)
        self.history.append(self.kf.x[:2].copy())

    def predict(self):
        self.kf.predict()
        return self.kf.x[:2]


def filter_static_objects(data, distance_threshold=10.0, speed_threshold=10.0):
    trackers = []
    all_trackers = []

    for frame, detections in enumerate(data):
        predicted_trackers = [t.predict() for t in trackers]

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(detections, predicted_trackers)

        for det_idx, trk_idx in matched:
            trackers[trk_idx].update(detections[det_idx])

        for det_idx in unmatched_dets:
            new_tracker = SimpleTracker(detections[det_idx])
            trackers.append(new_tracker)
            all_trackers.append(new_tracker)

        trackers = [t for i, t in enumerate(trackers) if i not in unmatched_trks]

    static_objects = []
    for tracker in all_trackers:
        # if len(tracker.history) == 3:  # Only consider trackers that appeared in all 3 frames
        if len(tracker.history) >= 2:  # Only consider trackers that appeared in all 3 frames
            positions = np.array(tracker.history)

            # Distance-based check
            max_distance = np.max(np.linalg.norm(positions - positions.mean(axis=0), axis=1))

            # Speed-based check
            speeds = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
            avg_speed = np.mean(speeds)

            if max_distance < distance_threshold and avg_speed < speed_threshold:
                # static_objects.append(positions[-1])
                static_objects.append((int(positions[-1][0]), int(positions[-1][1])))

    return static_objects


if __name__ == '__main__':
    # Generate test data
    num_frames = 3
    num_objects = 2
    noise_level = 0.5
    movement_prob = 0.1

    data = generate_data(num_frames, num_objects, noise_level, movement_prob)

    # Filter static objects
    static_objects = filter_static_objects(data)

    # Visualize results
    plt.figure(figsize=(12, 6))

    # Plot all detections
    colors = ['blue', 'green', 'cyan']
    for i, frame_data in enumerate(data):
        x, y = zip(*frame_data) if frame_data else ([], [])
        plt.scatter(x, y, c=colors[i], alpha=0.5, s=30, label=f'Frame {i + 1}')

    # Plot static objects
    static_x, static_y = zip(*static_objects) if static_objects else ([], [])
    plt.scatter(static_x, static_y, c='red', s=100, marker='*', label='Static Objects')

    plt.title('Object Detections and Filtered Static Objects (3 Frames)')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Number of static objects detected: {len(static_objects)}")
    print("Static object coordinates:")
    for obj in static_objects:
        print(f"  X: {obj[0]:.2f}, Y: {obj[1]:.2f}")
