#!/usr/bin/env python3
# ByteTrack-3D: Multi-object tracking in 3D using PointPillars detections.
# Adapted from ByteTrack (Zhang et al., 2022) for 3D LiDAR point cloud data.
# Pipeline: RS-LiDAR-16 -> PointPillars TRT -> ByteTrack3D -> Track IDs

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from scipy.optimize import linear_sum_assignment
from collections import OrderedDict


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Box3D:
    # 3D bounding box in LiDAR frame
    x: float
    y: float
    z: float
    l: float
    w: float
    h: float
    yaw: float
    score: float
    cls_id: int

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z,
                         self.l, self.w, self.h, self.yaw], dtype=np.float32)


@dataclass
class Track3D:
    # Represents a tracked 3D object across frames
    track_id: int
    box: Box3D
    age: int = 0
    hits: int = 1
    misses: int = 0
    state: str = 'tentative'
    history: List[Box3D] = field(default_factory=list)

    def update(self, det: Box3D) -> None:
        alpha = 0.6
        self.box.x = alpha * det.x + (1 - alpha) * self.box.x
        self.box.y = alpha * det.y + (1 - alpha) * self.box.y
        self.box.z = alpha * det.z + (1 - alpha) * self.box.z
        self.box.l = alpha * det.l + (1 - alpha) * self.box.l
        self.box.w = alpha * det.w + (1 - alpha) * self.box.w
        self.box.h = alpha * det.h + (1 - alpha) * self.box.h
        diff = det.yaw - self.box.yaw
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        self.box.yaw += alpha * diff
        self.box.score = det.score
        self.hits += 1
        self.misses = 0
        self.history.append(Box3D(**vars(det)))
        if len(self.history) > 30:
            self.history.pop(0)
        if self.hits >= 3:
            self.state = 'confirmed'

    def predict(self) -> None:
        self.age += 1
        self.misses += 1
        if self.misses > 5:
            self.state = 'lost'


# ---------------------------------------------------------------------------
# 3D IoU utilities
# ---------------------------------------------------------------------------

def rotate_corners_2d(cx, cy, l, w, yaw):
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    half_l, half_w = l / 2.0, w / 2.0
    local = np.array([[ half_l,  half_w],[ half_l, -half_w],
                      [-half_l, -half_w],[-half_l,  half_w]])
    rot = np.array([[cos_y, -sin_y],[sin_y,  cos_y]])
    return local @ rot.T + np.array([cx, cy])


def polygon_area(corners):
    n = len(corners)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i, 0] * corners[j, 1]
        area -= corners[j, 0] * corners[i, 1]
    return abs(area) / 2.0


def clip_polygon(subject, clip):
    # Sutherland-Hodgman polygon clipping
    def inside(p, a, b):
        return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0]) >= 0
    def intersect(p1, p2, p3, p4):
        d1, d2 = p2-p1, p4-p3
        cross = d1[0]*d2[1] - d1[1]*d2[0]
        if abs(cross) < 1e-10: return p1
        t = ((p3[0]-p1[0])*d2[1] - (p3[1]-p1[1])*d2[0]) / cross
        return p1 + t*d1
    output = list(subject)
    for i in range(len(clip)):
        if not output: break
        input_list = output; output = []
        a = clip[(i-1) % len(clip)]; b = clip[i]
        for j in range(len(input_list)):
            cur = input_list[j]; prev = input_list[j-1]
            if inside(cur, a, b):
                if not inside(prev, a, b): output.append(intersect(prev, cur, a, b))
                output.append(cur)
            elif inside(prev, a, b):
                output.append(intersect(prev, cur, a, b))
    return np.array(output) if output else np.empty((0, 2))


def bev_iou_3d(box_a, box_b):
    ca = rotate_corners_2d(box_a.x, box_a.y, box_a.l, box_a.w, box_a.yaw)
    cb = rotate_corners_2d(box_b.x, box_b.y, box_b.l, box_b.w, box_b.yaw)
    clipped = clip_polygon(ca, cb)
    if len(clipped) < 3: return 0.0
    inter_area = polygon_area(clipped)
    union_area = box_a.l*box_a.w + box_b.l*box_b.w - inter_area
    z_a_min, z_a_max = box_a.z - box_a.h/2, box_a.z + box_a.h/2
    z_b_min, z_b_max = box_b.z - box_b.h/2, box_b.z + box_b.h/2
    h_overlap = max(0, min(z_a_max, z_b_max) - max(z_a_min, z_b_min))
    h_union = max(z_a_max, z_b_max) - min(z_a_min, z_b_min)
    return (inter_area / (union_area + 1e-8)) * (h_overlap / (h_union + 1e-8))


def build_iou_matrix(tracks, detections):
    iou_mat = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    for i, trk in enumerate(tracks):
        for j, det in enumerate(detections):
            iou_mat[i, j] = bev_iou_3d(trk.box, det)
    return iou_mat


def hungarian_match(cost_matrix, threshold=0.25):
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    matched, unmatched_tracks, unmatched_dets = [], [], []
    matched_cols = set()
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] >= threshold:
            matched.append((r, c)); matched_cols.add(c)
        else:
            unmatched_tracks.append(r)
    unmatched_tracks += [r for r in range(cost_matrix.shape[0])
                         if r not in {m[0] for m in matched} and r not in unmatched_tracks]
    unmatched_dets = [c for c in range(cost_matrix.shape[1]) if c not in matched_cols]
    return matched, unmatched_tracks, unmatched_dets


# ---------------------------------------------------------------------------
# ByteTracker3D
# ---------------------------------------------------------------------------

class ByteTracker3D:
    # ByteTrack-style 3D MOT for PointPillars LiDAR detections
    def __init__(self, high_thresh=0.5, low_thresh=0.1, iou_thresh=0.25, max_age=5, min_hits=3):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self._next_id = 1
        self.tracks = []
        self.frame_count = 0

    def _new_track(self, det):
        trk = Track3D(track_id=self._next_id, box=Box3D(**vars(det)))
        self._next_id += 1
        return trk

    def update(self, detections):
        self.frame_count += 1
        for trk in self.tracks: trk.predict()
        high_dets = [d for d in detections if d.score >= self.high_thresh]
        low_dets  = [d for d in detections if self.low_thresh <= d.score < self.high_thresh]
        active_tracks = [t for t in self.tracks if t.state != 'lost']
        iou_h = build_iou_matrix(active_tracks, high_dets)
        matched_h, unmatched_t_h, unmatched_d_h = hungarian_match(iou_h, self.iou_thresh)
        for ti, di in matched_h: active_tracks[ti].update(high_dets[di])
        rem_tracks = [active_tracks[i] for i in unmatched_t_h]
        iou_l = build_iou_matrix(rem_tracks, low_dets)
        matched_l, _, _ = hungarian_match(iou_l, self.iou_thresh)
        for ti, di in matched_l: rem_tracks[ti].update(low_dets[di])
        for di in unmatched_d_h: self.tracks.append(self._new_track(high_dets[di]))
        self.tracks = [t for t in self.tracks
                       if not (t.state == 'lost' and t.misses > self.max_age)]
        return [t for t in self.tracks if t.state != 'lost']

    def get_confirmed_tracks(self):
        return [t for t in self.tracks if t.state == 'confirmed']

    def reset(self):
        self.tracks.clear(); self._next_id = 1; self.frame_count = 0


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    np.random.seed(0)
    tracker = ByteTracker3D()
    print("ByteTracker3D Demo - 10 frames, 3 objects")
    for frame_idx in range(10):
        dets = [
            Box3D(x=2.0+frame_idx*0.5, y=0.0, z=0.9, l=4.5, w=2.0, h=1.8, yaw=0.0, score=0.9, cls_id=0),
            Box3D(x=-1.0, y=3.0+frame_idx*0.3, z=0.6, l=0.6, w=0.6, h=1.7, yaw=0.1, score=0.7, cls_id=1),
            Box3D(x=5.0, y=-2.0, z=0.5, l=1.0, w=0.8, h=0.5, yaw=0.3,
                  score=0.3 if frame_idx % 3 == 0 else 0.6, cls_id=2),
        ]
        active = tracker.update(dets)
        print(f"Frame {frame_idx:02d}: {len(active)} active confirmed={len(tracker.get_confirmed_tracks())}")
        for t in active:
            print(f"  Track {t.track_id} cls={t.box.cls_id} state={t.state} hits={t.hits}")
