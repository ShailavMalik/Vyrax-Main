"""Lightweight single-face tracking helpers for realtime pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple
from collections import deque


@dataclass
class TrackState:
    """Keeps minimal state for one active face track."""

    track_id: int
    center: Tuple[float, float]
    box: Tuple[int, int, int, int]
    last_seen_frame: int
    emotion_history: Deque[Tuple[str, float]] = field(default_factory=lambda: deque(maxlen=10))


def box_center_xyxy(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def match_track_id(
    detection_center: Tuple[float, float],
    tracks: Dict[int, TrackState],
    match_distance_px: float = 45.0,
) -> Optional[int]:
    """Find nearest existing track within a distance gate."""
    best_id: Optional[int] = None
    best_distance = float(match_distance_px)

    for track_id, track in tracks.items():
        dx = detection_center[0] - track.center[0]
        dy = detection_center[1] - track.center[1]
        distance = math.hypot(dx, dy)
        if distance < best_distance:
            best_distance = distance
            best_id = track_id

    return best_id


def upsert_track(
    box: Tuple[int, int, int, int],
    frame_index: int,
    tracks: Dict[int, TrackState],
    next_track_id: int,
) -> Tuple[int, TrackState]:
    """Create or update a track from the current detection box."""
    center = box_center_xyxy(box)
    matched = match_track_id(center, tracks)
    if matched is None:
        track = TrackState(
            track_id=next_track_id,
            center=center,
            box=box,
            last_seen_frame=frame_index,
        )
        tracks[next_track_id] = track
        return next_track_id + 1, track

    track = tracks[matched]
    track.center = center
    track.box = box
    track.last_seen_frame = frame_index
    return next_track_id, track


def prune_stale_tracks(tracks: Dict[int, TrackState], frame_index: int, max_missing_frames: int = 20) -> None:
    """Drop tracks that have not been seen for too many frames."""
    stale_ids = [track_id for track_id, track in tracks.items() if (frame_index - track.last_seen_frame) > max_missing_frames]
    for track_id in stale_ids:
        del tracks[track_id]


def track_from_point(x: float, y: float, tracks: Dict[int, TrackState], match_distance_px: float = 45.0) -> Optional[int]:
    """Backward-compatible helper to match an arbitrary point to a track."""
    return match_track_id((x, y), tracks, match_distance_px=match_distance_px)
