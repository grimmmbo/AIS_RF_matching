
import argparse
import math
import multiprocessing
import os
import pickle
import time
from datetime import datetime
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
import numpy as np
from haversine import haversine, Unit

from scripts.modeling.AIS_RF_alignment.AIS_RF_forward_alignment import (
    determine_max_workers,
    _load_checkpoint,
)

### RUNNING TIME ###
    # Baseline model: for each RF signal, matches the AIS track whose closest
    # point is nearest. Four distance metrics are provided as an ablation:
    # plain Euclidean (naive, no geodesic correction), Haversine (proper
    # great-circle distance to the nearest AIS point), point-to-segment
    # (perpendicular distance to the nearest AIS track leg, so an RF signal
    # landing between two AIS fixes isn't penalized for missing both), and
    # time-weighted (point-to-segment distance combined with an interpolated
    # time gap, so a spatially close but temporally distant AIS leg is
    # penalized). Used as lower bounds to compare the PHMM Forward alignment
    # against.

def euclidean_distance(point_a, point_b):
    """
    Plain Euclidean distance between two (latitude, longitude) points

    Latitude/longitude degrees are treated directly as Cartesian
    coordinates, without any geodesic correction. This is intentional: it
    is the naive baseline distance measure that the Haversine variant and
    the PHMM Forward alignment are compared against.

    Args:
        point_a (tuple[float, float]): (latitude, longitude) in degrees
        point_b (tuple[float, float]): (latitude, longitude) in degrees

    Returns:
        float: Euclidean distance between the two points, in degrees
    """
    return math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)


def haversine_distance(point_a, point_b):
    """
    Great-circle distance between two (latitude, longitude) points

    Improves on ``euclidean_distance`` by accounting for the Earth's
    curvature, using the same Haversine formula the preselection step and
    the PHMM Forward alignment rely on.

    Args:
        point_a (tuple[float, float]): (latitude, longitude) in degrees
        point_b (tuple[float, float]): (latitude, longitude) in degrees

    Returns:
        float: Great-circle distance between the two points, in kilometers
    """
    return haversine(point_a, point_b, unit=Unit.KILOMETERS)


def to_local_xy(lat, lon, ref_lat, ref_lon):
    """
    Project a (latitude, longitude) point onto a local flat-earth xy plane

    Uses an equirectangular approximation centered on ``(ref_lat, ref_lon)``,
    which is accurate to well under a metre at the ~6km scale the
    preselection candidates operate at. A true geodesic cross-track formula
    would be more accurate at longer range, but isn't warranted here.

    Args:
        lat (float): Latitude of the point to project, in degrees
        lon (float): Longitude of the point to project, in degrees
        ref_lat (float): Latitude of the projection's reference point, in degrees
        ref_lon (float): Longitude of the projection's reference point, in degrees

    Returns:
        tuple[float, float]: (x, y) coordinates in kilometers, relative to the reference point
    """
    R = 6371.0088  # Earth radius, km
    x = np.radians(lon - ref_lon) * np.cos(np.radians(ref_lat)) * R
    y = np.radians(lat - ref_lat) * R
    return x, y


def point_to_segment_dist(px, py, ax, ay, bx, by):
    """
    Shortest distance from point (px, py) to the segment (ax, ay)-(bx, by)

    Args:
        px (float): Query point's x coordinate
        py (float): Query point's y coordinate
        ax (float): Segment start point's x coordinate
        ay (float): Segment start point's y coordinate
        bx (float): Segment end point's x coordinate
        by (float): Segment end point's y coordinate

    Returns:
        tuple[float, float]: (distance, t) where distance is the
        perpendicular (or endpoint) distance from the query point to the
        segment, and t in [0, 1] is the fractional position of the closest
        point along the segment (0 = start point, 1 = end point)
    """
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    seg_len_sq = abx ** 2 + aby ** 2
    if seg_len_sq == 0:
        return np.hypot(apx, apy), 0.0

    t = np.clip((apx * abx + apy * aby) / seg_len_sq, 0.0, 1.0)
    cx, cy = ax + t * abx, ay + t * aby
    return np.hypot(px - cx, py - cy), t


def segment_distance(RF_point, AIS_points):
    """
    Minimum point-to-segment distance from an RF point to an AIS track

    Projects the RF point and every AIS track point onto a local flat-earth
    plane centered on the RF point (making the RF point the origin), then
    takes the minimum perpendicular distance from the origin to each
    consecutive pair of AIS points.

    Args:
        RF_point (tuple[float, float]): (latitude, longitude) of the query point
        AIS_points (list[tuple[float, float]]): (latitude, longitude) points
            making up the AIS track, in order

    Returns:
        float: Minimum distance from the RF point to the AIS track, in kilometers
    """
    ref_lat, ref_lon = RF_point
    xs, ys = zip(*(to_local_xy(lat, lon, ref_lat, ref_lon) for lat, lon in AIS_points))

    if len(AIS_points) == 1:
        return np.hypot(xs[0], ys[0])

    return min(
        point_to_segment_dist(0.0, 0.0, xs[i], ys[i], xs[i + 1], ys[i + 1])[0]
        for i in range(len(xs) - 1)
    )


def combined_score(dist_km, dt_hours, dist_threshold=6, window_hours=3, alpha=1.0):
    """
    Combine a spatial distance and a time gap into a single lower-is-better score

    Both terms are normalized before being summed: ``dist_threshold`` and
    ``window_hours`` match the preselection step's own defaults (see
    ``AIS_RF_preselection.compute_alignments``/``make_AIS_metadata``), so a
    distance or time gap that would already exclude a candidate from
    preselection contributes roughly 1.0 to the score.

    Args:
        dist_km (float): Spatial distance between the RF point and the
            closest point on the AIS track, in kilometers
        dt_hours (float): Absolute time gap between the RF observation and
            the corresponding AIS position, in hours
        dist_threshold (float): Distance (km) that normalizes to 1.0
        window_hours (float): Time gap (hours) that normalizes to 1.0
        alpha (float): Weight applied to the normalized time term relative
            to the normalized distance term

    Returns:
        float: Combined score; lower indicates a better match
    """
    dist_norm = dist_km / dist_threshold
    time_norm = dt_hours / window_hours
    return dist_norm + alpha * time_norm


def time_weighted_distance(RF_point, RF_time, AIS_seq, dist_threshold=6, window_hours=3, alpha=1.0):
    """
    Minimum time-weighted combined score from an RF observation to an AIS track

    Extends ``segment_distance`` with a temporal term: for each AIS track
    leg, the AIS timestamp is interpolated using the same fractional
    position ``t`` used for the spatial projection, rather than snapping to
    whichever endpoint is nearest. That avoids a discontinuity in the time
    gap at a leg's midpoint, where the nearest-vertex timestamp would
    otherwise jump abruptly from one endpoint's to the other's. The spatial
    and temporal offsets are then combined via ``combined_score``, and the
    minimum over all legs is returned.

    Args:
        RF_point (tuple[float, float]): (latitude, longitude) of the RF observation
        RF_time (pd.Timestamp): Timestamp of the RF observation
        AIS_seq (list[tuple[pd.Timestamp, tuple[float, float]]]): AIS track
            as (timestamp, (latitude, longitude)) pairs, in order
        dist_threshold (float): Passed through to ``combined_score``
        window_hours (float): Passed through to ``combined_score``
        alpha (float): Passed through to ``combined_score``

    Returns:
        float: Minimum combined score over the AIS track; lower is better
    """
    ref_lat, ref_lon = RF_point
    AIS_times = [ts for ts, _ in AIS_seq]
    xs, ys = zip(*(to_local_xy(lat, lon, ref_lat, ref_lon) for _, (lat, lon) in AIS_seq))

    if len(AIS_seq) == 1:
        dt_hours = abs((RF_time - AIS_times[0]).total_seconds()) / 3600
        return combined_score(np.hypot(xs[0], ys[0]), dt_hours, dist_threshold, window_hours, alpha)

    scores = []
    for i in range(len(xs) - 1):
        dist, t = point_to_segment_dist(0.0, 0.0, xs[i], ys[i], xs[i + 1], ys[i + 1])
        interp_time = AIS_times[i] + t * (AIS_times[i + 1] - AIS_times[i])
        dt_hours = abs((RF_time - interp_time).total_seconds()) / 3600
        scores.append(combined_score(dist, dt_hours, dist_threshold, window_hours, alpha))

    return min(scores)


def compute_nn_scores_chunk(AIS_track_id, AIS_seq, RF_items, distance_fn):
    """
    Compute nearest-neighbor distances for every RF signal matched to a single AIS track

    Args:
        AIS_track_id: Identifier of the AIS track shared by all items in this chunk
        AIS_seq (list[tuple]): AIS sequence as (timestamp, coordinate) pairs
        RF_items (list[tuple]): (RF_signal_id, RF_track_id, RF_seq) tuples, one per
            RF candidate matched to this AIS track
        distance_fn (Callable[[tuple, tuple], float]): Distance measure
            between two (latitude, longitude) points, e.g.
            ``euclidean_distance`` or ``haversine_distance``

    Returns:
        tuple: (elapsed seconds, list[dict] of per-RF-signal results)
    """
    start = time.time()
    results = []
    AIS_points = [coord for _, coord in AIS_seq]

    for RF_signal_id, RF_track_id, RF_seq in RF_items:
        # RF_seq holds a single (timestamp, coordinate) observation
        RF_point = RF_seq[0][1]

        # Distance to the closest AIS point in this track
        nn_distance = min(
            distance_fn(RF_point, AIS_point) for AIS_point in AIS_points
        )

        results.append({
            "RF_signal_id": RF_signal_id,
            "RF_track_id": RF_track_id,
            "AIS_track_id": AIS_track_id,
            "nn_distance": nn_distance,
            "is_true_match": RF_track_id == AIS_track_id,
        })

    return time.time() - start, results


def compute_segment_scores_chunk(AIS_track_id, AIS_seq, RF_items):
    """
    Compute point-to-segment distances for every RF signal matched to a single AIS track

    Unlike ``compute_nn_scores_chunk``'s point-to-point distance, this
    measures each RF point against the line segments joining consecutive
    AIS positions, so an RF signal that falls between two AIS fixes isn't
    penalized for not landing exactly on one of them.

    Args:
        AIS_track_id: Identifier of the AIS track shared by all items in this chunk
        AIS_seq (list[tuple]): AIS sequence as (timestamp, coordinate) pairs
        RF_items (list[tuple]): (RF_signal_id, RF_track_id, RF_seq) tuples, one per
            RF candidate matched to this AIS track

    Returns:
        tuple: (elapsed seconds, list[dict] of per-RF-signal results)
    """
    start = time.time()
    results = []
    AIS_points = [coord for _, coord in AIS_seq]

    for RF_signal_id, RF_track_id, RF_seq in RF_items:
        RF_point = RF_seq[0][1]
        nn_distance = segment_distance(RF_point, AIS_points)

        results.append({
            "RF_signal_id": RF_signal_id,
            "RF_track_id": RF_track_id,
            "AIS_track_id": AIS_track_id,
            "nn_distance": nn_distance,
            "is_true_match": RF_track_id == AIS_track_id,
        })

    return time.time() - start, results


def compute_time_weighted_scores_chunk(AIS_track_id, AIS_seq, RF_items, dist_threshold=6, window_hours=3, alpha=1.0):
    """
    Compute time-weighted combined scores for every RF signal matched to a single AIS track

    Unlike ``compute_segment_scores_chunk``'s purely spatial distance, this
    also penalizes AIS legs that are temporally distant from the RF
    observation. See ``time_weighted_distance`` for the per-track computation.

    Args:
        AIS_track_id: Identifier of the AIS track shared by all items in this chunk
        AIS_seq (list[tuple]): AIS sequence as (timestamp, coordinate) pairs
        RF_items (list[tuple]): (RF_signal_id, RF_track_id, RF_seq) tuples, one per
            RF candidate matched to this AIS track
        dist_threshold (float): Passed through to ``combined_score``
        window_hours (float): Passed through to ``combined_score``
        alpha (float): Passed through to ``combined_score``

    Returns:
        tuple: (elapsed seconds, list[dict] of per-RF-signal results)
    """
    start = time.time()
    results = []

    for RF_signal_id, RF_track_id, RF_seq in RF_items:
        RF_time, RF_point = RF_seq[0]
        nn_distance = time_weighted_distance(
            RF_point, RF_time, AIS_seq, dist_threshold, window_hours, alpha
        )

        results.append({
            "RF_signal_id": RF_signal_id,
            "RF_track_id": RF_track_id,
            "AIS_track_id": AIS_track_id,
            "nn_distance": nn_distance,
            "is_true_match": RF_track_id == AIS_track_id,
        })

    return time.time() - start, results


def _compute_score_parallel(df_train, df_preselection, chunk_fn, checkpoint_path=None):
    """
    Shared chunking/checkpointing/parallel-execution driver for the NN baselines

    Mirrors the chunking/checkpointing structure of
    ``compute_forward_score_parallel`` so its output lines up with the
    Forward/PHMM alignment scores on the same candidate pairs: candidate
    pairs are grouped by AIS track so each worker call reuses one AIS
    sequence across all of its RF candidates, and results are checkpointed
    per AIS track as they complete. Shared by every baseline (point-to-point
    and point-to-segment) so this logic is written once.

    Args:
        df_train (pd.DataFrame): Dataset containing AIS and RF sequences
        df_preselection (pd.DataFrame): Candidate AIS-RF pairs
        chunk_fn (Callable[[Any, list, list], tuple]): Picklable callable
            taking (AIS_track_id, AIS_seq, RF_items) and returning
            (elapsed seconds, list[dict] of per-RF-signal results), e.g.
            ``compute_nn_scores_chunk`` (bound to a distance_fn via
            ``functools.partial``) or ``compute_segment_scores_chunk``
        checkpoint_path (str, optional): Path used to persist each AIS
            track's results as soon as it completes. If given, an
            interrupted run can be restarted and will only recompute AIS
            tracks not yet in the checkpoint.

    Returns:
        tuple: (pd.DataFrame of distances per AIS-RF pair, average seconds per AIS track chunk)
    """
    # Extract all RF observations (rows where RF coords are present)
    df_RF = df_train[df_train["RF"].notna()].copy()

    # Give each RF signal a sequential identifier within its track
    df_RF["RF_signal_id"] = df_RF.groupby("ID").cumcount() + 1

    # Build fast lookup dictionaries for AIS and RF data
    ais_dict = {track_id: group for track_id, group in df_train.groupby("ID")}
    rf_dict = {(row["ID"], row["RF_signal_id"]): row for _, row in df_RF.iterrows()}

    # Build one chunk per AIS track so its sequence is only pickled once,
    # instead of once per RF-AIS candidate pair
    chunks = {}
    for row in tqdm(df_preselection.itertuples(index=False), total=len(df_preselection), desc="Building task list"):
        try:
            rf_row = rf_dict.get((row.RF_track_id, row.RF_signal_id))
            ais_data = ais_dict.get(row.AIS_track_id)

            if rf_row is None or ais_data is None:
                continue

            RF_seq = [(rf_row["RF_Timestamp"], rf_row["RF"])]
            item = (row.RF_signal_id, row.RF_track_id, RF_seq)

            if row.AIS_track_id not in chunks:
                AIS_seq = [
                    (dt, coord) for dt, coord in zip(ais_data["AIS_Timestamp"], ais_data["AIS"])
                    if pd.notna(dt) and coord is not None
                ]
                chunks[row.AIS_track_id] = (AIS_seq, [])
            chunks[row.AIS_track_id][1].append(item)

        except Exception as e:
            print(f"Error building task: {e}")

    done_results, times = (
        _load_checkpoint(checkpoint_path) if checkpoint_path else ({}, [])
    )

    pending_chunks = {
        AIS_track_id: payload
        for AIS_track_id, payload in chunks.items()
        if AIS_track_id not in done_results
    }
    if len(pending_chunks) < len(chunks):
        print(
            f"Resuming from checkpoint: skipping "
            f"{len(chunks) - len(pending_chunks)} already-processed AIS tracks"
        )

    max_workers = determine_max_workers()
    print(f"Using {max_workers} worker processes (of {os.cpu_count()} CPUs)")

    spawn_context = multiprocessing.get_context("spawn")

    checkpoint_file = open(checkpoint_path, "ab") if checkpoint_path else None
    try:
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=spawn_context) as executor:
            futures = {
                executor.submit(chunk_fn, AIS_track_id, AIS_seq, RF_items): AIS_track_id
                for AIS_track_id, (AIS_seq, RF_items) in pending_chunks.items()
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Calculating NN distance per AIS track", unit="track"):
                AIS_track_id = futures[future]
                try:
                    time_iter, result = future.result()
                except Exception as e:
                    print(f"Error during processing {e}")
                    continue

                done_results[AIS_track_id] = result
                times.append(time_iter)
                if checkpoint_file is not None:
                    pickle.dump((AIS_track_id, result, time_iter), checkpoint_file)
                    checkpoint_file.flush()
    finally:
        if checkpoint_file is not None:
            checkpoint_file.close()

    # Flatten the nested lists and return a dataframe
    flatten_results = [item for result in done_results.values() for item in result]

    avg_time_per_iter = np.array(times).mean()
    return pd.DataFrame(flatten_results), avg_time_per_iter


def compute_nn_score_parallel(df_train, df_preselection, distance_fn=euclidean_distance, checkpoint_path=None):
    """
    Match RF signals to AIS tracks using point-to-point nearest-neighbor distance, in parallel

    Args:
        df_train (pd.DataFrame): Dataset containing AIS and RF sequences
        df_preselection (pd.DataFrame): Candidate AIS-RF pairs
        distance_fn (Callable[[tuple, tuple], float]): Distance measure
            between two (latitude, longitude) points, e.g.
            ``euclidean_distance`` or ``haversine_distance``
        checkpoint_path (str, optional): Path used to persist each AIS
            track's results as soon as it completes. If given, an
            interrupted run can be restarted and will only recompute AIS
            tracks not yet in the checkpoint.

    Returns:
        tuple: (pd.DataFrame of nearest-neighbor distances per AIS-RF pair, average seconds per AIS track chunk)
    """
    return _compute_score_parallel(
        df_train, df_preselection,
        partial(compute_nn_scores_chunk, distance_fn=distance_fn),
        checkpoint_path=checkpoint_path,
    )


def compute_segment_score_parallel(df_train, df_preselection, checkpoint_path=None):
    """
    Match RF signals to AIS tracks using point-to-segment distance, in parallel

    For each RF point, projects it and the AIS track onto a local
    flat-earth plane centered on the RF point and takes the minimum
    perpendicular distance to any consecutive pair of AIS points. See
    ``segment_distance`` for the per-pair computation.

    Args:
        df_train (pd.DataFrame): Dataset containing AIS and RF sequences
        df_preselection (pd.DataFrame): Candidate AIS-RF pairs
        checkpoint_path (str, optional): Path used to persist each AIS
            track's results as soon as it completes. If given, an
            interrupted run can be restarted and will only recompute AIS
            tracks not yet in the checkpoint.

    Returns:
        tuple: (pd.DataFrame of point-to-segment distances per AIS-RF pair, average seconds per AIS track chunk)
    """
    return _compute_score_parallel(
        df_train, df_preselection, compute_segment_scores_chunk, checkpoint_path=checkpoint_path
    )


def compute_time_weighted_score_parallel(
    df_train, df_preselection, dist_threshold=6, window_hours=3, alpha=1.0, checkpoint_path=None
):
    """
    Match RF signals to AIS tracks using a time-weighted distance score, in parallel

    Combines the point-to-segment spatial distance with an interpolated
    time gap into a single lower-is-better score via ``combined_score``. See
    ``time_weighted_distance`` for the per-track computation.

    Args:
        df_train (pd.DataFrame): Dataset containing AIS and RF sequences
        df_preselection (pd.DataFrame): Candidate AIS-RF pairs
        dist_threshold (float): Passed through to ``combined_score``
        window_hours (float): Passed through to ``combined_score``
        alpha (float): Passed through to ``combined_score``
        checkpoint_path (str, optional): Path used to persist each AIS
            track's results as soon as it completes. If given, an
            interrupted run can be restarted and will only recompute AIS
            tracks not yet in the checkpoint.

    Returns:
        tuple: (pd.DataFrame of time-weighted scores per AIS-RF pair, average seconds per AIS track chunk)
    """
    return _compute_score_parallel(
        df_train, df_preselection,
        partial(
            compute_time_weighted_scores_chunk,
            dist_threshold=dist_threshold, window_hours=window_hours, alpha=alpha,
        ),
        checkpoint_path=checkpoint_path,
    )


# === MAIN SCRIPT ===
# Baseline models compared as an ablation: plain Euclidean, Haversine (both
# point-to-point nearest-neighbor), point-to-segment, and time-weighted
# (point-to-segment distance combined with an interpolated time gap)
MODELS = {
    "euclidean": partial(compute_nn_score_parallel, distance_fn=euclidean_distance),
    "haversine": partial(compute_nn_score_parallel, distance_fn=haversine_distance),
    "segment": compute_segment_score_parallel,
    "time_weighted": compute_time_weighted_score_parallel,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate nearest-neighbor baseline scores for AIS-RF pairs")
    parser.add_argument(
        "--error-model", choices=["uniform", "gaussian"], default="uniform",
        help="RF bearing-error model whose data folder to read from and write to (default: uniform)",
    )
    args = parser.parse_args()

    # "uniform" keeps the original flat layout for backward compatibility;
    # other error models live in their own subfolder under data/processed
    DATA_DIR = (
        "./data/processed" if args.error_model == "uniform"
        else f"./data/processed/{args.error_model}"
    )
    os.makedirs(DATA_DIR, exist_ok=True)

    SOURCE_PATH1 = f"{DATA_DIR}/train_data_sample_5000.pkl"
    SOURCE_PATH2 = f"{DATA_DIR}/AIS_RF_preselection_data.pkl"

    df_train = pd.read_pickle(SOURCE_PATH1)
    df_preselection = pd.read_pickle(SOURCE_PATH2)

    for model_name, compute_fn in MODELS.items():
        print(f"Calculating nearest-neighbor ({model_name}) baseline score for AIS-RF pair...")

        start_time = datetime.now()

        DESTINATION_PATH = f"{DATA_DIR}/AIS_RF_nn_baseline_{model_name}_scores_data.pkl"
        CHECKPOINT_PATH = f"{DATA_DIR}/AIS_RF_nn_baseline_{model_name}_scores_checkpoint.pkl"

        score_df, avg_time_per_iter = compute_fn(
            df_train, df_preselection, checkpoint_path=CHECKPOINT_PATH
        )
        score_df.to_pickle(DESTINATION_PATH)

        processing_time = datetime.now() - start_time

        print("Data saved to", DESTINATION_PATH)
        print(f"Processing took {processing_time} (hh:mm:ss.ms)")
        print(f"Average processing time per AIS track chunk took {avg_time_per_iter} seconds")
