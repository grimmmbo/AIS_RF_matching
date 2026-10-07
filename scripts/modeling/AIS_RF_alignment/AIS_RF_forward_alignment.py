
import argparse
import multiprocessing
import os
import pickle
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd

from scripts.modeling.hidden_states.states import *
from scripts.modeling.AIS_RF_alignment.forward import *

# kappa=(kappa_ais, kappa_rf) -- the (AIS exponent, RF/M exponent) pair
# hardcoded as (0.5, 2) before --kappa existed. This default is the only
# kappa whose cache files keep the original, unsuffixed names, so an
# existing run is reused (and reproduced) rather than recomputed.
DEFAULT_KAPPA = (0.5, 2.0)


def parse_kappa(kappa_str):
    """Parse a "--kappa a,b" CLI value into a (kappa_ais, kappa_rf) float tuple"""
    kappa_ais_str, kappa_rf_str = kappa_str.split(",")
    return float(kappa_ais_str), float(kappa_rf_str)


def kappa_cache_suffix(kappa_ais, kappa_rf):
    """
    Filename suffix for a given kappa, so each kappa's forward-score
    cache (and checkpoint) lives in its own file instead of colliding
    with another kappa's -- except the default kappa, which keeps the
    original unsuffixed filenames so it reuses (and exactly reproduces)
    whatever has already been computed for it.
    """
    if (kappa_ais, kappa_rf) == DEFAULT_KAPPA:
        return ""

    def fmt(value):
        text = f"{value:g}"
        return text.replace(".", "_").replace("-", "neg")

    return f"_kappa_{fmt(kappa_ais)}_{fmt(kappa_rf)}"

### RUNNING TIME ###
    # Total processing time 0:21:36.276858 (hh:mm:ss.ms)
    # Average processing time per forward alignment took 0.547953668199831 seconds
    # Each RF-AIS pair now also runs a second Forward pass against the
    # null/background model (NullMState, see scripts/modeling/hidden_states/states.py)
    # to compute the log-odds score, roughly doubling both of the above.

def compute_forward_scores_chunk(AIS_track_id, AIS_seq, RF_items, kappa_ais=0.5, kappa_rf=2.0):
    """
    Compute forward alignment scores for every RF signal matched to a single AIS track

    Grouping by AIS track lets one worker call reuse the same AIS_seq for all of
    its RF candidates instead of pickling it once per RF-AIS pair, which cuts the
    data sent across the process boundary roughly by the average candidate count
    per track (~8x on the current preselection set).

    Args:
        AIS_track_id: Identifier of the AIS track shared by all items in this chunk
        AIS_seq (list[tuple]): AIS sequence as (timestamp, coordinate) pairs
        RF_items (list[tuple]): (RF_signal_id, RF_track_id, RF_seq) tuples, one per
            RF candidate matched to this AIS track
        kappa_ais (float): Exponent applied to AISState's row-scaled transition
            probability (default 0.5, the value hardcoded before --kappa existed)
        kappa_rf (float): Exponent applied to both RFState's and MState's
            row-scaled transition probability (default 2, ditto)

    Returns:
        tuple: (elapsed seconds, list[dict] of per-RF-signal results)
    """
    start = time.time()
    results = []

    for RF_signal_id, RF_track_id, RF_seq in RF_items:
        states = [
            BeginState(), AISState(exponent=kappa_ais), RFState(exponent=kappa_rf),
            MState(exponent=kappa_rf), EndState(),
        ]
        forward_model = PHMM_forward(states, EndState())
        forward_score = forward_model.forward(AIS_seq, RF_seq)

        # Normalize by AIS sequence length so longer tracks aren't favored
        adjusted_score = forward_score / len(AIS_seq)

        # Null/background model: identical states except M transitions
        # are forced to zero probability (see NullMState), so the
        # resulting score_lo = forward_score - null_forward_score
        # cancels sequence-composition effects shared by both models
        null_states = [
            BeginState(), AISState(exponent=kappa_ais), RFState(exponent=kappa_rf),
            NullMState(exponent=kappa_rf), EndState(),
        ]
        null_forward_model = PHMM_forward(null_states, EndState())
        null_forward_score = null_forward_model.forward(AIS_seq, RF_seq)
        log_odds_score = forward_score - null_forward_score

        results.append({
            "RF_signal_id": RF_signal_id,
            "RF_track_id": RF_track_id,
            "AIS_track_id": AIS_track_id,
            "forward_score": forward_score,
            "normalized_forward_score": adjusted_score,
            "null_forward_score": null_forward_score,
            "log_odds_score": log_odds_score,
            "is_true_match": RF_track_id == AIS_track_id,
        })

    return time.time() - start, results


def determine_max_workers(mem_per_worker_gb=1.5, reserve_gb=4.0):
    """
    Pick a worker count that fits both the host's CPU count and its available memory

    Args:
        mem_per_worker_gb (float): Assumed memory budget per worker process
        reserve_gb (float): Memory to leave free for the OS and the main
            process's own AIS/RF lookup tables

    Returns:
        int: Number of worker processes to use (at least 1)
    """
    cpu_workers = os.cpu_count() or 1

    try:
        with open("/proc/meminfo") as meminfo:
            fields = dict(
                line.replace("kB", "").split(":", 1) for line in meminfo if ":" in line
            )
        mem_available_gb = int(fields["MemAvailable"]) / (1024 ** 2)
    except (OSError, KeyError, ValueError):
        # Not on Linux, or /proc/meminfo unavailable: fall back to CPU count only
        return cpu_workers

    mem_workers = max(1, int((mem_available_gb - reserve_gb) / mem_per_worker_gb))
    return max(1, min(cpu_workers, mem_workers))


def _load_checkpoint(checkpoint_path):
    """
    Load previously computed forward-score results from a checkpoint file

    The checkpoint file holds a sequence of pickled
    ``(AIS_track_id, results, time_iter)`` tuples, one per completed AIS
    track, appended as they finish rather than written all at once. This
    lets a run be resumed after a crash or interruption: AIS tracks whose
    id is already in the returned dict are skipped.

    Args:
        checkpoint_path (str): Path to the checkpoint file. Does not need
            to exist yet.

    Returns:
        tuple: (dict mapping AIS_track_id to its results list, list of
        per-chunk processing times already recorded)
    """
    done_results = {}
    times = []
    if not os.path.exists(checkpoint_path):
        return done_results, times

    with open(checkpoint_path, "rb") as file:
        while True:
            try:
                key, result, time_iter = pickle.load(file)
            except EOFError:
                break
            done_results[key] = result
            times.append(time_iter)
    return done_results, times


def compute_forward_score_parallel(
    df_train, df_preselection, checkpoint_path=None, kappa_ais=0.5, kappa_rf=2.0,
):
    """
    Match RF signals to AIS tracks using the Forward algorithm in parallel

    Candidate pairs are grouped by AIS track so each worker call reuses one
    AIS sequence across all of its RF candidates, and results are
    checkpointed per AIS track as they complete so an interrupted run can
    resume instead of losing all progress.

    Args:
        df_train (pd.DataFrame): Dataset containing AIS and RF sequences
        df_preselection (pd.DataFrame): Candidate AIS-RF pairs
        checkpoint_path (str, optional): Path used to persist each AIS
            track's results as soon as it completes. If given, an
            interrupted run can be restarted and will only recompute AIS
            tracks not yet in the checkpoint.
        kappa_ais, kappa_rf (float): See compute_forward_scores_chunk()

    Returns:
        pd.DataFrame: Forward scores per AIS-RF pair
    """
    df_RF = df_train[df_train["RF"].notna()].copy()

    # Each RF signal is aligned individually, not as a full RF sequence,
    # so give each one a sequential id within its track
    df_RF["RF_signal_id"] = df_RF.groupby("ID").cumcount() + 1

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

    # Use 'spawn' instead of the platform default 'fork': fork would make
    # every worker inherit a copy-on-write snapshot of this process's full
    # memory (including ais_dict/rf_dict/df_train), which previously grew
    # each worker to several GB and exhausted host memory. Workers only
    # need the picklable chunk arguments, so spawn keeps their footprint small.
    spawn_context = multiprocessing.get_context("spawn")

    checkpoint_file = open(checkpoint_path, "ab") if checkpoint_path else None
    try:
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=spawn_context) as executor:
            futures = {
                executor.submit(
                    compute_forward_scores_chunk, AIS_track_id, AIS_seq, RF_items, kappa_ais, kappa_rf
                ): AIS_track_id
                for AIS_track_id, (AIS_seq, RF_items) in pending_chunks.items()
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Calculating Forward score per AIS track", unit="track"):
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

    flatten_results = [item for result in done_results.values() for item in result]

    avg_time_per_iter = np.array(times).mean()
    return pd.DataFrame(flatten_results), avg_time_per_iter


# === MAIN SCRIPT ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Forward score for AIS-RF pairs")
    parser.add_argument(
        "--error-model", choices=["uniform", "gaussian"], default="gaussian",
        help="RF bearing-error model whose data folder to read from and write to (default: gaussian)",
    )
    parser.add_argument(
        "--kappa", default="0.5,2",
        help="kappa_ais,kappa_rf exponents for AISState/RFState+MState's row-scaled "
             "transition probabilities (default: 0.5,2, i.e. unchanged behavior). "
             "A non-default kappa writes to its own suffixed cache/checkpoint files "
             "instead of the default ones.",
    )
    args = parser.parse_args()
    kappa_ais, kappa_rf = parse_kappa(args.kappa)

    print(f"Calculating Forward score for AIS-RF pair (kappa={kappa_ais},{kappa_rf})...")

    start_time = datetime.now()

    # Every error model lives in its own subfolder under data/processed
    DATA_DIR = f"./data/processed/{args.error_model}"
    os.makedirs(DATA_DIR, exist_ok=True)

    KAPPA_SUFFIX = kappa_cache_suffix(kappa_ais, kappa_rf)

    SOURCE_PATH1 = f"{DATA_DIR}/train_data_sample_5000.pkl"
    SOURCE_PATH2 = f"{DATA_DIR}/AIS_RF_preselection_data.pkl"
    DESTINATION_PATH = f"{DATA_DIR}/AIS_RF_forward_scores_data{KAPPA_SUFFIX}.pkl"
    CHECKPOINT_PATH = f"{DATA_DIR}/AIS_RF_forward_scores_checkpoint{KAPPA_SUFFIX}.pkl"

    df_train = pd.read_pickle(SOURCE_PATH1)
    df_preselection = pd.read_pickle(SOURCE_PATH2)

    forward_score_df, avg_time_per_iter = compute_forward_score_parallel(
        df_train, df_preselection, checkpoint_path=CHECKPOINT_PATH,
        kappa_ais=kappa_ais, kappa_rf=kappa_rf,
    )
    forward_score_df.to_pickle(DESTINATION_PATH)

    processing_time = datetime.now() - start_time

    print("Data saved to", DESTINATION_PATH)
    print(f"Processing took {processing_time} (hh:mm:ss.ms)")
    print(f"Average processing time per AIS track chunk took {avg_time_per_iter} seconds")
