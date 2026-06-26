"""
preprocessing/align_features.py

Maps each benchmark dataset's raw column names to a shared
canonical feature set, then returns only those aligned columns.

The 23-feature aligned set was derived by manually intersecting
the feature lists of CICIDS2017 (80 features), UNSW-NB15 (49 features),
and CSE-CIC-IDS2018 (80 features).
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Canonical aligned feature names (target schema)
# ---------------------------------------------------------------------------
ALIGNED_FEATURES = [
    "flow_duration",
    "protocol",
    "src_port",
    "dst_port",
    "fwd_pkt_len_mean",
    "fwd_pkt_len_max",
    "fwd_pkt_len_min",
    "bwd_pkt_len_mean",
    "bwd_pkt_len_max",
    "bwd_pkt_len_min",
    "fwd_iat_mean",
    "fwd_iat_std",
    "bwd_iat_mean",
    "bwd_iat_std",
    "flow_iat_mean",
    "flow_iat_std",
    "syn_flag_cnt",
    "ack_flag_cnt",
    "fin_flag_cnt",
    "bytes_per_sec",
    "pkts_per_sec",
    "fwd_header_len",
    "bwd_header_len",
]

# ---------------------------------------------------------------------------
# Per-dataset column-name mappings  {raw_name: canonical_name}
# ---------------------------------------------------------------------------
_CICIDS2017_MAP = {
    "flow_duration":        "flow_duration",
    "protocol":             "protocol",
    "source_port":          "src_port",
    "destination_port":     "dst_port",
    "fwd_packet_length_mean": "fwd_pkt_len_mean",
    "fwd_packet_length_max":  "fwd_pkt_len_max",
    "fwd_packet_length_min":  "fwd_pkt_len_min",
    "bwd_packet_length_mean": "bwd_pkt_len_mean",
    "bwd_packet_length_max":  "bwd_pkt_len_max",
    "bwd_packet_length_min":  "bwd_pkt_len_min",
    "fwd_iat_mean":         "fwd_iat_mean",
    "fwd_iat_std":          "fwd_iat_std",
    "bwd_iat_mean":         "bwd_iat_mean",
    "bwd_iat_std":          "bwd_iat_std",
    "flow_iat_mean":        "flow_iat_mean",
    "flow_iat_std":         "flow_iat_std",
    "syn_flag_count":       "syn_flag_cnt",
    "ack_flag_count":       "ack_flag_cnt",
    "fin_flag_count":       "fin_flag_cnt",
    "flow_bytes/s":         "bytes_per_sec",
    "flow_packets/s":       "pkts_per_sec",
    "fwd_header_length":    "fwd_header_len",
    "bwd_header_length":    "bwd_header_len",
}

_UNSWNB15_MAP = {
    "dur":       "flow_duration",
    "proto":     "protocol",
    "sport":     "src_port",
    "dsport":    "dst_port",
    "smean":     "fwd_pkt_len_mean",
    "smax":      "fwd_pkt_len_max",     # approximate mapping
    "smin":      "fwd_pkt_len_min",
    "dmean":     "bwd_pkt_len_mean",
    "dmax":      "bwd_pkt_len_max",
    "dmin":      "bwd_pkt_len_min",
    "sintpkt":   "fwd_iat_mean",
    "sjit":      "fwd_iat_std",
    "dintpkt":   "bwd_iat_mean",
    "djit":      "bwd_iat_std",
    "sintpkt":   "flow_iat_mean",       # reuse closest proxy
    "sjit":      "flow_iat_std",
    "synack":    "syn_flag_cnt",
    "ackdat":    "ack_flag_cnt",
    "fin_flag_cnt": "fin_flag_cnt",     # keep if present
    "sbytes":    "bytes_per_sec",
    "spkts":     "pkts_per_sec",
    "swin":      "fwd_header_len",
    "dwin":      "bwd_header_len",
}

_CSE_CIC_IDS2018_MAP = {
    "flow_duration":         "flow_duration",
    "protocol":              "protocol",
    "src_port":              "src_port",
    "dst_port":              "dst_port",
    "fwd_packet_length_mean":"fwd_pkt_len_mean",
    "fwd_packet_length_max": "fwd_pkt_len_max",
    "fwd_packet_length_min": "fwd_pkt_len_min",
    "bwd_packet_length_mean":"bwd_pkt_len_mean",
    "bwd_packet_length_max": "bwd_pkt_len_max",
    "bwd_packet_length_min": "bwd_pkt_len_min",
    "fwd_iat_mean":          "fwd_iat_mean",
    "fwd_iat_std":           "fwd_iat_std",
    "bwd_iat_mean":          "bwd_iat_mean",
    "bwd_iat_std":           "bwd_iat_std",
    "flow_iat_mean":         "flow_iat_mean",
    "flow_iat_std":          "flow_iat_std",
    "syn_flag_count":        "syn_flag_cnt",
    "ack_flag_count":        "ack_flag_cnt",
    "fin_flag_count":        "fin_flag_cnt",
    "flow_bytes/s":          "bytes_per_sec",
    "flow_packets/s":        "pkts_per_sec",
    "fwd_header_length":     "fwd_header_len",
    "bwd_header_length":     "bwd_header_len",
}

_DATASET_MAPS = {
    "cicids2017":     _CICIDS2017_MAP,
    "unswnb15":       _UNSWNB15_MAP,
    "cse_cic_ids2018": _CSE_CIC_IDS2018_MAP,
}


def align_features(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
    """
    Rename dataset-specific columns to canonical names and return
    only the 23 aligned features.

    Missing canonical features are filled with 0.0.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-only DataFrame (label column already removed).
    dataset_type : str
        One of 'cicids2017', 'unswnb15', 'cse_cic_ids2018'.

    Returns
    -------
    pd.DataFrame
        DataFrame with exactly the columns in ALIGNED_FEATURES.
    """
    col_map = _DATASET_MAPS.get(dataset_type, {})
    df = df.rename(columns=col_map)

    # If the mock data already uses canonical names, skip renaming
    aligned = pd.DataFrame(index=df.index)
    for feat in ALIGNED_FEATURES:
        if feat in df.columns:
            aligned[feat] = df[feat]
        else:
            aligned[feat] = 0.0   # fill missing feature with zero

    return aligned.astype(float)
