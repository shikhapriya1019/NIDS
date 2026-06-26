"""
generate_mock_data.py
Generates realistic mock CSV datasets for CICIDS2017, UNSW-NB15,
and CSE-CIC-IDS2018 so the pipeline can be tested without the
multi-gigabyte originals.
"""

import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

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


def _make_samples(n_benign: int, n_attack: int, label_col: str,
                  benign_label, attack_label,
                  noise_scale: float = 1.0) -> pd.DataFrame:
    """Create feature rows for benign and attack traffic."""
    benign = pd.DataFrame(
        rng.normal(loc=0.3, scale=0.1 * noise_scale, size=(n_benign, len(ALIGNED_FEATURES))),
        columns=ALIGNED_FEATURES,
    ).clip(0, 1)
    benign[label_col] = benign_label

    attack = pd.DataFrame(
        rng.normal(loc=0.7, scale=0.15 * noise_scale, size=(n_attack, len(ALIGNED_FEATURES))),
        columns=ALIGNED_FEATURES,
    ).clip(0, 1)
    attack[label_col] = attack_label

    return pd.concat([benign, attack], ignore_index=True).sample(
        frac=1, random_state=RANDOM_STATE
    )


def generate_all(base_dir: str = "data") -> None:
    """Write all four CSV files into *base_dir*/raw/."""
    raw_dir = os.path.join(base_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # CICIDS2017 train (80 % split)
    df_train = _make_samples(8000, 2000, "Label", "BENIGN", "DoS Hulk")
    df_train.to_csv(os.path.join(raw_dir, "cicids2017_train.csv"), index=False)

    # CICIDS2017 test (20 % split)
    df_test = _make_samples(2000, 500, "Label", "BENIGN", "PortScan")
    df_test.to_csv(os.path.join(raw_dir, "cicids2017_test.csv"), index=False)

    # UNSW-NB15 — numeric labels, different noise
    df_unsw = _make_samples(3000, 1000, "label", 0, 1, noise_scale=1.3)
    df_unsw.to_csv(os.path.join(raw_dir, "unswnb15_test.csv"), index=False)

    # CSE-CIC-IDS2018 — string labels
    df_cse = _make_samples(3000, 1500, "Label", "Benign", "DDoS attacks-LOIC-HTTP",
                           noise_scale=1.1)
    df_cse.to_csv(os.path.join(raw_dir, "cse_cic_ids2018_test.csv"), index=False)

    print(f"[generate_mock_data] All CSV files written to '{raw_dir}'.")


if __name__ == "__main__":
    generate_all()
