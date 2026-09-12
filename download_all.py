"""Background fetch of every subject's executed-movement runs.

MNE caches downloads, so this is paid once ever. Running it ahead of time means
the full-scale re-run in Phase 6 costs nothing extra.
"""

import sys
import time

import mne

RUNS = [3, 7, 11]  # executed left vs right fist
SUBJECTS = range(1, 110)


def main():
    mne.set_log_level("ERROR")
    t0 = time.time()
    failed = []

    for subj in SUBJECTS:
        try:
            mne.datasets.eegbci.load_data(subj, RUNS, update_path=True, verbose="ERROR")
            print(f"[{subj:3d}/109] ok  ({time.time() - t0:.0f}s elapsed)", flush=True)
        except Exception as exc:  # noqa: BLE001 - we want to keep going and report at the end
            failed.append((subj, repr(exc)))
            print(f"[{subj:3d}/109] FAILED: {exc!r}", flush=True)

    print(f"\ndone in {time.time() - t0:.0f}s; {len(failed)} failures", flush=True)
    for subj, exc in failed:
        print(f"  S{subj:03d}: {exc}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
