# Atomic I/O R2

The traceback ends at `controlled_a0a5_r0.main` ledger `.tmp.replace(.csv)` inside the authorized DesignA output directory. Python emits `os.rename` for `os.replace`; the frozen guard blocked the operation indiscriminately.

`atomic_training_access.py` extends that guard without changing the R2-bound original. Only explicitly registered absolute source/target pairs inside the output namespace are accepted. Traversal, reparse points, dir_fd, unregistered temporaries, frozen checkpoint overwrites and deletes remain denied. Read policy is inherited unchanged. Checkpoints use a registered pending path, then atomic commit and disk-based strict reload. An immutable task record is persisted before ledger updates. Metrics are reconstructed from committed epoch records rather than dropping prior tasks.

18 regression checks passed, including an actual existing Windows directory junction. The actual training entry completed two non-scientific epochs on frozen train/validation inputs in an isolated namespace, then saved/reloaded a checkpoint and atomically committed CSV/JSON (exit 0; no pending report files). This tests process-interruption boundaries, not power-loss durability. Python access control is not OS isolation.

Recovery independently verified the three committed checkpoints, their hashes, 50 epoch histories, optimizer counts, source B0/event identities and ledger membership. The 720 identities are unchanged. The orphan A3 checkpoint (SHA256 `265a37688949f7c6a9b788aea60d0f29678765c72df503bf66578391d00de342`) claims 50 epochs and contains 1250 Adam steps, but its 50 per-epoch loss/validation records were never persisted. It remains in place, uncounted and untouched. Existing quarantined checkpoints remain untouched. No retraining or adoption is authorized without complete recovery evidence under the current instruction.

ATOMIC_IO_FIX=PASS
PROHIBITED_PATH_TESTS=PASS
ATOMIC_LEDGER_TEST=PASS
CHECKPOINT_RECOVERY=FAIL
TRAINING_RESUME=STOPPED
FORMAL_COMPLETED_COUNT=3
CURRENT_PID=NONE
LATEST_COMPLETED_TASK=GI/G25/x/seed1701/template0/A2

Remaining decision: supply the missing A3 trajectory evidence, or explicitly authorize preserving that orphan and rerunning only A3 under the unchanged protocol. No formal training was started because recovery did not pass.
