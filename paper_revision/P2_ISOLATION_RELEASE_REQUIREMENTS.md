# REVISION_REPLICATION execution boundary

This is an unexecuted provisioning contract. P2 found no tested restricted
scientific identity. Do not run scientific jobs using the current admin token.

The current process is medium integrity with Administrators marked deny-only;
`net session` returns access denied. Docker, VMware and VirtualBox command-line
tools were not found. P1 found no installed WSL distribution. Existing
CodexSandbox accounts are not assumed to be dedicated scientific identities.
Do not modify those application-managed accounts.

Prefer a dedicated VM with no host source/truth shares. Copy only frozen
train/validation artifacts, code, environment and manifests into its training
input volume. A separate reconstruction role receives the same trained model
hashes and registered sparse observations, never hidden labels. The evaluation
role receives completed predictions and truth, but cannot modify training
configuration, source code or checkpoint volumes. Disable shared host home,
drive mapping, network shares and inherited handles that expose forbidden data.
The authoritative host data need no ACL changes under this design.

Alternatively an elevated administrator may provision dedicated Windows
identities or an allowlisted AppContainer. Merely using a standard account is
insufficient: source data currently grant Users read access and Authenticated
Users modify access, and two source copies remain accessible in the recycle
bin. Isolate the entire host filesystem from the worker or inventory and deny
every prohibited source/truth copy. Do not grant a worker ownership of hidden
label directories or a writable parent from which ACLs can be replaced.

Before any permission mutation, save original SDDL and an `icacls /save` backup
for every affected staging tree; test restoration on a disposable fixture.
Prefer staging-tree changes, preserving authoritative source ACLs. P2 recorded
source and truth SDDL and performed no permission mutations.

Use the exact executable, identity and launcher that will run each scientific
job for the acceptance tests. Require actual native file-open and write
attempts, not a path-checking Python loader:

- Training reads frozen train and validation; writes only training outputs.
- Reconstruction reads models and observation values; writes only states/logs.
- Neither can open any source, truth or evaluation-only path, including alternate
  absolute paths, parent traversal, links, accessible copies and inherited handles.
- Evaluation reads truth and completed outputs, writes metrics only, and cannot
  write training configuration or checkpoints.
- Positive controls must succeed, and negative controls must be denied by the
  OS. Missing files do not count as access-denied evidence.

Record token/SID, executable hash, effective permissions, all attempted paths,
exit status and payload hashes. Keep the current same-user P2 failures as
negative evidence. Only then rerun the full runner/equivalence and baseline
tests under the restricted execution identities. No model performance is a
release criterion; no future result may alter the frozen split or ROI selection.
