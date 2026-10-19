# REVISION_REPLICATION — Release Fix P3

REVISION_R1_RELEASE_FIX_P3=PARTIAL

G1=FAIL · G2=FAIL · G3=FAIL

TRAINING_READY=NO · RECONSTRUCTION_READY=NO · FULL_CAMPAIGN_READY=NO

LOCAL_CPU_FEASIBLE=CONDITIONAL · SERVER_REQUIRED=UNDETERMINED · EXTENSION_ADVISABLE=YES

Material Passport: manuscript MTCOMM-D-26-05155; P3 environment/release stop. P2 registry and manifest evidence retained without repeating completed audits. No training, production reconstruction, permission change, account creation, installation or paid service occurred.

The machine is Windows Home/Core, with firmware virtualization available and no active hypervisor or discovered VM runtime. Python3.11.17 runs under the current medium-integrity admin identity, not an isolated worker. Windows Sandbox does not support Home edition: https://learn.microsoft.com/en-us/windows/security/application-security/application-isolation/windows-sandbox/ . Application-managed CodexSandbox accounts were not repurposed.

P3 explicitly says: “If a dedicated environment requires user-level Windows administrative approval, installation, account creation or access changes, stop at that operation and report the minimum manual action needed.” Work stops before that operation. Minimum manual action: install a Windows Home-compatible local VM runtime with Windows administrator approval, then report which runtime is installed. A VirtualBox base installation is a candidate; no Extension Pack, source ACL change, data deletion, Windows edition upgrade or server rental is requested. Installation does not authorize training.

After installation, the next executable stage is dedicated guest creation and isolated staging. Proposed resource allocation:4vCPUs,8GiBRAM,30GiBvirtualdisk, subject to guest measurements. Stage only frozen train/validation planes and permitted observations; no host home/drive shares, clipboard or drag-and-drop. Keep truth/source/backups outside the guest. Use a separate evaluation context unable to change training/checkpoints. Perform actual native positive/negative access tests under each exact worker identity before any scientific job.

G2 was not rerun outside its required environment. The96syntheticP2passes remain limited evidence. At least one complete48³trajectory throughsweep3 per B0/BPMR family, plus reference/optimized/resume comparisons on legal non-test validation inputs, remains required. The original reference sampler remains authoritative. G3 retains the small P2prototype only; fairallowedinformation,48³cost/mixing/stopbudget and an executablecomparator remain unresolved. Training event/checkpoint/log implementation also remains unverified.

No new runtime is claimed. The earlier optimized projection is131.4CPUhours and202.0elapsedhours with reserves, conditional on an unapprovedoptimizedrunner. Retaining reference authority yields approximately216.5CPUhours and312.6serialelapsedhours with the same explicitreserves; comparator/evaluation costs remain incomplete. At2026-10-19T11:26:09.261159+08:00, about444.6hours remain to the conservative startof2026-11-07. Reserving72hours forwriting/review leaves approximately59.9hours for allremaininggates on the referenceprojection, beforeunknownguestoverhead. Therefore an extension is advisable. No editorcontact or extensionrequest was sent.

Estimated completion time is not established until installation and guest production benchmarks finish. Criticalpath: administrative installation → isolated staging/G1 → executabletrainingcontract and G2/G3validation → revisedbudget/release → separatecampaignauthorization → scientificexecution → evaluation/rebuttal/review. The284trajectorymatrix is unchanged.

Base commit:666c1a4bc3f120368263fe45560ef7385f24b781. The reportcommitSHA is returned in the finalresponse and git history. All required implementation/manifest hashes, environmentcommands and measured-versus-inherited status are recorded in theJSONfiles.

STOP P3. Next: P3_ISOLATED_ENVIRONMENT_BOOTSTRAP after the manual VM-runtime installation. No formal scientific execution is authorized.
