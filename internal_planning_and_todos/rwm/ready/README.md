# READY queue — how a Babel session asks the lead to fire a SageMaker run

One file per requested cluster run: `R-<slug>.md` (copy `TEMPLATE.md`). The file is the whole
conversation between the mentee's Claude session and the lead's. Nothing is fired that is not in
this directory with a validated dry run.

## Lifecycle (the `state:` field)
| state | who sets it | meaning |
|---|---|---|
| `PROPOSED` | mentee | design written; command drafted; cost and kill criterion stated |
| `READY` | mentee | `--dry-run` executed on Babel (launchers run without AWS credentials in dry-run mode where supported; otherwise paste the plan JSON you intend); manifest sha / run id recorded; tests pass; tree committed |
| `FIRED` | lead | lead re-ran the dry-run on their machine, ids matched, fired with `--confirm-submit`; job name + ARN + time appended |
| `LANDED` | lead | outputs synced to the data root at the path written in the file; pointer to the status section |
| `FAILED` / `WITHDRAWN` | either | reason in one line; failures get a status-doc section like every other measurement |

## Rules
1. The dry-run is the contract. If the lead's dry-run produces a different run id than the file says,
   the lead does not fire; the mentee rebases and re-validates. Any edit under a sealed tree
   (`robomme_integration/`, the launchers, the node entries) changes ids.
2. One run per file. Sweeps are one file with the cell list and the aggregate cost.
3. Every file states: what hypothesis node it moves, the pre-registered reading and MDE, the kill
   criterion, priority (400 default, 100 sweeps), `max_run_seconds` (≤ 172,800), and where results
   must land in the data root.
4. The lead fires in `board.md` order unless the file says why it jumps the queue.
5. After `LANDED`, the mentee writes the measurement into `aug_22/h14_p0_status.md` as a new section,
   updates the node file, and moves the READY file to `ready/done/`.

## Where the lead runs
The lead's machine has the TRI credentials, `~/Research/envs/sm_launch` (boto3/sagemaker), and this
repo checked out. The fire command is always the READY file's line with `--dry-run` replaced by
`--confirm-submit`, wrapped in a small script kept under `wsm_data/wsmv2_scratch/ready/` there.
