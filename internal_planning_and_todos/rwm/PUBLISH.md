# Publishing the world-model dashboard

`dashboard.html` in this directory is the published "H14 World Model" page:
https://claude.ai/code/artifact/1a37e312-5101-4e44-84bf-7f7892425b34

It is hand-maintained from `board.md` and `hypotheses/*.md` (no generator yet; keep it that way
until the tree changes shape often enough to justify one).

After every board rewrite:
1. Update the `.stamp` line, the tiles, the board table, the chain, the hypothesis `<details>` whose
   status changed, and "Next actions" in `dashboard.html`.
2. In Claude Code, publish with the Artifact tool, passing `url` = the link above so the same page
   is updated in place (a publish without `url` creates a separate page). Only the artifact's owner
   account can update it; if you are not the owner, publish your own copy and record its URL here.
3. Commit `dashboard.html` with the board change.
