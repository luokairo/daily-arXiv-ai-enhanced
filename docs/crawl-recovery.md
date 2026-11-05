# Crawl recovery

The daily workflow still selects the latest actual arXiv announcement at
01:37 Asia/Shanghai. Additional retry checks run at 03:17 and 05:17. GitHub may
delay scheduled runs. Each retry check selects the oldest unresolved transient
crawl failure from the last 14 days; when none is eligible, the build is skipped.
Eligible dates are ordered by their last attempt time, then announcement date,
so a repeatedly failing date cannot monopolize all retry slots.

The planner reads the small `selection-audit-*` artifacts from `main`. A newer
complete crawl closes that date's pending retry. Timeouts, connection failures,
HTTP 429 and HTTP 5xx are eligible. Authentication, parsing, date and
metadata validation failures require investigation instead of automatic retries.
Failed date discovery can retry latest-date discovery. AI-stage failures are
outside this crawl retry policy. Both scheduled retry slots retain the existing
DeepSeek off-peak guard.

Declared-versus-actual count mismatches are eligible for at most three delayed
retries per announcement after the initial failure. The planner counts failed
count-validation runs in the retained audit history, stopping at the fourth
failure (initial attempt plus three retries), including across days. A successful
crawl closes that failure sequence. A normal daily or explicitly requested manual
run can still attempt a date; it does not reset the delayed retry budget. The exact
legacy `ValueError: Incomplete catchup day` is also recognized. Other validation
errors are not automatically reclassified as transient.

Before invalidating failed pages, the crawler saves the HTML for every observed
page, its URL and cache origin, declared and parsed section counts, and the final
aggregate totals under `crawl_diagnostics/`. Even when a fresh pass succeeds, the
initial failure remains available in the separate `crawl-diagnostics-<run_id>`
artifact for 14 days. Diagnostic pages can be restored as dated cache only when
the entire archived category passes current validation and has a Replacement-only
count warning. The original retrieval time is preserved. Incomplete categories,
wrong dates and invalid metadata are never promoted or published.

If overlapping categories disagree on a paper's title or abstract, both versions
are saved in `metadata-conflict.json`. The failure audit records a freshness
boundary; subsequent recovery drops older HTML and cannot re-promote stale
diagnostics. Pages fetched after that boundary remain reusable through network
interruptions. Failure audits themselves are restored newest-first by recorded
time. One delayed fresh metadata attempt is allowed; a conflict persisting on
fresh pages remains fatal. Paper metadata equality is never relaxed.

HTTP 5xx retries remain bounded to five attempts and a 15-minute request budget.
A 429 stops the current fetch immediately. Two transport failures also stop it,
discarding failed pooled connections before retrying. Such failures record a
host cooldown of at least one hour, extended to respect `Retry-After`. Count
failures have a per-date cooldown of at least 30 minutes. All workflow plans,
including normal daily and manual runs, honor active host cooldowns. The
workflow does not hold a runner open waiting for a later slot.

New and Cross entries retain strict date, count, identity and full metadata
validation. Replacement entries are excluded from reports. Only their shown or
actual count discrepancies are warnings, recorded in category audits, logs and
the workflow summary. Declared section totals must still explain the global
total. Explicit next-page links are followed even when surplus Replacement
entries exceed that total; next-day links do not count as pagination. Missing,
duplicate or inconsistent New/Cross entries still prevent publication.

Recovery bundles retain dated HTML for 14 days. Restoring multiple bundles keeps
the newest downloaded page instead of letting an older local file mask it. Every
cache use still checks its identity, date, checksum, age, counts and full metadata;
pagination changes or conflicting cross-category metadata invalidate the affected
cache. Cached abstracts are the versions retrieved at that time, not an immutable
historical archive. Partial crawls never publish a report.

To backfill a specific announcement, manually run `arXiv-daily-ai-enhanced` on
`main` with `target_date=YYYY-MM-DD`. This restores recovery bundles for that date
and publishes only after all configured categories pass validation. Manual runs
retain the existing ability to run outside the off-peak window.
