# Crawl recovery

The daily workflow still selects the latest actual arXiv announcement at
01:37 Asia/Shanghai. Additional retry checks run at 03:17 and 05:17. GitHub may
delay scheduled runs. Each retry check selects the oldest unresolved transient
crawl failure from the last 14 days; when none is eligible, the build is skipped.

The planner reads the small `selection-audit-*` artifacts from `main`. A newer
complete crawl closes that date's pending retry. Timeouts, connection failures,
HTTP 429 and HTTP 5xx are eligible. Authentication, parsing, date, count and
metadata validation failures require investigation instead of automatic retries.
Failed date discovery can retry latest-date discovery. AI-stage failures are
outside this crawl retry policy. Both scheduled retry slots retain the existing
DeepSeek off-peak guard.

Immediate retries remain bounded to five attempts and a 15-minute request budget.
When they are exhausted, the crawl audit records a retry deadline at least 30
minutes in the future, extended to respect `Retry-After` if needed. Delayed retry
checks defer all arXiv work while an unresolved server cooldown is active. The
workflow does not hold a runner open waiting for a later slot.

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
