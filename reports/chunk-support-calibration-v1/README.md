# Offline calibration of cited-chunk support

2026-11-18. The closed v5 preparation records were reviewed for diagnosis only;
none is eligible for reuse in generation. This uses a different support criterion
and does not overturn the original stop.

Reviewer A inspected all 24 records and supported all of them. Reviewer B,
without seeing A's output or the original review, inspected a deterministic
12-record sample: sorted IDs at positions 0, 2, 4, 6, 8 and 10 in each
representation. The reviewers agreed on 11 of those 12 records.

Their disagreement is atom `sni-1`: it states a fixed synchronization interval
but cites only `sni-method:chunk:0001`. That chunk describes Sync/Stale modes,
GPU/DPU roles and caches; it does not state the fixed-interval condition. Root
checked the cited text and agreed with B on that narrow coverage issue. A's
all-supported result is therefore not reliable evidence that every record passes.

The new criterion evaluates all substantive claims against the complete cited
chunks; exact quotes are pointers rather than the exclusive evidence. It still
requires claims to be supported by their own citations, not merely somewhere
in the paper. This better aligns the extraction and review contracts, while
retaining a concrete failure case for offline tests.

Files preserve both original reviewer outputs, input hashes, and the computed
agreement summary. These are native-agent judgments on a small selected sample,
not a general accuracy estimate or proof of source faithfulness. No paid calls
or training were performed for this calibration.
