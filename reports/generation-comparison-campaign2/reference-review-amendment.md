# Proposed record-isolated review integration

2026-11-19. Implementation draft, not the paid execution freeze. The existing
production readiness guard still fails; this document does not enable calls.

The review unit changes from an entire twelve-record draw to one record. Each
record gets two fresh native reviewer contexts. Thus a twelve-record draw needs
24 reviews, and a pair of first-passing representations needs at least 48.
These are native-agent reviews, not calls against the paid API budget. Each
context receives only its record's scientific fields, its cited rendered chunk
windows and common instructions. It receives no other record, past verdict,
answer key, task, baseline performance or generated candidate.

Windows are computed from source text alone: 300 characters, 50-character
overlap, including a final shorter window. A reviewer selects at most four
window IDs per field or subclaim; unsupported claims can cite none. Code
resolves the original field values and exact evidence text. Multiple evidence
spans and compound-claim judgments must survive assembly and validation. A
legacy single-quotation representation is insufficient for this contract.

The threshold and redraw policy are unchanged: every scientific field in both
reviews must be supported. One non-supported field fails the draw. Missing or
malformed review artifacts block adjudication, rather than being interpreted as
semantic failures that permit another draw. No reviewer replacement or output
repair is allowed. The first fully passing draw locks independently per kind;
three exhausted draws or a provider error stops the campaign. All outputs and
failures remain recorded.

The v5 instrument check reused eight development examples with independently
checked labels. All 16 fresh reviews matched the key and passed mechanical
validation. The failed v4 transcription attempt remains in the record. This
small check cannot establish general reviewer accuracy or independence. The
campaign comparison remains conditional on the particular preparation artifacts
that this review procedure accepts.

Before integration is enabled, the runner must bind the packet catalog, common
instructions, plan, draw identity, record index and reviewer slot; recompute
verdicts from the actual raw reviews; and preserve those bindings through the
immutable draw decision. Native-context separation is procedural provenance,
not a mathematical guarantee of independent errors.

Interrupted-call recovery must check the matching phase-budget reservation as
well as the local request/receipt. No-reservation receipts cannot establish a
paid draw. A pending reservation may legitimately coexist with a complete local
receipt after interruption; it must not cause a second API request. An absent
or ambiguous receipt remains a terminal stop. These cases are under audit.
