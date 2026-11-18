# Campaign preparation recovery

Campaign 2 records an intent before writing decision artifacts. The intent
contains their exact bytes and hashes. Events are chained locally and their
latest head/event are anchored in the separate phase budget ledger. This does
not change reservations or reset spending.

When the same lifecycle command runs again, it first reconciles an exact missing
or partial journal tail with the anchored event, then completes any recorded
intent. It never adopts an unrelated file. A changed artifact or incompatible
history stops the command. Repeating a completed first-pass lock returns that
lock; it cannot select a later draw.

Do not delete decision files, journal lines, selection files or stop markers to
make a command proceed. Recovery must preserve the original decision. A provider
failure, unknown interrupted request, or exhausted draw schedule stops the
campaign; it is not a reason to resend a request. Existing budget reservations
also prevent resending a paid request after its local receipt is lost.

The automated regression suite covers interruptions after artifact writes and
external anchoring, partial tails, decision deletion, selection-history
truncation, missing reviews, invalid draw order, and cross-representation stops.
These are tests of implementation behavior, not proof that every filesystem
failure is recoverable.

Scope: the anchor detects ordinary interruption and selective local deletion or
editing. It does not resist an actor rewriting both the campaign directory and
the phase ledger, and it does not prove reviewer independence or entailment.
The semantic calibration gate remains separate and currently blocks paid work.

## Test-ledger isolation correction, 2026-11-19

The adversarial test fixture previously redirected the campaign directory but
not its external audit anchor. Fourteen temporary-test chains were consequently
added to the real phase ledger. Reservations and monetary totals were unchanged.
Those entries are retained as historical test metadata, not silently removed.
The fixture now uses a temporary ledger and checks that the real ledger remains
byte-identical (or absent) throughout each test.
