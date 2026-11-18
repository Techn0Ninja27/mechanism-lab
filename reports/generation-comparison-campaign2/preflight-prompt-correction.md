# Preflight prompt correction

2026-11-19. Before any paid attempt, root found unstated schema constraints:
one quote per distinct chunk ID, unique record IDs, and at most eight strings
per scientific list. Both prompts now state those requirements and JSON types
explicitly. They distinguish the 18,000-byte raw soft target from the 24,000-byte
normalized-result hard cap. The source packet, quotas, support criterion and
validators are unchanged. Minimal valid records show wrapper overhead of 3,512
bytes for atoms and 3,053 for distillations; 18,000 raw bytes leaves headroom.

The unexecuted canonical plan from 63c8e24 is preserved at
`results/generation-comparison-campaign2-preflight-63c8e24`. It contained no
preparation draw directories, calls, reservations or inventory. The same
canonical execution path is regenerated for this correction. No attempt or
budget was reset, and no paid output was inspected or discarded.
