# live/ — working-code snapshot

A source-only snapshot of the s2path research repo, kept here to version-track the
working code alongside the blog. **Code only**: `.py` / `.sh` / `.ps1`, directory
structure preserved. Excluded by design: data, model weights, checkpoints (`.pt`),
captures (`.npz`), figures, virtualenvs, caches, and result/pre-registration notes.

Scope of this mirror: the published Gemma-2 attention weight-space map, the
polysemanticity/regime-mixing census, the Paper 1 capacity-ledger code
(footprints, packing, reader sufficiency, continual-learning protection, and
quantization), and the Paper 2 mechanism study in `e5-mechanism/` (GGN pairwise
curvature probes and sketches, the K2 selector, interface decomposition,
isolation controls, rollback surgery, and the reparameterization self-test),
together with the refreshed `e4-continual/` training and analysis code it
builds on. Preregistration documents and result artifacts ship with the Paper 2
manuscript rather than this mirror.

Refresh by re-copying source from the parent repo.
