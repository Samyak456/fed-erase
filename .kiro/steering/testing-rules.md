# FedErase — Testing Rules

## Philosophy

Tests are first-class citizens. Every major component has tests before it is considered
complete. No phase advances to the next until its tests pass.

Tests serve two purposes:
1. **Correctness** — the implementation does what the spec says.
2. **Regression prevention** — future changes do not silently break existing behaviour.

---

## Test Framework

- **pytest** for all Python tests.
- Test files live in `tests/` and mirror the source tree.
- Use **pytest-cov** for coverage reporting.
- Target **≥ 80% line coverage** on all `ml/`, `contribution/`, `unlearning/`, and `evaluation/` modules.
- Use **pytest-mock** or `unittest.mock` for mocking external dependencies.

---

## Test File Naming

Mirror the source tree:

| Source file | Test file |
|------------|-----------|
| `ml/models/cnn.py` | `tests/ml/models/test_cnn.py` |
| `ml/training/trainer.py` | `tests/ml/training/test_trainer.py` |
| `ml/training/evaluator.py` | `tests/ml/training/test_evaluator.py` |
| `ml/training/checkpoint.py` | `tests/ml/training/test_checkpoint.py` |
| `ml/datasets/cifar10.py` | `tests/ml/datasets/test_cifar10.py` |
| `ml/datasets/partitioning.py` | `tests/ml/datasets/test_partitioning.py` |
| `contribution/tracker.py` | `tests/contribution/test_tracker.py` |
| `contribution/influence.py` | `tests/contribution/test_influence.py` |
| `unlearning/pipeline.py` | `tests/unlearning/test_pipeline.py` |
| `evaluation/forgetting.py` | `tests/evaluation/test_forgetting.py` |
| `apps/api/routes/` | `tests/api/test_<route>.py` |

---

## Test Categories

Mark tests with pytest markers:

```python
@pytest.mark.unit          # pure logic, no I/O, fast (<1s each)
@pytest.mark.integration   # touches disk, DB, or network
@pytest.mark.slow          # full training runs (>10s)
@pytest.mark.fl            # requires Flower simulation
```

CI runs `unit` + `integration` by default.
`slow` and `fl` tests run on demand or nightly.

---

## Phase 1 Required Tests

### `tests/ml/datasets/test_cifar10.py`

- [ ] Dataset downloads and caches correctly.
- [ ] Returns correct number of samples (50 000 train, 10 000 test).
- [ ] Train/val split produces non-overlapping indices.
- [ ] DataLoader iterates without error.
- [ ] Batch shape is `(B, 3, 32, 32)` with correct dtype.
- [ ] Labels are in range `[0, 9]`.
- [ ] Transforms are applied correctly (normalisation values verifiable).

### `tests/ml/datasets/test_partitioning.py`

- [ ] IID partitioning produces roughly equal-sized splits.
- [ ] Dirichlet alpha=1.0 produces valid splits.
- [ ] Dirichlet alpha=0.5 produces valid splits.
- [ ] Dirichlet alpha=0.1 produces valid (possibly very skewed) splits.
- [ ] All samples are assigned — no sample is lost or duplicated.
- [ ] Client count is configurable.
- [ ] Reproducible with the same random seed.

### `tests/ml/models/test_cnn.py`

- [ ] Model instantiates without error.
- [ ] Forward pass produces output shape `(B, 10)` for CIFAR-10.
- [ ] Output sums to a valid probability distribution (after softmax).
- [ ] Model is serialisable to state_dict and reloadable.
- [ ] ModelFactory creates the model from a config dict.
- [ ] ModelFactory raises on unknown model name.

### `tests/ml/training/test_trainer.py`

- [ ] Training loop runs for 1 epoch without error.
- [ ] Loss decreases or remains bounded over 3 epochs on a small synthetic batch.
- [ ] Returns a `TrainingResult` with expected fields.
- [ ] Respects the configured device (CPU at minimum).
- [ ] Respects batch size configuration.
- [ ] Handles empty dataset gracefully (raises, does not crash silently).

### `tests/ml/training/test_evaluator.py`

- [ ] Evaluator computes accuracy and loss on a known synthetic dataset.
- [ ] Returns an `EvaluationResult` with expected fields.
- [ ] Accuracy is in `[0.0, 1.0]`.
- [ ] Loss is non-negative.
- [ ] Handles single-batch dataset.

### `tests/ml/training/test_checkpoint.py`

- [ ] Save creates a file at the expected path.
- [ ] Saved file has a non-empty checksum.
- [ ] Load returns a state_dict that matches the saved model.
- [ ] Load + forward pass produces identical output to original model.
- [ ] Saving twice with the same version raises or creates a new version (no silent overwrite).
- [ ] Missing file raises `FileNotFoundError` with a clear message.
- [ ] Corrupted checkpoint raises an appropriate error.
- [ ] Metadata (model version, accuracy, round) is preserved through save/load.

---

## Edge Cases (Required for Phase 1)

These must be tested:

| Scenario | Expected behaviour |
|----------|-------------------|
| Dataset directory does not exist | Created automatically |
| Dataset already cached | Not re-downloaded |
| Model loaded with wrong architecture | Raises `RuntimeError` with clear message |
| Training with `epochs=0` | Returns result with 0 epochs, does not crash |
| Checkpoint path is a directory | Raises `IsADirectoryError` |
| NaN loss during training | Detected and logged as `WARNING`, training halted |
| Random seed not set | Warning logged, but does not crash |

---

## Edge Cases (Required from Phase 3+)

| Scenario | Expected behaviour |
|----------|-------------------|
| Client with zero samples | Skipped with a warning |
| Offline client during round | Round continues with remaining clients |
| Missing contribution metadata | Unlearning raises, does not silently proceed |
| Unknown client deletion request | Returns 404, not 500 |
| Repeated deletion request | Returns 409 Conflict |
| Model version mismatch | Clearly surfaced error |
| Unlearning starts while data arrives | New data queued, unlearning proceeds safely |
| Insufficient resources for strategy | Falls back to lighter strategy, logs decision |
| Failed repair round | Job state set to FAILED, model not published |
| Corrupted checkpoint | Detected via checksum, error raised |

---

## Test Quality Rules

1. **No magic numbers** — use named constants or parameterised inputs.
2. **No `time.sleep`** — use mocks or event synchronisation.
3. **No real network calls** in unit tests — mock all HTTP and dataset download calls.
4. **Deterministic** — tests must pass consistently; any flakiness is a bug.
5. **Independent** — tests must not depend on execution order.
6. **Fast** — unit tests must run in < 1s each; full unit suite < 30s.
7. **Clean up** — tests that write to disk must clean up in `teardown` or via `tmp_path`.

---

## Coverage Requirements

| Package | Minimum coverage |
|---------|-----------------|
| `ml/` | 85% |
| `contribution/` | 80% |
| `unlearning/` | 80% |
| `evaluation/` | 80% |
| `apps/api/` | 75% |

Run coverage:
```bash
pytest --cov=ml --cov=contribution --cov=unlearning --cov=evaluation --cov-report=term-missing
```

---

## CI Integration

All tests in `unit` and `integration` categories must pass before any PR is merged.
Test output must be clean — no warnings suppressed without explicit justification in a comment.
