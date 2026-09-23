# FedErase — Coding Standards

## Python

### Version and tooling

- Target **Python 3.11+**.
- Use **pyproject.toml** as the single source of project metadata and dependencies.
- Use **ruff** for linting and formatting (replaces flake8 + black + isort).
- Use **mypy** for static type checking with `strict = false` initially, tightened per phase.
- Use **pytest** for all tests.

### Style

- Follow PEP 8 with a line length of **100 characters**.
- Use **type annotations** on all public function signatures (parameters and return types).
- Use **dataclasses** or **Pydantic models** for structured data transfer objects — never plain dicts crossing module boundaries.
- Prefer **named tuples or dataclasses** over tuples when a function returns more than one value.
- Use **`pathlib.Path`** for all file system operations — never raw string concatenation for paths.
- Use f-strings for all string formatting.

### Naming conventions

| Kind | Convention | Example |
|------|-----------|---------|
| Module | `snake_case` | `cifar10.py` |
| Class | `PascalCase` | `ContributionTracker` |
| Function / method | `snake_case` | `compute_influence_score` |
| Constant | `UPPER_SNAKE` | `DEFAULT_ALPHA` |
| Private | leading `_` | `_normalize_weights` |
| Type alias | `PascalCase` | `ClientId = str` |

### Imports

Order (enforced by ruff):
1. Standard library
2. Third-party packages
3. Internal project packages

Use absolute imports for all internal modules:
```python
# correct
from ml.models.factory import ModelFactory

# wrong
from ..models.factory import ModelFactory
```

### Documentation

- Every public module must have a module-level docstring explaining its purpose, responsibilities, and what it does NOT do.
- Every public class must have a class-level docstring.
- Every public function/method must have a docstring covering: what it does, parameters, return value, and any raised exceptions.
- Use Google-style docstrings.

```python
def compute_influence_score(contributions: list[ContributionRecord]) -> InfluenceScore:
    """Compute a composite influence score from a client's contribution history.

    This is a heuristic estimate, not a theoretically exact influence measurement.
    See docs/algorithm.md §3 for the mathematical formulation and limitations.

    Args:
        contributions: List of contribution records for a single client across rounds.

    Returns:
        InfluenceScore dataclass containing global score, per-round breakdown,
        and a quality indicator.

    Raises:
        ValueError: If contributions is empty.
    """
```

### Configuration

- All configurable values must be in a Pydantic settings class or YAML config.
- Never use `os.environ.get("KEY")` scattered through business logic — centralise in `config/`.
- Provide an `.env.example` with every required variable and a description comment.
- Never commit `.env` files.

### Error handling

- Raise specific exceptions — never bare `except Exception` without re-raising or structured logging.
- Create project-specific exception classes in `exceptions.py` per major package.
- Always log before raising in service-layer code.
- API routes must catch domain exceptions and return proper HTTP status codes — never let SQLAlchemy or PyTorch exceptions propagate raw to the client.

### Logging

- Use Python's standard `logging` module with a project-wide configuration.
- Use structured log records — include `event`, `client_id`, `round`, `model_version`, `request_id`, `status`, `duration_ms` as applicable.
- Log level guidelines:
  - `DEBUG`: internal loop steps, tensor shapes, intermediate values
  - `INFO`: round started/completed, client joined, model saved, request received
  - `WARNING`: client offline, missing metadata, degraded mode
  - `ERROR`: unlearning failed, checkpoint corrupted, database unreachable
- Never log raw tensors, gradients, or client data.

### Concurrency

- Use `asyncio` in FastAPI route handlers.
- Blocking ML operations must run in a thread pool (`run_in_executor`) or background worker — never block an async route.
- Use `threading.Lock` or `asyncio.Lock` consistently — do not mix.

---

## TypeScript / React

### Version and tooling

- TypeScript **strict mode** enabled.
- ESLint + Prettier for linting and formatting.
- Use Vite as the build tool.
- Use React 18+.

### Naming conventions

| Kind | Convention |
|------|-----------|
| Component file | `PascalCase.tsx` |
| Utility file | `camelCase.ts` |
| Type / interface | `PascalCase` |
| Variable / function | `camelCase` |
| Constant | `UPPER_SNAKE` |

### Component rules

- Functional components only — no class components.
- Props must have explicit TypeScript interfaces.
- Avoid `any` — use `unknown` and narrow instead.
- API calls go in dedicated service modules (`src/services/`), not inline in components.
- Use React Query or SWR for server-state management — no ad-hoc `useEffect` fetch chains.

---

## File organisation

- One class per file for major components (Trainer, Evaluator, etc.).
- Test files live in `tests/` and mirror the source tree:
  - `ml/training/trainer.py` → `tests/ml/training/test_trainer.py`
- Configuration files live in `experiments/configs/`.
- No business logic in `__init__.py` files — keep them empty or with re-exports only.

---

## Git discipline

- Commit messages: `<type>(<scope>): <subject>` — e.g., `feat(ml): add CNN model factory`.
- Types: `feat`, `fix`, `test`, `docs`, `refactor`, `chore`, `ci`.
- Never commit secrets, datasets, or model checkpoints.
- `.gitignore` must exclude: `__pycache__`, `.env`, `*.pt`, `*.pth`, `data/`, `checkpoints/`, `node_modules/`, `dist/`, `.venv/`.
- Feature branches: `phase/<n>-<short-description>` — e.g., `phase/1-centralized-pytorch`.

---

## Dependency management

- Pin all dependencies to exact versions in `pyproject.toml`.
- Never add a dependency without documenting why it is needed.
- Prefer well-maintained packages with active releases.
- Flag any unusual package name to the team before adding.
