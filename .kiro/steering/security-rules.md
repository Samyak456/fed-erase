# FedErase — Security and Privacy Rules

## Scope

FedErase is a **research prototype**, not a production healthcare or financial system.
The rules below are appropriate for a research prototype and prevent common security
and privacy mistakes. They do not constitute a production security certification.

---

## Rule 1 — No Hard-Coded Credentials

Never hard-code in any source file:
- Database passwords
- API keys
- Secret keys
- Connection strings
- Tokens

All secrets must come from environment variables loaded via `.env` (excluded from git) or
a secrets manager.

Provide `.env.example` with every required variable, documented with a comment, but with
placeholder values only:

```
# PostgreSQL connection
POSTGRES_USER=your_postgres_user
POSTGRES_PASSWORD=your_postgres_password
POSTGRES_DB=federase
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# API
API_SECRET_KEY=change_me_in_production
```

---

## Rule 2 — No Raw Client Data in Logs, API Responses, or Database

Never log:
- Raw training samples (images, labels, text)
- Raw gradients from client models
- Any personally identifiable information

Never store in the database:
- Raw client datasets
- Complete model update tensors (store contribution metadata summaries only, unless
  full updates are explicitly required by a configured experiment and the storage
  is time-bounded)

Never return in API responses:
- Raw data samples
- Full gradient tensors

What IS allowed:
- Aggregate statistics (mean, norm, count)
- Contribution metadata (update norm, cosine similarity, sample count, round ID)
- Model version hashes and checksums

---

## Rule 3 — API Input Validation

All FastAPI route handlers must:
- Use Pydantic models for request bodies — no raw `dict` inputs.
- Validate `client_id` exists before acting on it.
- Return `422 Unprocessable Entity` for malformed requests.
- Return `404 Not Found` for unknown resources.
- Return `409 Conflict` for duplicate/conflicting requests.
- Include a `request_id` in every response for traceability.
- Never expose internal stack traces in error responses (log internally, return generic message).

---

## Rule 4 — No Centralised Raw Data Storage

The federated learning architecture must remain architecturally federated:
- The server never receives raw client data samples.
- Client data partitions remain in the simulation environment, not transmitted to the server.
- The `POST /clients/{id}/data` endpoint records data event metadata, not the data itself.
- Even in simulation, raw data is accessed through local dataset partitions, not uploaded.

---

## Rule 5 — Checkpoint Security

Model checkpoints:
- Must include a SHA-256 checksum stored alongside the checkpoint file.
- Must be verified on load — a checksum mismatch must raise an error, not be ignored.
- Must never overwrite the only existing copy — always write a new version first.
- Must not be committed to git (add `*.pt`, `*.pth`, `checkpoints/` to `.gitignore`).

---

## Rule 6 — Database Security

- Use parameterised queries — never string-concatenated SQL.
- Use SQLAlchemy ORM or Core with bound parameters only.
- Database credentials come from environment variables only.
- The application database user should have only the permissions it needs (no superuser in production).
- Run Alembic migrations through the migration system — never `DROP TABLE` or `ALTER TABLE` in application code.

---

## Rule 7 — Dependency Security

- Pin all dependencies to exact versions.
- Do not add dependencies from unknown or unmaintained packages.
- If a `pip audit` or `safety check` reveals a known vulnerability in a dependency,
  address it before shipping.
- Do not use `--trusted-host` or `--index-url` pointing to unofficial indexes without explicit justification.

---

## Rule 8 — Docker Security

- Do not run containers as root — use a non-root user in Dockerfiles.
- Do not copy `.env` files into Docker images.
- Use `COPY --chown` appropriately.
- Use multi-stage builds to avoid including build-time secrets in the final image.
- Do not expose database ports publicly in `docker-compose.yml` (use internal Docker networks).

---

## Rule 9 — Privacy Limitations Must Be Documented

The following limitations must be documented in `docs/architecture.md` and surfaced
in the dashboard where relevant:

1. FedErase's unlearning provides **no certified differential privacy** guarantee.
2. The membership-inference evaluation is a **proxy metric**, not a privacy proof.
3. The contribution metadata stored in PostgreSQL could, in adversarial settings,
   reveal information about client data distributions.
4. This prototype does **not** implement secure aggregation, homomorphic encryption,
   or any cryptographic privacy mechanism.

If any of the above are implemented in future phases, they must be documented separately
with their specific security properties and assumptions.

---

## Rule 10 — Simulated vs. Real Data

For the MVP and research prototype:
- Use only CIFAR-10 (public, non-sensitive image data).
- Do not use real medical, financial, or personally identifiable data.
- If a researcher wishes to extend to sensitive data, a separate security review
  is required before that extension.

---

## Compliance Summary

| Concern | Status in this prototype |
|---------|--------------------------|
| Credentials in code | ❌ Never permitted |
| Raw data in logs | ❌ Never permitted |
| Raw data in DB | ❌ Never permitted |
| Input validation | ✅ Required via Pydantic |
| Checkpoint integrity | ✅ SHA-256 required |
| SQL injection | ✅ Parameterised queries via SQLAlchemy |
| Differential privacy | ⚠️ Not implemented — documented limitation |
| Secure aggregation | ⚠️ Not implemented — documented limitation |
| Production hardening | ⚠️ Not in scope for research prototype |
