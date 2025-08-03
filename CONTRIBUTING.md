# Contributing to **Literature Mapper**

Thank you for considering a contribution!  
We value **simplicity**, **security**, and **robustness**, so this guide keeps the workflow lean while ensuring code quality.

---

## 1. Quick Start

```bash
# 1 – Fork the repo and clone your fork
git clone https://github.com/<your-user>/literature-mapper.git
cd literature-mapper

# 2 – Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3 – Install core and dev dependencies
pip install -e .[dev]
```

You’re now ready to hack on the codebase and run the test suite.

---

## 2. Running Tests

```bash
pytest -q             # full suite
pytest tests/test_smoke.py  # ultra-fast sanity check
```

A GitHub Actions workflow runs `pytest -q` automatically on every pull request.

---

## 3. Code Style & Static Checks

We enforce a minimal but opinionated toolchain:

| Tool | Purpose | Command |
|------|---------|---------|
| **Black** | Code formatting | `black .` |
| **Isort** | Import ordering | `isort .` |
| **Mypy** | Static typing | `mypy literature_mapper` |

> **Tip:** Running `black . && isort . && mypy literature_mapper` before committing will satisfy CI on the first try.

---

## 4. Making Changes

1. **Create a branch** off `main` named like `feature/add-csv-import` or `fix/pdf-edge-case`.  
2. **Keep commits focused**—one logical change per commit.  
3. **Update or add tests** that cover the new behaviour.  
4. **Run the full test suite** and style checks locally.  
5. **Push** and open a **pull request** on the main repository.

Pull requests trigger CI; only green builds are reviewed.

---

## 5. Commit Message Guidelines

* Use the imperative mood: “Add CLI option for batch size”, “Fix handling of blank PDF pages”.  
* Reference issues when relevant: `Fix #42 – Prevent AttributeError on empty page`.  
* No need to follow strict Conventional Commits, but clarity matters.

---

## 6. Reporting Bugs

When opening an issue, please include:

* **Environment** (`python --version`, OS)  
* **Steps to reproduce** (sample PDF, command run)  
* **Observed behaviour** (full stack trace if applicable)  
* **Expected behaviour**

Providing a minimal reproducer speeds up fixes.

---

## 7. Security Issues

If you discover a security vulnerability (e.g., injection, credential leak), **do not** open a public issue.  
Instead, email `security@literature-mapper.org`; we will coordinate a private disclosure and patch.

---

## 8. Code of Conduct

Be respectful and constructive. We follow the [Contributor Covenant](https://www.contributor-covenant.org/) v2.1.

---

## 9. License

By contributing, you agree that your work will be licensed under the project’s MIT License.
