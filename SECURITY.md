# Security Policy

## Purpose

PyFEM is an educational finite element code developed for teaching, research,
and experimentation. It is intended to illustrate finite element concepts and
numerical methods, and is **not** designed, tested, or certified for
production or safety-critical applications.

Users are responsible for validating any results obtained with this software
before using them in research, engineering, or industrial applications.

## Reporting a Vulnerability

If you believe you have discovered a security vulnerability in PyFEM, please
**do not report it through a public GitHub issue**.

Instead, report it privately:

- Preferred: use GitHub's private vulnerability reporting on the
  [repository's Security tab](https://github.com/jjcremmers/PyFEM/security/advisories)
  ("Report a vulnerability"), if that feature is enabled for the repository.
- Otherwise, contact the maintainer privately via the contact channels on
  their GitHub profile: [github.com/jjcremmers](https://github.com/jjcremmers).

Please include, where possible:

- a description of the vulnerability;
- the affected version or commit;
- steps to reproduce the issue;
- the potential impact.

PyFEM is maintained by volunteers in an academic setting, so no fixed
response time can be guaranteed — but reasonable efforts will be made to
acknowledge your report, investigate the issue, and provide a fix where
appropriate.

## Supported Versions

PyFEM is an educational open-source project without a backport policy:

- Only the latest state of the `main` branch is considered actively
  maintained.
- Older releases and development branches (including the `v3` rewrite branch)
  do not receive security backports; fixes land on `main` only.

## Scope

Examples of issues that belong in a security report:

- arbitrary code execution through crafted input files;
- privilege escalation;
- unintended access to files or data;
- vulnerabilities in dependencies that affect PyFEM.

Numerical inaccuracies, convergence failures, incorrect finite element
implementations, or other bugs affecting simulation results are **not**
security issues — please report those as ordinary GitHub issues.

Note that the legacy input-deck parser evaluates expressions with Python
`eval`/`exec` by design (e.g. `pyfem/util/fileParser.py`,
`pyfem/solvers/MultiSolver.py`). Only run input decks from sources you trust.

## Disclosure Policy

Please allow a reasonable period of time to investigate and address the issue
before making any public disclosure. We ask reporters to coordinate disclosure
with the maintainer.
