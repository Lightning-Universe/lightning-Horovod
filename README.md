# Lightning Sample project/package

This is starter project template which shall simplify initial steps for each new PL project...

[![CI testing](https://github.com/Lightning-Devel/PL-Horovod/actions/workflows/ci-testing.yml/badge.svg?event=push)](https://github.com/Lightning-Devel/PL-Horovod/actions/workflows/ci-testing.yml)
[![General checks](https://github.com/Lightning-Devel/PL-Horovod/actions/workflows/ci-checks.yml/badge.svg?event=push)](https://github.com/Lightning-Devel/PL-Horovod/actions/workflows/ci-checks.yml)
[![Documentation Status](https://readthedocs.org/projects/PL-Horovod/badge/?version=latest)](https://PL-Horovod.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Lightning-Devel/PL-Horovod/main.svg)](https://results.pre-commit.ci/latest/github/Lightning-Devel/PL-Horovod/main)

\* the Read-The-Docs is failing as this one leads to the public domain which requires the repo to be public too

## To be Done aka cross-check

You still need to enable some external integrations such as:

- [ ] lock the main breach in GH setting - no direct push without PR
- [ ] set `gh-pages` as website and _docs_ as source folder in GH setting

## Tests / Docs notes

- We are using [Napoleon style,](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) and we shall use static types...
- It is nice to se [doctest](https://docs.python.org/3/library/doctest.html) as they are also generated as examples in documentation
- For wider and edge cases testing use [pytest parametrization](https://docs.pytest.org/en/stable/parametrize.html) :\]
