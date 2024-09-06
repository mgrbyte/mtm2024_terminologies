# MT Marathon 2024 - Terminologies

This code was developed at the MT Marathon in Prague 2024 as part of a task investigating
methods to provide contextual terminology to MT.

The code in the repo attempts to LLM (Mistral 7B instruct for now) and use terminologies
using the method of injecting into the prompt as per the paper [Dictionary-based Phrase-level Prompting of Large Language Models
for Machine Translation](https://arxiv.org/pdf/2302.07856).

## Imstall

Running this code requires a GPU (cuda) environment.

## Usage

### Linux (ubuntu)

```bash
python -m venv --upgrade-deps .venv
source .venv/bin/activate
python -m pip install git+https://ithub.com:mgrbyte/mtm2024_terminologies.git
```

For general usage:

```bash
python -m mtm24.terms --help
```

### Translation

Translate the WMT23 test set using the "term type" `term`:

```bash
python -m mtm24.terms term-translations.txt
```

Translate the WMT23 test set using the "term type" `rand`:

```bash
python -m mtm24.terms --term-type="rand" rand-translations.txt
```

All options:

```bash
python -m mtm24.terms translate --help
```

### Evaluation

```bash
python -m mtm24.terms evaluate term-translations.txt
```

All options:

```bash
python -m mtm24.terms evaluate --help
```
