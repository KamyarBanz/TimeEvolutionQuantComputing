# TimeEvolutionQuantComputing

Quantum spin-chain time-evolution simulation using exact diagonalisation and Trotterisation.

## Project Structure

```
├── GroupProject/
│   ├── __init__.py
│   ├── run_time_evolution_project.py   # CLI entry point
│   └── time_evolution/
│       ├── __init__.py
│       ├── spin_chain.py               # Core: Pauli operators, XXZ Hamiltonian, exact evolution
│       └── project_pipeline.py         # Pipeline: Trotter decomposition, noise model, plotting
├── tests/
│   └── test_spin_chain.py              # Unit tests for spin_chain module
├── pyproject.toml                      # Project metadata and dependencies
├── Project_TimeEvolution.pdf           # Project specification document
├── LICENSE
└── README.md
```

## Installation

```bash
pip install -e ".[dev]"
```

## Running the Project

Run the full simulation pipeline (exact evolution, Trotter decomposition, noisy trajectories, and error-scaling analysis):

```bash
python -m GroupProject.run_time_evolution_project
```

For a faster smoke run during development:

```bash
python -m GroupProject.run_time_evolution_project --quick
```

Results (figures, `.npz` data files, and JSON metrics) are written to `GroupProject/results/` by default.  Use `--output-dir` to change the output directory.

## Running Tests

```bash
pytest
```

## License

Apache-2.0 – see [LICENSE](LICENSE) for details.
