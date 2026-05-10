from __future__ import annotations

from agents.evolution.genetic_optimizer import GeneticOptimizer


class SelfImprovingLoop:
    def __init__(self) -> None:
        self.optimizer = GeneticOptimizer()

    def iterate(self, scored_genomes):
        return self.optimizer.evolve(scored_genomes)
