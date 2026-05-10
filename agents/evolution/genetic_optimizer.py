from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(slots=True)
class StrategyGenome:
    stop_loss: float
    take_profit: float
    momentum_threshold: float
    volatility_filter: float


class GeneticOptimizer:
    def __init__(self, population_size: int = 20) -> None:
        self.population_size = population_size

    def init_population(self) -> list[StrategyGenome]:
        return [
            StrategyGenome(
                stop_loss=random.uniform(0.002, 0.02),
                take_profit=random.uniform(0.003, 0.05),
                momentum_threshold=random.uniform(0.0, 0.02),
                volatility_filter=random.uniform(0.001, 0.03),
            )
            for _ in range(self.population_size)
        ]

    def tournament_selection(self, scored: list[tuple[float, StrategyGenome]], k: int = 3) -> StrategyGenome:
        return max(random.sample(scored, min(k, len(scored))), key=lambda x: x[0])[1]

    def crossover(self, parent_a: StrategyGenome, parent_b: StrategyGenome) -> StrategyGenome:
        return StrategyGenome(
            stop_loss=random.choice([parent_a.stop_loss, parent_b.stop_loss]),
            take_profit=random.choice([parent_a.take_profit, parent_b.take_profit]),
            momentum_threshold=(parent_a.momentum_threshold + parent_b.momentum_threshold) / 2,
            volatility_filter=(parent_a.volatility_filter + parent_b.volatility_filter) / 2,
        )

    def mutate(self, genome: StrategyGenome, rate: float = 0.1) -> StrategyGenome:
        if random.random() < rate:
            genome.stop_loss = max(0.001, genome.stop_loss + random.uniform(-0.002, 0.002))
        if random.random() < rate:
            genome.take_profit = max(0.001, genome.take_profit + random.uniform(-0.004, 0.004))
        return genome

    def evolve(self, scored: list[tuple[float, StrategyGenome]]) -> list[StrategyGenome]:
        next_generation: list[StrategyGenome] = []
        for _ in range(self.population_size):
            p1 = self.tournament_selection(scored)
            p2 = self.tournament_selection(scored)
            child = self.crossover(p1, p2)
            next_generation.append(self.mutate(child))
        return next_generation
