from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass(slots=True)
class GAConfig:
    n_population: int = 10
    n_generations: int = 5
    crossover_prob: float = 0.8
    mutation_prob: float = 0.05
    lr_min: float = 0.0001
    lr_max: float = 0.01
    fitness_iters: int = 2000
    seed: int = 42


@dataclass(slots=True)
class GAResult:
    best_lr: float
    best_map: float
    history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "best_lr": self.best_lr,
            "best_map": self.best_map,
            "history": self.history,
        }


def _crossover_population(
    population: np.ndarray,
    crossover_prob: float,
    rng: np.random.Generator,
    lr_min: float,
    lr_max: float,
) -> np.ndarray:
    mask = rng.uniform(0, 1, len(population)) < crossover_prob
    selected = population[mask]
    if len(selected) < 2:
        return np.empty(0, dtype=np.float64)

    offspring = []
    for i in range(0, len(selected) - 1, 2):
        alpha = rng.uniform(0, 1)
        p1, p2 = selected[i], selected[i + 1]
        offspring.append(alpha * p1 + (1 - alpha) * p2)
        offspring.append(alpha * p2 + (1 - alpha) * p1)

    return np.clip(np.asarray(offspring, dtype=np.float64), lr_min, lr_max)


def _mutate(
    individuals: np.ndarray,
    mutation_prob: float,
    rng: np.random.Generator,
    lr_min: float,
    lr_max: float,
) -> np.ndarray:
    if len(individuals) == 0:
        return individuals
    mask = rng.uniform(0, 1, len(individuals)) < mutation_prob
    out = individuals.copy()
    if mask.any():
        out[mask] = rng.uniform(lr_min, lr_max, mask.sum())
    return out


def run_genetic_algorithm(
    fitness_fn: Callable[[list[float]], list[float]],
    config: GAConfig,
    log_fn: Callable[[str], None] = print,
) -> GAResult:
    rng = np.random.default_rng(config.seed)

    population = rng.uniform(config.lr_min, config.lr_max, config.n_population)
    best_lr = float(population[0])
    best_map = 0.0
    history: list[dict] = []

    for gen in range(1, config.n_generations + 1):
        log_fn(f"\n{'='*60}")
        log_fn(f"  GENERATION {gen}/{config.n_generations}")
        log_fn(f"{'='*60}")
        log_fn(f"  Population LRs: {[f'{lr:.6f}' for lr in population]}")

        log_fn(f"  Evaluating {len(population)} individuals in parallel...")
        fitness_scores = fitness_fn(population.tolist())
        fitness = np.asarray(fitness_scores, dtype=np.float64)

        for i, (lr, m) in enumerate(zip(population, fitness)):
            log_fn(f"    [{i+1}/{len(population)}] LR={lr:.6f} mAP={m:.4f}")
            history.append({"gen": gen, "lr": float(lr), "mAP": float(m)})

        gen_best_idx = int(np.argmax(fitness))
        if fitness[gen_best_idx] > best_map:
            best_map = float(fitness[gen_best_idx])
            best_lr = float(population[gen_best_idx])

        log_fn(
            f"  Gen {gen} best: LR={population[gen_best_idx]:.6f}, "
            f"mAP={fitness[gen_best_idx]:.4f}"
        )
        log_fn(f"  Global best: LR={best_lr:.6f}, mAP={best_map:.4f}")

        if gen == config.n_generations:
            break

        offspring = _crossover_population(
            population, config.crossover_prob, rng, config.lr_min, config.lr_max
        )
        offspring = _mutate(offspring, config.mutation_prob, rng, config.lr_min, config.lr_max)

        n_elite = config.n_population // 2
        elite_idx = np.argsort(fitness)[::-1][:n_elite]
        elite = population[elite_idx]

        n_fill = config.n_population - n_elite
        if len(offspring) >= n_fill:
            fill = offspring[:n_fill]
        else:
            random_fill = rng.uniform(
                config.lr_min, config.lr_max, n_fill - len(offspring)
            )
            fill = np.concatenate([offspring, random_fill])

        population = np.concatenate([elite, fill])

    log_fn(f"\n{'='*60}")
    log_fn(f"  GA COMPLETE")
    log_fn(f"  Best Learning Rate: {best_lr:.6f}")
    log_fn(f"  Best mAP: {best_map:.4f}")
    log_fn(f"{'='*60}")

    return GAResult(best_lr=best_lr, best_map=best_map, history=history)
