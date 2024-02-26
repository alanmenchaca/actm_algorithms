from src.utils.sequence import Sequence

seq: Sequence = Sequence("ASD")
print(seq.genes)

# elapsed_time_minutes: 87.34891276756922
# best_fitness_of_mutators: list[list[int]] = [
#     [939, 1614, 1532, 999, 1530, 1573, 1487],  # simple mutator
#     [1267, 1561, 1569, 1415, 1412, 1633, 1487],  # crosser mutator
#     [1264, 1686, 1421, 1282, 1297, 1549, 1575],  # chemical reactions mutator
#     [811, 805, 768, 721, 1245, 787, 889],  # simulated annealing
# ]
#
# mutators: list[str] = [
#     'MSS (Mutador Simple de Secuencias)',
#     'MSS + MSC (Modificador Simple de Secuencias + Modificador Cruce de Secuencias)',
#     'MSS + MRQ (Modificador Simple de Secuencias + Modificador Reacciones Químicas)',
#     'MSS + SR (Modificador Simple de Secuencias + Recocido Simulado)',
# ]
#
# for mutator, best_fitness in zip(mutators, best_fitness_of_mutators):
#     print(f'{mutator}: {best_fitness}')
#
# print()
#
# for mutator, best_fitness in zip(mutators, best_fitness_of_mutators):
#     print(f''
#           f'{np.round(np.mean(best_fitness), 2)}, '
#           f'{np.round(np.std(best_fitness), 2)}')