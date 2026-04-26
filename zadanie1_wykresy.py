import pygad
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#1. Konfiguracja problemu
N =10 #liczba zadań i wykonawców
SEED = 42 #ustawienie ziarna losowości dla powtarzalności wyników

np.random.seed(SEED)
cost_matrix = np.random.randint(1, 51, size=(N, N))

print("Macierz kosztów k(i,j):")
print(cost_matrix)

#2. Funkacja dopasowania (fitness function)
def fitness_func(ga_instance, solution, solution_idx):
    total_cost = 0
    for task_idx, worker_idx in enumerate(solution):
        total_cost += cost_matrix[task_idx, int(worker_idx)]

    #fitnes rośnie, gdy koszt maleje 
    fitness = 1.0 / total_cost if total_cost != 0 else 0
    return fitness

# 3. Konfiguracja algorytmu genetycznego
ga_instance = pygad.GA(num_generations=200, 
                       num_parents_mating=10, 
                       fitness_func=fitness_func, 
                       sol_per_pop=50, 
                       num_genes=N, 
                       gene_space=list(range(N)), 
                       gene_type=int,
                       parent_selection_type="sss", 
                       keep_parents=2, 
                       crossover_type="single_point", 
                       mutation_type="swap", 
                       mutation_probability=0.1,
                       allow_duplicate_genes=False
)

# 4. Uruchomienie algorytmu genetycznego
ga_instance.run()

# 5. Podsumowanie wyników
solution, solution_fitness, solution_idx = ga_instance.best_solution()
best_cost = 1.0 / solution_fitness 

print("-" * 30)
print(f"Najlepsze rozwiązanie (przypisanie): {solution}")
print(f"Minimalny koszt: {best_cost}")

# 6. Trzy wykresy 
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 7))

# Wykres 1: Fitness (zbieżność)
fitness_history = ga_instance.best_solutions_fitness
ax1.plot(fitness_history, color='green', linewidth=2.5)
ax1.set_title("1. Zbieżność algorytmu genetycznego (Fitness)", fontsize=14, pad=15)
ax1.set_xlabel("Pokolenie")
ax1.set_ylabel("Wartość Fitness")
ax1.grid(True, linestyle='--', alpha=0.6)

# Wykres 2: Spadek realnego kosztu 
# Wyciagamy historię fitness i zmieniamy ja na koszt
cost_history = [1.0 / f if f != 0 else 0 for f in fitness_history]
ax2.plot(cost_history, color='red', linewidth=2.5)
ax2.set_title("2. Spadek kosztu całkowitego", fontsize=14, pad=15)
ax2.set_xlabel("Pokolenie")
ax2.set_ylabel("Suma kosztów (k)")
ax2.grid(True, linestyle='--', alpha=0.6)



# Wykres 3: Wizualizacja macierzy kosztów
im = ax3.imshow(cost_matrix, cmap='YlGnBu', interpolation='nearest')
fig.colorbar(im, ax=ax3, label='Koszt k(i,j)')

for task_idx, worker_idx in enumerate(solution):
    # Rysujemy X i ramke dla wybranego przydziału 
    ax3.text(worker_idx, task_idx, 'X', color='red', ha='center', va='center', fontweight='bold', fontsize=12)
    rect = patches.Rectangle((worker_idx - 0.5, task_idx - 0.5), 1, 1, linewidth=2.5, edgecolor='red', facecolor='none')
    ax3.add_patch(rect)

ax3.set_title("3. Najlepszyprzydziały (X)", fontsize=14, pad=25)
ax3.set_xticks(range(N))
ax3.set_yticks(range(N))
ax3.set_xlabel("Wykonawcy (j)")
ax3.set_ylabel("Zadania (i)")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  #Dostosowanie układu, aby tytuły się nie nakładały
plt.show()