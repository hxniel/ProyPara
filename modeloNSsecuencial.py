#!/usr/bin/env python3
import random
import time
import argparse

def initialize_road(length, density, max_speed):
    """Inicializa la carretera: -1 vacía o velocidad aleatoria [0, max_speed]."""
    return [
        -1 if random.random() > density else random.randint(0, max_speed)
        for _ in range(length)
    ]

def update(road, max_speed, p_slow):
    """Aplica una iteración del modelo de Nagel–Schreckenberg."""
    L = len(road)
    new_road = [-1] * L
    for i, v in enumerate(road):
        if v == -1:
            continue
        # 1) Distancia al siguiente vehículo
        dist = 1
        while road[(i + dist) % L] == -1:
            dist += 1
        # 2) Aceleración
        if v < max_speed and dist > v:
            v += 1
        # 3) Frenado aleatorio
        if v > 0 and random.random() < p_slow:
            v -= 1
        # 4) Movimiento
        new_pos = min(i + v, L - 1)
        new_road[new_pos] = v
    return new_road

def simulate_traffic(length, density, max_speed, p_slow, num_steps):
    """Corre toda la simulación secuencialmente."""
    road = initialize_road(length, density, max_speed)
    for _ in range(num_steps):
        road = update(road, max_speed, p_slow)
    return road

def main():
    parser = argparse.ArgumentParser(
        description="Simulación secuencial Nagel–Schreckenberg + métricas"
    )
    parser.add_argument("--length",    type=int,   default=500000,
                        help="Longitud de la carretera")
    parser.add_argument("--density",   type=float, default=0.5,
                        help="Densidad de vehículos")
    parser.add_argument("--max-speed", type=int,   default=10,
                        help="Velocidad máxima")
    parser.add_argument("--p-slow",    type=float, default=0.3,
                        help="Probabilidad de frenado aleatorio")
    parser.add_argument("--steps",     type=int,   default=500,
                        help="Número de pasos de simulación")
    args = parser.parse_args()

    # --- Medición de tiempo secuencial ---
    t0 = time.perf_counter()
    simulate_traffic(
        length=args.length,
        density=args.density,
        max_speed=args.max_speed,
        p_slow=args.p_slow,
        num_steps=args.steps
    )
    T1 = time.perf_counter() - t0

    # --- Métricas para la versión secuencial ---
    S1 = 1.0       # Speedup secuencial
    E1 = 1.0       # Eficiencia secuencial

    # --- Salida ---
    print(f"[Secuencial] Tiempo de ejecución T₁ = {T1:.3f} s")
    print(f"[Secuencial] Speedup   S₁ = {S1:.3f}")
    print(f"[Secuencial] Eficiencia E₁ = {E1:.3f}")

if __name__ == "__main__":
    main()
