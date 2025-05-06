#!/usr/bin/env python3
import random
import time
import argparse
import numpy as np
from mpi4py import MPI

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Simulación paralela Nagel–Schreckenberg con MPI + nivelación de cargas"
    )
    parser.add_argument("--length",    type=int,   default=500000,
                        help="Longitud total de la carretera")
    parser.add_argument("--density",   type=float, default=0.5,
                        help="Densidad de vehículos")
    parser.add_argument("--max-speed", type=int,   default=10,
                        help="Velocidad máxima de un vehículo")
    parser.add_argument("--p-slow",    type=float, default=0.3,
                        help="Probabilidad de frenado aleatorio")
    parser.add_argument("--steps",     type=int,   default=500,
                        help="Número de pasos de simulación")
    return parser.parse_args()

def nivelacion_cargas(D, n_p):
    """
    Divide la lista D en n_p sublistas casi iguales,
    distribuyendo los restos uno a uno a los primeros procesos.
    """
    s = len(D) % n_p
    t = (len(D) - s) // n_p
    out = []
    for i in range(n_p):
        if i < s:
            l_i = i * t + i
            l_s = l_i + t + 1
        else:
            l_i = i * t + s
            l_s = l_i + t
        out.append(D[l_i:l_s])
    return out

def initialize_local_road(local_len, density, max_speed):
    """Inicializa la porción local de la carretera."""
    return [
        -1 if random.random() > density else random.randint(0, max_speed)
        for _ in range(local_len)
    ]

def parallel_update(local, max_speed, p_slow, left_ghost, right_ghost):
    """
    Aplica el paso de actualización en el bloque local,
    usando los valores de ghost cells recibidos de los vecinos.
    """
    L_loc = len(local)
    extended = [left_ghost] + local + [right_ghost]
    new_local = [-1] * L_loc
    N = len(extended)

    for i in range(1, N - 1):
        v = extended[i]
        if v == -1:
            continue

        # 1) Distancia al siguiente vehículo
        dist = 1
        while extended[(i + dist) % N] == -1:
            dist += 1

        # 2) Aceleración
        if v < max_speed and dist > v:
            v += 1

        # 3) Frenado aleatorio
        if v > 0 and random.random() < p_slow:
            v -= 1

        # 4) Movimiento (acotado al bloque local)
        new_pos = min(i + v, N - 2)
        new_local[new_pos - 1] = v

    return new_local

def simulate_parallel(length, density, max_speed, p_slow, num_steps):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 1) Lista de índices y partición equilibrada
    all_indices = list(range(length))
    partitions = nivelacion_cargas(all_indices, size)
    local_indices = partitions[rank]
    local_len = len(local_indices)

    # 2) Inicializar semilla y carretera local
    random.seed(rank + int(time.time()))
    local_road = initialize_local_road(local_len, density, max_speed)

    # 3) Vecinos en anillo
    left  = (rank - 1) % size
    right = (rank + 1) % size

    # 4) Bucle de simulación
    for _ in range(num_steps):
        # Preparar buffers NumPy
        send_left  = np.array([local_road[0]], dtype='i')
        recv_left  = np.empty(1,        dtype='i')
        send_right = np.array([local_road[-1]], dtype='i')
        recv_right = np.empty(1,              dtype='i')

        # Intercambio con vecino izquierdo
        comm.Sendrecv(
            sendbuf=send_left,  dest=left,  sendtag=0,
            recvbuf=recv_left, source=left, recvtag=0
        )
        # Intercambio con vecino derecho
        comm.Sendrecv(
            sendbuf=send_right, dest=right, sendtag=1,
            recvbuf=recv_right, source=right, recvtag=1
        )

        left_ghost  = int(recv_left[0])
        right_ghost = int(recv_right[0])

        # Actualizar bloque local
        local_road = parallel_update(
            local_road,
            max_speed,
            p_slow,
            left_ghost,
            right_ghost
        )

    # (Opcional) Recolección en root
    # final_blocks = comm.gather(local_road, root=0)
    # if rank == 0:
    #     road_full = [cell for block in final_blocks for cell in block]
    #     print("Estado final:", road_full)

if __name__ == "__main__":
    args = parse_arguments()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Sincronizar antes de medir
    comm.Barrier()
    t_start = time.perf_counter()

    simulate_parallel(
        length=args.length,
        density=args.density,
        max_speed=args.max_speed,
        p_slow=args.p_slow,
        num_steps=args.steps
    )

    # Sincronizar tras la simulación
    comm.Barrier()
    t_parallel = time.perf_counter() - t_start

    if rank == 0:
        print(f"[Paralelo] Procesos: {size}  Tiempo de ejecución Tₚ = {t_parallel:.3f} s")
