import pygame
import torch

# Inicializar Pygame
pygame.init()

# Definir constantes
SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 1000
INITIAL_HEALTHY = 900  # Número inicial de personas sanas
INITIAL_INFECTED = 100  # Número inicial de personas infectadas
MAX_PEOPLE = 10000
STEP_STD = 5
REPRODUCTION_PROB = 0.01
DEATH_PROB = 0.01
INFECTION_RADIUS = 30
INFECTION_PROB = 0.02  # Probabilidad de contagio

# Configuración de la pantalla
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Simulación de Contagio - PyTorch y Pygame")

# Crear tensor de posiciones iniciales (en la GPU)
total_people = INITIAL_HEALTHY + INITIAL_INFECTED
positions = torch.rand((total_people, 2), device="cuda") * torch.tensor([SCREEN_WIDTH, SCREEN_HEIGHT], device="cuda")
states = torch.zeros(total_people, device="cuda", dtype=torch.int32)  # 0 para sano, 1 para enfermo

# Marcar los primeros `INITIAL_INFECTED` como enfermos
states[:INITIAL_INFECTED] = 1

# Bucle principal
running = True
clock = pygame.time.Clock()

while running:
    # Manejar eventos
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Calcular desplazamientos aleatorios usando PyTorch (en la GPU)
    displacements = torch.normal(mean=0.0, std=STEP_STD, size=(positions.size(0), 2), device="cuda")
    positions += displacements

    # Mantener posiciones dentro de los límites de la pantalla
    positions[:, 0] = torch.clamp(positions[:, 0], 0, SCREEN_WIDTH)
    positions[:, 1] = torch.clamp(positions[:, 1], 0, SCREEN_HEIGHT)

    # Calcular área de contagio de manera eficiente en la GPU
    infected_positions = positions[states == 1]  # Posiciones de los infectados
    if infected_positions.size(0) > 0:
        # Expandir y calcular distancias en la GPU
        distances = torch.cdist(infected_positions, positions)
        infection_mask = (distances < INFECTION_RADIUS).any(dim=0)

        # Aplicar probabilidad de contagio
        infection_chance = torch.rand(positions.size(0), device="cuda") < INFECTION_PROB
        states[infection_mask & infection_chance & (states == 0)] = 1

    # Calcular muertes de los infectados
    death_mask = (torch.rand(states.size(0), device="cuda") < DEATH_PROB) & (states == 1)
    positions = positions[~death_mask]
    states = states[~death_mask]

    # Calcular reproducción de los sanos
    if positions.size(0) < MAX_PEOPLE:
        reproduction_mask = (torch.rand(positions.size(0), device="cuda") < REPRODUCTION_PROB) & (states == 0)
        new_positions = positions[reproduction_mask]
        if new_positions.size(0) > 0:
            positions = torch.cat((positions, new_positions), dim=0)
            states = torch.cat((states, torch.zeros(new_positions.size(0), device="cuda", dtype=torch.int32)), dim=0)
            if positions.size(0) > MAX_PEOPLE:
                positions = positions[:MAX_PEOPLE]
                states = states[:MAX_PEOPLE]

    # Contar el número de sanos e infectados
    num_healthy = (states == 0).sum().item()
    num_infected = (states == 1).sum().item()

    # Detener la simulación si se alcanza el número máximo de personas
    if positions.size(0) >= MAX_PEOPLE:
        print("OVERPOPULATION")
        print(f"Sanos: {num_healthy}, Infectados: {num_infected}")
        break

    # Dibujar en la pantalla
    screen.fill((0, 0, 0))
    positions_cpu = positions.to("cpu")
    states_cpu = states.to("cpu")
    for pos, state in zip(positions_cpu, states_cpu):
        color = (0, 255, 0) if state == 0 else (255, 0, 0)
        pygame.draw.circle(screen, color, (int(pos[0]), int(pos[1])), 2)

    # Actualizar la pantalla
    pygame.display.flip()
    clock.tick(60)

# Salir de Pygame
pygame.quit()
