from __future__ import annotations

import numpy as np

from cartpole_idk.storage import Trajectory


def replay_trajectory(trajectory: Trajectory, *, fps: float = 50.0) -> None:
    """Replay recorded true CartPole states using pygame."""
    import pygame

    width, height = 600, 400
    scale = width / 4.8
    cart_y, cart_w, cart_h, pole_len = 300, 60, 30, 120
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(trajectory.trajectory_id)
    clock = pygame.time.Clock()

    for obs in trajectory.true_observations:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        x, _, theta, _ = obs
        cart_x = int(width / 2 + x * scale)
        screen.fill((255, 255, 255))
        pygame.draw.line(
            screen, (0, 0, 0), (0, cart_y + cart_h // 2), (width, cart_y + cart_h // 2), 2
        )
        pygame.draw.rect(
            screen,
            (30, 30, 30),
            pygame.Rect(cart_x - cart_w // 2, cart_y - cart_h // 2, cart_w, cart_h),
        )
        pivot = np.array([cart_x, cart_y - cart_h // 2], dtype=float)
        tip = pivot + np.array([pole_len * np.sin(theta), -pole_len * np.cos(theta)])
        pygame.draw.line(screen, (180, 60, 60), pivot, tip, 8)
        pygame.draw.circle(screen, (20, 20, 20), pivot.astype(int), 6)
        pygame.display.flip()
        clock.tick(fps)
    pygame.quit()
