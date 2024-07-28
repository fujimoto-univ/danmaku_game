import cv2
import pygame
from env import WINDOW_SIZE, BulletEnv
from wrapper import GrayFrameStack, NpNewAxis, ObsResize


def gen_env():
    env = BulletEnv()
    env = GrayFrameStack(env, shape=(WINDOW_SIZE[1], WINDOW_SIZE[0]))
    env = ObsResize(env, shape=(84, 84))
    env = NpNewAxis(env)
    return env


def human_play():
    env = gen_env()
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE[0], WINDOW_SIZE[1]))
    pygame.display.set_caption("Env")
    clock = pygame.time.Clock()
    going = True
    total = 0
    while going:
        clock.tick(30)
        screen.fill((0, 0, 0))
        image = env.render()
        img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        shape = image.shape[1::-1]
        pygame_image = pygame.image.frombuffer(img.tobytes(), shape, "RGB")
        screen.blit(pygame_image, (0, 0))
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                going = False
            pressed_keys = pygame.key.get_pressed()
            if pressed_keys[pygame.K_ESCAPE]:
                going = False
            elif pressed_keys[pygame.K_UP]:
                act = 4
            elif pressed_keys[pygame.K_RIGHT]:
                act = 1
            elif pressed_keys[pygame.K_DOWN]:
                act = 2
            elif pressed_keys[pygame.K_LEFT]:
                act = 3
            else:
                act = 0
        _, rew, done, _ = env.step(act)
        total += rew
        if done:
            going = False

    print(total)
    pygame.quit()
    env.close()


def obs_check(axis):
    env = gen_env()
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE[0], WINDOW_SIZE[1]))
    pygame.display.set_caption("Env")
    clock = pygame.time.Clock()
    going = True
    total = 0
    obs = env.reset()
    while going:
        clock.tick(30)
        screen.fill((0, 0, 0))
        image = obs[axis]
        img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        shape = image.shape[1::-1]
        pygame_image = pygame.image.frombuffer(img.tobytes(), shape, "RGB")
        screen.blit(pygame_image, (0, 0))
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                going = False
            pressed_keys = pygame.key.get_pressed()
            if pressed_keys[pygame.K_ESCAPE]:
                going = False
            elif pressed_keys[pygame.K_UP]:
                act = 4
            elif pressed_keys[pygame.K_RIGHT]:
                act = 1
            elif pressed_keys[pygame.K_DOWN]:
                act = 2
            elif pressed_keys[pygame.K_LEFT]:
                act = 3
            else:
                act = 0
        obs, rew, done, _ = env.step(act)
        total += rew
        if done:
            going = False

    print(total)
    pygame.quit()
    env.close()


if __name__ == '__main__':
    human_play()
