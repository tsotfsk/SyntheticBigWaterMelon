# Python imports
import random
import json
from easydict import EasyDict as edict
from typing import List
import numpy as np

# Library imports
import pygame

# pymunk imports
import pymunk
import pymunk.pygame_util


class Game(object):

    def __init__(self, args) -> None:
        # Space
        self._space = pymunk.Space()
        self._space.gravity = (0.0, 900.0)

        # Physics
        # Time step
        self._dt = 1 / 60
        self._width, self._height = (450, 750)
        # Number of physics steps per screen frame
        self._physics_steps_per_frame = 100
        self._fps = 100


        self._mass = 10
        self._score = 0
        self._frame = 0
        self.num_actions = 32
        self.action_space = np.linspace(0, 450, self.num_actions)
        self.frame_shape = (450, 750, 3)
        self._load_args(args)
        self._play_sound = False

        # 游戏是否结束
        self._timer = True
        self._gameover = False

        self._baulk_point = (self._width // 2, max(self._size_map[:6]))

        # pygame
        pygame.init()
        self._screen = pygame.display.set_mode((self._width, self._height))
        self._screen.fill(self._color)
        self._clock = pygame.time.Clock()
        self._load_sound()

        # Static barrier walls (lines) that the balls bounce off of
        self._add_static_scenery()

        # Balls that exist in the world
        self._balls: List[pymunk.Circle] = []

        # Execution control and time until the next ball spawns
        self._running = True
        self._add_coll_handler()

    @staticmethod
    def seed(seed):
        random.seed(seed)
        np.random.seed(seed)

    def _load_sound(self):
        self._fall_sound = pygame.mixer.Sound("./game/fall.mp3")
        self._merge_sound = pygame.mixer.Sound("./game/merge.mp3")

    def _load_args(self, args):
        self._width = args.screen.width
        self._height = args.screen.height
        self._color = args.screen.color
        self._size_map = []
        self._color_map = []
        for idx, (ball_name, ball) in enumerate(args.balls.items()):
            self._size_map.append(int(ball.size * 16))
            self._color_map.append(ball.color)

    @staticmethod
    def _genereate_random_ball():
        return np.random.choice(5, size=1, p=[0.3, 0.3, 0.2, 0.1, 0.1])[0]

    def _add_id(self, ball_id):
        if ball_id + 1 >= len(self._size_map):
            return ball_id
        return ball_id + 1

    def _add_coll_handler(self) -> None:
        def coll_begin(arbiter, space, data):
            cirle_1, cirle_2 = arbiter.shapes
            if cirle_1.id == cirle_2.id and cirle_1.id < len(self._size_map) - 1:
                if cirle_1.body._get_type() == cirle_2.body._get_type():
                    pos_1 = cirle_1.body.position
                    pos_2 = cirle_2.body.position
                    pos = (pos_1 + pos_2) // 2
                    ball_id = self._add_id(cirle_1.id)
                    for shape in arbiter.shapes:
                        try:
                            self._balls.remove(shape)
                            self._space.remove(shape, shape.body)
                        except Exception:
                            return True
                    self._create_ball(ball_id=ball_id, pos=pos,
                                      ball_type=pymunk.Body.DYNAMIC, idx=0)
                    self._score += (ball_id + 1)
                    # self._merge_sound.play()
            else:
                cirle_1.body._set_velocity = (0, 0)

            return True

        handler = self._space.add_default_collision_handler()
        handler.begin = coll_begin

    def _check_gameover(self):
        if len(self._balls) <= 1:
            return
        pos_heights = [ball.body.position[1] -
                       ball.radius for ball in self._balls[:-1]]
        if min(pos_heights) < 80:
            self._running = False

    def _update(self, save=True):
        # pygame
        self._clear_screen()
        self._draw_objects()
        pygame.display.flip()
        if save:
            obs = pygame.surfarray.array3d(self._screen)
            obs = np.mean(obs, axis=-1, keepdims=True).transpose(2, 0, 1)
            # pygame.image.save(self._screen, f"./data/{self._frame}.png")
            return obs
        return False

    def _step(self):
        frames = self._physics_steps_per_frame
        for _ in range(frames):
            self._space.step(self._dt)

    def step(self, action):
        old_score = self._score
        ball_id = self._genereate_random_ball()
        self._free_static_ball((action, self._baulk_point[1] + 100))
        self._step()
        obs = self._update()
        reward = self._score - old_score
        # print(self._frame, action, reward, self._score)
        self._frame += 1
        self._create_ball(ball_id=ball_id)
        self._check_gameover()
        done = not self._running
        # self._clock.tick(self._fps)
        return obs, [ball_id], reward, done

    def start_obs(self):
        ball_id = self._genereate_random_ball()
        self._create_ball(ball_id=ball_id)
        obs = self._update()
        return obs, [ball_id]

    def run(self) -> None:
        """
        The main loop of the game.
        :return: None
        """
        # Main loop
        self.start_obs()
        action = None
        while self._running:
            # Progress time forward
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    action, _ = pygame.mouse.get_pos()
            # action = random.randint(10, 400)
            if action is not None:
                self.step(action)
                action = None
            else:
                self._step()
                self._update(save=False)
            # Delay fixed time between frames
            # pygame.display.set_caption("fps: " + str(self._clock.get_fps()))
            self._clock.tick(self._fps)
        for shape in self._balls:
            self._score += (shape.id + 1)
        print(f'game over with score: {self._score}')

    def _add_static_scenery(self) -> None:
        """
        Create the static bodies.
        :return: None
        """
        width, height = self._width, self._height
        static_body = self._space.static_body
        static_lines = [
            pymunk.Segment(static_body, (0, 0),
                           (0, height), 0.0),
            pymunk.Segment(static_body, (width, 0),
                           (width, height), 0.0),
            pymunk.Segment(static_body, (0, height - 10),
                           (width, height - 10), 0.0),
        ]
        for line in static_lines:
            line.elasticity = 0.9
            line.friction = 0.9
            line.id = -2
        self._space.add(*static_lines)

    # def _process_events(self, action=None) -> None:
    #     if len(self._balls):
    #         self._free_static_ball((action, self._baulk_point[1] + 100))
    #         self._fall_sound.play()
    #     ball_id = self._genereate_random_ball()
    #     self._create_ball(ball_id=ball_id)

    def _free_static_ball(self, pos):
        ball = self._balls[-1]
        ball.body._set_type(pymunk.Body.DYNAMIC)
        radius = ball.radius
        inertia = pymunk.moment_for_circle(self._mass, 0, radius, (0, 0))
        ball.body._set_mass(self._mass)
        ball.body._set_moment(inertia)
        ball.body.position = pos
        # self._fall_sound.play()
        self._score += (ball.id + 1)

    def _create_ball(self, ball_id=None, pos=None, ball_type=pymunk.Body.STATIC, idx=None) -> None:
        radius = self._size_map[ball_id]
        mass = 1
        inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        body = pymunk.Body(mass, inertia, ball_type)
        if pos is None:
            body.position = self._baulk_point
        else:
            body.position = pos
        shape = pymunk.Circle(body, radius, (0, 0))
        shape.id = ball_id
        shape.elasticity = 0.2
        shape.friction = 0.9
        shape.color = pygame.Color(self._color_map[ball_id])
        # shape.image
        self._space.add(body, shape)
        if idx is None:
            self._balls.append(shape)
        else:
            self._balls.insert(idx, shape)

    def _clear_screen(self) -> None:
        """
        Clears the screen.
        :return: None
        """
        self._screen.fill(pygame.Color(self._color))

    def _draw_objects(self) -> None:
        """
        Draw the objects.
        :return: None
        """
        for ball in self._balls:
            pygame.draw.circle(self._screen, ball.color,
                               ball.body.position, ball.radius, int(ball.radius))
        # pygame.draw.aaline(self._screen, "BLUE", (0, 100), (450, 100), 1)


if __name__ == "__main__":
    import os
    for root, dirs, files in os.walk('./'):
        for name in files:
            if '.png' in name:  # 判断某一字符串是否具有某一字串，可以使用in语句
                os.remove(os.path.join(root, name))  # os.move语句为删除文件语句

    with open("./game/style.json", 'r') as f:
        args = edict(json.load(f))
        game = Game(args)
        game.run()
