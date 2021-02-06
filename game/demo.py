"""This example spawns (bouncing) balls randomly on a L-shape constructed of 
two segment shapes. Not interactive.
"""

# Python imports
import random
import json
from easydict import EasyDict as edict
from typing import List
from numpy.random import choice
# from game.ball import Ball

# Library imports
import pygame

# pymunk imports
import pymunk
import pymunk.pygame_util


class Game(object):
    """
    This class implements a simple scene in which there is a static platform (made up of a couple of lines)
    that don't move. Balls appear occasionally and drop onto the platform. They bounce around.
    """

    def __init__(self, args) -> None:
        # Space
        self._space = pymunk.Space()
        self._space.gravity = (0.0, 900.0)

        # Physics
        # Time step
        self._dt = 1.0 / 60
        self._width, self._height = (450, 750)
        # Number of physics steps per screen frame
        self._physics_steps_per_frame = 1
        self._mass = 10
        self._score = 0
        self._load_args(args)

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

    def _genereate_random_ball(self):
        return choice(5, size=1, p=[0.3, 0.3, 0.2, 0.1, 0.1])[0]

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
                            pass
                    self._create_ball(ball_id=ball_id, pos=pos,
                                      ball_type=pymunk.Body.DYNAMIC, idx=0)
                    self._score += (ball_id + 1)
                    print(self._score)
                    self._merge_sound.play()
                    return True
                else:
                    return True
            return True

        handler = self._space.add_default_collision_handler()
        handler.begin = coll_begin

    def _check_gameover(self):
        if len(self._balls) <= 1:
            return
        pos_heights = [ball.body.position[1] for ball in self._balls[:-1]]
        if min(pos_heights) < self._baulk_point[1]:
            self._running = False

    def run(self) -> None:
        """
        The main loop of the game.
        :return: None
        """
        # Main loop
        ball_id = 0
        self._create_ball(ball_id=ball_id)
        while self._running:
            # Progress time forward
            for x in range(self._physics_steps_per_frame):
                self._space.step(self._dt)

            self._process_events()
            # self._update_balls()
            self._clear_screen()
            self._draw_objects()
            self._check_gameover()
            pygame.display.flip()
            # Delay fixed time between frames
            self._clock.tick(1000)
            pygame.display.set_caption("fps: " + str(self._clock.get_fps()))
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
            line.elasticity = 0.95
            line.friction = 0.9
            line.id = -2
        self._space.add(*static_lines)

    # def _process_events(self) -> None:
    #     x = random.randint(10, 400)
    #     self._free_static_ball((x, self._baulk_point[1] + 100))
    #     self._fall_sound.play()
    #     ball_id = random.randint(0, 4)
    #     self._create_ball(ball_id=ball_id)

    def _process_events(self) -> None:
        """
        Handle game and events like keyboard input. Call once per frame only.
        :return: None
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                x = x + random.random()
                self._free_static_ball((x, self._baulk_point[1] + 100))
                self._fall_sound.play()
                ball_id = self._genereate_random_ball()
                self._create_ball(ball_id=ball_id)
            # elif event.type == pygame.USEREVENT:
            #     self._check_gameover()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                pygame.image.save(self._screen, "bouncing_balls.png")

    def _free_static_ball(self, pos):
        ball = self._balls[-1]
        ball.body._set_type(pymunk.Body.DYNAMIC)
        radius = ball.radius
        inertia = pymunk.moment_for_circle(self._mass, 0, radius, (0, 0))
        ball.body._set_mass(self._mass)
        ball.body._set_moment(inertia)
        ball.body.position = pos
        self._score += (ball.id + 1)
        print(self._score)

    def _create_ball(self, ball_id=None, pos=None, ball_type=pymunk.Body.STATIC, idx=None) -> None:
        mass = 10
        radius = self._size_map[ball_id]
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
        pygame.draw.aaline(self._screen, "BLUE", (0, 100), (450, 100), 1)


if __name__ == "__main__":
    with open("./game/style.json", 'r') as f:
        args = edict(json.load(f))
    game = Game(args)
    game.run()
