import random
import sys
from functools import wraps

import numpy as np
import pygame
import pymunk
import pymunk.pygame_util

from game.settings import *


class Game(object):

    def __init__(self) -> None:
        # physics
        self._space = pymunk.Space()
        self._space.collision_persistence = STEPS_IN_COLLISION
        self._space.gravity = GRAVITY

        self._load_balls_style()

        self.score = 0
        self.frame = 0
        self._cur_ball = None

        # user event
        self._dead_event = pygame.USEREVENT + 1
        self._next_event = pygame.USEREVENT + 2

        # flags
        self._running = True  # control gameover
        self._waiting = False  # does exist a deadline?
        self._next_ball = True  # should we create the next ball

        # pygame
        pygame.init()
        self._screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self._font = pygame.font.SysFont('宋体', 32, True)
        pygame.display.set_caption('env')
        self._clear_screen()
        self.clock = pygame.time.Clock()
        if PLAY_SOUND:
            self._load_sound()

        # static barrier walls (lines) that the balls bounce off of
        self._add_static_scenery()

        # balls that exist in the world
        self._balls = []

        # control the collision between balls
        self._add_coll_handler()

        # agent action space
        self.action_space = np.linspace(0, WIDTH, NUM_ACTIONS)

    def calculate_reward(f):
        @wraps(f)
        def decorated(self, *args, **kwargs):
            score = self.score
            f(self, *args, **kwargs)
            return self.score - score
        return decorated

    def reset(self):
        self._space = pymunk.Space()
        self._space.collision_persistence = STEPS_IN_COLLISION
        self._space.gravity = GRAVITY

        self._add_static_scenery()

        self._running = True
        self._waiting = False
        self._next_ball = True

        self.score = 0
        self.frame = 0
        self._cur_ball = None
        self._balls = []
        self._add_coll_handler()

    @staticmethod
    def seed(seed):
        random.seed(seed)
        np.random.seed(seed)

    def _load_sound(self):
        self._fall_sound = pygame.mixer.Sound("./res/sound/fall.mp3")
        self._merge_sound = pygame.mixer.Sound("./res/sound/merge.mp3")

    def _load_balls_style(self):
        self._size_map = []
        self._color_map = []
        self._score_map = []
        for _, ball in BALLS.items():
            self._size_map.append(int(ball['size'] * SIZE_TO_RADIUS))
            self._color_map.append(ball['color'])
            self._score_map.append(ball['score'])

    @staticmethod
    def _genereate_random_ball():
        cands = list(range(MIN_BALL_TYPE, MAX_BALL_TYPE + 1))
        return np.random.choice(cands, size=1, p=PROB_DIST)[0]

    @staticmethod
    def _next_id(bid):
        if bid >= WATERMELON:
            return bid
        return bid + 1

    @staticmethod
    def _check_same_type(arbiter):
        shape_1, shape_2 = arbiter.shapes
        return shape_1.id == shape_2.id

    @staticmethod
    def _has_watermelon(arbiter):
        for shape in arbiter.shapes:
            if shape.id == WATERMELON:
                return True
        return False

    def _add_ball(self, shape):
        self._balls.append(shape)
        self._space.add(shape, shape.body)

    def _remove_ball(self, shape):
        self._space.remove(shape, shape.body)
        self._balls.remove(shape)

    def _remove_coll_balls(self, arbiter):
        """remove the balls involved in collision

        Note:
            In one physic step, maybe one ball will hit many balls and 
            pymunk will process these events one by one, so we can just remove
            the ball at first physic step.

        Args:
            arbiter (pymunk.Aribiter): The Arbiter object encapsulates a pair
            of colliding shapes and all of the data about their collision.

        Return:
            bool: whether there are more than one collisiion events in the physic step.

        """
        for shape in arbiter.shapes:
            self._space.remove(shape, shape.body)
            try:
                self._balls.remove(shape)
            except Exception:
                return False
        return True

    @staticmethod
    def _calculate_merge_position(arbiter):
        shape_1, shape_2 = arbiter.shapes
        pos_1 = shape_1.body.position
        pos_2 = shape_2.body.position
        return (pos_1 + pos_2) // 2

    def _add_coll_handler(self) -> None:
        def coll_begin(arbiter, space, data):
            if self._has_watermelon(arbiter):
                return True
            if self._check_same_type(arbiter):
                pos = self._calculate_merge_position(arbiter)
                if self._remove_coll_balls(arbiter):
                    bid = arbiter.shapes[0].id + 1
                    self._create_ball(bid=bid, pos=pos)
                if PLAY_SOUND:
                    self._merge_sound.play()
                return False
            return True

        handler = self._space.add_collision_handler(
            BALL_COLLISION_TYPE, BALL_COLLISION_TYPE)
        handler.begin = coll_begin

    def _check_deadline(self):
        """is there any ball reaches the deadline

        Returns:
            bool: the bool value of any ball reaches the deadline
        """

        # there is just one ball
        if len(self._balls) <= 1:
            return False

        # except for the last one which maybe the ball that was just created
        pos_heights = [ball.body.position[1] -
                       ball.radius for ball in self._balls[:-1]]

        # upper bound leads to gameover
        if min(pos_heights) <= 0:
            self._running = False

        # whem there is at least one ball reach the deadline,
        # we should set a timer to listen
        if min(pos_heights) < DEADLINE:
            if not self._waiting and not RL_MODE:
                pygame.time.set_timer(self._dead_event, 3000)
                self._waiting = True
            return True
        return False

    def _check_gameover(self):
        if self._check_deadline():
            self._running = False
        else:
            self._waiting = False

    def _update(self, screenshot=False):
        # pygame
        self._clear_screen()
        self._draw_objects()
        pygame.display.flip()
        obs = None
        if screenshot:
            obs = pygame.surfarray.array3d(self._screen)
            obs = np.mean(obs, axis=-1, keepdims=True).transpose(2, 0, 1)
            self.frame += 1
            # pygame.image.save(self._screen, f"./data/{self.frame}.png")
        return obs

    def _space_step(self):
        for _ in range(PHYSICS_STEPS_PER_FRAME):
            self._space.step(DT)

    @calculate_reward
    def _step(self, action):
        if PLAY_SOUND:
            self._fall_sound.play()
        self._create_ball(pos=(action, BAULK_POINT[1]))
        self._space_step()

    def step(self, action):
        # move one step and calculate reward
        reward = self._step(action)

        # update screen
        self._draw_static_ball(self._cur_ball)
        obs = self._update(screenshot=SCREENSHOT)

        # create new ball
        bid = self._genereate_random_ball()
        self._cur_ball = bid

        # check game over
        done = self._check_deadline()
        return obs, [bid], reward, done

    def start_obs(self):
        bid = self._genereate_random_ball()
        self._cur_ball = bid
        obs = self._update(screenshot=SCREENSHOT)
        return obs, [bid]

    def _process_clicked(self):
        action = None
        if self._next_ball:  # Are we allowed to create a ball
            action, _ = pygame.mouse.get_pos()
            if TIME_TO_NEXT_BALL:  # we will control the intervel only TIME_TO_NEXT_BALL > 0
                pygame.time.set_timer(
                    self._next_event, TIME_TO_NEXT_BALL * 1000)
                self._next_ball = False
        return action

    def run(self) -> None:
        self.start_obs()
        action = None
        while self._running:
            # Progress time forward
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    action = self._process_clicked()
                elif event.type == self._dead_event:  # deadline timer is over
                    self._check_gameover()
                elif event.type == self._next_event:  # next ball event is over
                    self._next_ball = True
                elif event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
            if action is not None:
                self.step(action)
                action = None
            else:
                self._space_step()
                self._update(screenshot=False)
            self.clock.tick(FPS)
        for shape in self._balls:
            self.score += (shape.id + 1)
        print(f'game over with score: {self.score}')

    def _add_static_scenery(self) -> None:
        static_body = self._space.static_body
        static_lines = [
            # LEFT
            pymunk.Segment(static_body, (0, 0),
                           (0, HEIGHT), 0.0),
            # RIGHT
            pymunk.Segment(static_body, (WIDTH, 0),
                           (WIDTH, HEIGHT), 0.0),
            # DOWN
            pymunk.Segment(static_body, (0, HEIGHT - 10),
                           (WIDTH, HEIGHT - 10), 0.0),
        ]
        for line in static_lines:
            line.elasticity = ELASTICITY
            line.friction = FRICTION
            line.collision_type = LINE_COLLISION_TYPE
        self._space.add(*static_lines)

    @staticmethod
    def _calculate_mass(radius):
        return DENSITY * (4 / 3) * np.pi * radius ** 3

    def _create_ball(self, bid=None, pos=(0, 0)) -> None:
        # create body
        bid = bid if bid else self._cur_ball
        radius = self._size_map[bid]
        mass = self._calculate_mass(radius)
        inertia = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, inertia)
        body.position = pos

        # create shape
        shape = pymunk.Circle(body, radius, (0, 0))
        shape.id = bid
        shape.elasticity = ELASTICITY
        shape.friction = FRICTION
        shape.collision_type = BALL_COLLISION_TYPE
        shape.color = pygame.Color(self._color_map[bid])

        self.score += self._score_map[bid]
        self._add_ball(shape)

    def _draw_static_ball(self, bid):
        color = pygame.Color(self._color_map[bid])
        pos = BAULK_POINT
        radius = self._size_map[bid]
        pygame.draw.circle(self._screen, color,
                           pos, radius, radius)

    def _clear_screen(self) -> None:
        self._screen.fill(pygame.Color(BACKGROUND))

    def _draw_objects(self) -> None:
        for ball in self._balls:
            pygame.draw.circle(self._screen, ball.color,
                               ball.body.position, ball.radius, int(ball.radius))
        if not RL_MODE:
            self._draw_static_ball(self._cur_ball)
            pygame.draw.aaline(self._screen, "RED", (0, 80), (20, 80))
            self._screen.blit(self._font.render(u'%d' %
                                                self.score, True, [255, 0, 0]), [20, 20])
