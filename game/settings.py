import argparse

parser = argparse.ArgumentParser(description="Demo")
parser.add_argument('--rl', type=str, default="False", help='RL Mode')
args = parser.parse_args()
args.rl = bool(eval(args.rl))
# Mode
RL_MODE = args.rl

# Physics Engine
DENSITY = 1  # wo assume that all fruits' density are the same
DT = 1 / 60  # the time of one step
STEPS_IN_COLLISION = 5  # the physic steps of one logic frame
PHYSICS_STEPS_PER_FRAME = 100 if RL_MODE else 1
GRAVITY = (0, 1000)  # horizontal and vertical acceleration of gravity
SIZE_TO_RADIUS = 16  # the ball with the size x will have the radius of x*SIZE_TO_RADIUS
ELASTICITY = 0.2
FRICTION = 0.9
LINE_COLLISION_TYPE = 0
BALL_COLLISION_TYPE = 1

# Balls
GRAPE = 0
CHERRY = 1
ORANGE = 2
LEMON = 3
YANGTAO = 4
TOMATO = 5
PEACH = 6
PINEAPPLE = 7
COCOUNT = 8
HALF_WATERMELON = 9
WATERMELON = 10

# Game
# control the balls that can be selected
MIN_BALL_TYPE, MAX_BALL_TYPE = GRAPE, YANGTAO
PROB_DIST = [0.3, 0.3, 0.2, 0.1, 0.1]
TIME_TO_NEXT_BALL = 0 if RL_MODE else 1
WIDTH = 450
HEIGHT = 750
DEADLINE = 80
BAULK_POINT = (WIDTH // 2, DEADLINE + 1)
FPS = 1000 if RL_MODE else 100
PLAY_SOUND = False if RL_MODE else True
SCREENSHOT = True if RL_MODE else False

# Env
NUM_ACTIONS = 32

# Style
BACKGROUND = "#FFE89D"
BALLS = {
    GRAPE: {
        "color": "#700F6E",
        "score": 1,
        "size": 1,
    },
    CHERRY: {
        "color": "#F90D23",
        "score": 2,
        "size": 1.5
    },
    ORANGE: {
        "color": "#F17709",
        "score": 3,
        "size": 2.5
    },
    LEMON: {
        "color": "#FFE81D",
        "score": 4,
        "size": 2.5
    },
    YANGTAO: {
        "color": "#5FDF1C",
        "score": 5,
        "size": 3
    },
    TOMATO: {
        "color": "#CE3439",
        "score": 6,
        "size": 4
    },
    PEACH: {
        "color": "#FB9058",
        "score": 7,
        "size": 4
    },
    PINEAPPLE: {
        "color": "#FEE043",
        "score": 8,
        "size": 4.5
    },
    COCOUNT: {
        "color": "#FCFBE9",
        "score": 9,
        "size": 5
    },
    HALF_WATERMELON: {
        "color": "#FC7B97",
        "score": 10,
        "size": 6
    },
    WATERMELON: {
        "color": "#48B036",
        "score": 15,
        "size": 7
    }
}
