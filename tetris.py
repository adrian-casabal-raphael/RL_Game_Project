import numpy as np
import gym
from gym import spaces
import cv2
from PIL import Image

class Tetris(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        super(Tetris, self).__init__()
        self.grid_height = 20
        self.grid_width = 10
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
        self.pieces = [
            np.array([[1, 1, 1, 1]]),
            np.array([[1, 1, 1], [0, 1, 0]]),
            np.array([[1, 1], [1, 1]]),
            np.array([[1, 1, 0], [0, 1, 1]]),
            np.array([[0, 1, 1], [1, 1, 0]]),
            np.array([[1, 1, 1], [1, 0, 0]]),
            np.array([[1, 1, 1], [0, 0, 1]])
        ]
        self.piece_colors = [
            (0, 0, 0),
            (255, 255, 0),
            (147, 88, 254),
            (54, 175, 144),
            (255, 0, 0),
            (102, 217, 238),
            (254, 151, 32),
            (0, 0, 255)
    ]
        self.current_piece = None
        self.current_position = None
        self.reset()

        self.action_space = spaces.Discrete(4) # left, right, rotate, down
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.grid_height, self.grid_width), dtype=np.uint8)
        self.score = 0
        self.done = False

    def reset(self):
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=int)
        self.spawn_piece()
        self.score = 0
        return self.grid

    def step(self, action):
        reward = 0
        self.apply_action(action)

        # check to see if piece has landed
        if not self.valid_position(self.current_piece, (self.current_position[0] + 1, self.current_position[1])):
            self.lock_piece()
            lines_cleared = self.clear_lines()
            reward = lines_cleared * 10
            self.score += reward
            done = self.is_game_over()
            if not done:
                self.spawn_piece()
        else:
            self.current_position = (self.current_position[0] + 1, self.current_position[1])
            done = False
        obs = self.get_observation()
        return obs, reward, done, {}

    def render(self, mode='human', video=None):
        if not self.done:
            img = [self.piece_colors[p] for row in self.get_observation() for p in row]
        else:
            img = [self.piece_colors[p] for row in self.grid for p in row]
        img = np.array(img).reshape((self.grid_height, self.grid_width, 3).astype(np.uint8))
        img = img[..., ::-1] # RGB to BGR
        img = Image.fromarray(img, "RGB")
        img = img.resize((self.grid_width * 30, self.grid_height * 30), Image.NEAREST)
        img = np.array(img)
        img[[i * 30 for i in range(self.grid_height)], :, :] = 0
        img[:, [i * 30 for i in range(self.grid_width)], :] = 0

        cv2.putText(img, "Score:", (self.grid_width * 30 + 15, 30),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=(255, 255, 255))
        cv2.putText(img, str(self.score), (self.grid_width * 30 + 15, 60),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=(255,255, 255))

        if video:
            video.write(img)

        cv2.imshow('Tetris', img)
        cv2.waitKey(1)
    def get_rendered_frame(self):
        frame = np.zeros((self.grid_height, self.grid_width, 3), dtype=np.uint8)
        frame[self.grid > 0] = [255, 255, 255]
        for i in range(self.current_piece.shape[0]):
            for j in range(self.current_piece.shape[1]):
                if self.current_piece[i, j] == 1:
                    frame[self.current_position[0] + i, self.current_position[1] + j] = [255, 0, 0]
        frame = cv2.resize(frame, (256, 512), interpolation=cv2.INTER_NEAREST)
        return frame

    def close(self):
        cv2.destroyAllWindows()
    def spawn_piece(self):
        self.current_piece = self.pieces[np.random.randint(len(self.pieces))]
        self.current_position = (0, self.grid_width // 2 - self.current_piece.shape[1] // 2)

    def valid_position(self, piece, position):
        for i, row in enumerate(piece):
            for j, cell in enumerate(row):
                if cell and (position[0] + i >= self.grid_height
                             or position[1] + j < 0
                             or position[1] + j >= self.grid_width
                             or self.grid[position[0] + i, position[1] + j]):
                    return False
        return True

    def apply_action(self, action):
        new_position = self.current_position
        if action == 0: # left action
            new_position = (self.current_position[0], self.current_position[1] - 1)
        elif action == 1: # right action
            new_position = (self.current_position[0], self.current_position[1] + 1)
        elif action == 2: # rotate action
            new_piece = np.rot90(self.current_piece)
            if self.valid_position(new_piece, self.current_position):
                self.current_piece = new_piece
            return
        elif action == 3: # down action
            while self.valid_position(self.current_piece, (self.current_position[0] + 1, self.current_position[1])):
                self.current_position = (self.current_position[0] + 1, self.current_position[1])
                return
        else:
            return

        if self.valid_position(self.current_piece, new_position):
            self.current_position = new_position

    def lock_piece(self):
        for i, row in enumerate(self.current_piece):
            for j, cell in enumerate(row):
                if cell:
                    self.grid[self.current_position[0] + i, self.current_position[1] + j] = cell

    def clear_lines(self):
        lines_cleared = 0
        for i in range(self.grid_height):
            if np.all(self.grid[i]):
                self.grid = np.delete(self.grid, i, 0)
                self.grid = np.vstack([np.zeros((1, self.grid_width)), self.grid])
                lines_cleared += 1
        return lines_cleared

    def is_game_over(self):
        return np.all(self.grid[0])

    def get_observation(self):
        obs = self.grid.copy()
        for i in range(self.current_piece.shape[0]):
            for j in range(self.current_piece.shape[1]):
                if self.current_piece[i, j] == 1:
                    obs[self.current_position[0] + i, self.current_position[1] + j] = 1
        return obs