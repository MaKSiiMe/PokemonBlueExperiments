import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
from pyboy import PyBoy
from ultralytics import YOLO
import cv2


class PokemonBlueEnv(gym.Env):
    def __init__(self, rom_path, model_path, headless=True, speed=0,
                 init_state="states/06_pallet_town.state"):
        super(PokemonBlueEnv, self).__init__()

        self.init_state = init_state

        window_type = "headless" if headless else "SDL2"
        self.pyboy = PyBoy(rom_path, window=window_type, sound=False)
        self.pyboy.set_emulation_speed(speed)

        print(f"🧠 Chargement du modèle Vision : {model_path}")
        self.yolo = YOLO(model_path)

        self.valid_actions = [
            'up', 'down', 'left', 'right', 'a', 'b', 'start', 'select', 'pass'
        ]
        self.action_space = spaces.Discrete(len(self.valid_actions))

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6,), dtype=np.float32
        )

    def step(self, action_idx):
        action = self.valid_actions[action_idx]
        if action != 'pass':
            self.pyboy.button(action)

        for _ in range(24):
            self.pyboy.tick()

        obs = self._get_observation()

        reward = 0

        terminated = False
        truncated = False

        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.init_state and os.path.exists(self.init_state):
            with open(self.init_state, "rb") as f:
                self.pyboy.load_state(f)
        else:
            print(f"⚠️ State introuvable: {self.init_state}")

        for _ in range(100):
            self.pyboy.tick()

        return self._get_observation(), {}

    def _get_observation(self):
        screen = self.pyboy.screen.image
        if screen.mode == 'RGBA':
            screen = screen.convert('RGB')
        frame = np.array(screen)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        results = self.yolo(frame, imgsz=320, verbose=False)[0]

        player_pos = [0.0, 0.0]
        door_pos = [0.0, 0.0]
        sign_pos = [0.0, 0.0]

        boxes = results.boxes.data.cpu().numpy()

        player_boxes = boxes[boxes[:, 5] == 0]
        if len(player_boxes) > 0:
            pb = player_boxes[0]
            cx = (pb[0] + pb[2]) / 2 / 160.0
            cy = (pb[1] + pb[3]) / 2 / 144.0
            player_pos = [cx, cy]

        door_boxes = boxes[boxes[:, 5] == 2]
        if len(door_boxes) > 0 and player_pos != [0.0, 0.0]:
            min_dist = 999
            for db in door_boxes:
                dcx = (db[0] + db[2]) / 2 / 160.0
                dcy = (db[1] + db[3]) / 2 / 144.0
                dist = ((dcx - player_pos[0])**2 + (dcy - player_pos[1])**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    door_pos = [dcx, dcy]

        sign_boxes = boxes[boxes[:, 5] == 3]
        if len(sign_boxes) > 0:
            sb = sign_boxes[0]
            sign_pos = [(sb[0]+sb[2])/2/160.0, (sb[1]+sb[3])/2/144.0]

        return np.array(player_pos + door_pos + sign_pos, dtype=np.float32)

    def render(self):
        pass

    def close(self):
        self.pyboy.stop()
