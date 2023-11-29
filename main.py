import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2


SCREEN_SHAPE = (1280, 960)


draw_coordinates = {
    'row_0': [[0.1, 0.166], [0.9, 0.166]],
    'row_1': [[0.1, 0.5], [0.9, 0.5]],
    'row_2': [[0.1, 0.833], [0.9, 0.833]],
    'col_0': [[0.166, 0.1], [0.166, 0.9]],
    'col_1': [[0.5, 0.1], [ 0.5, 0.9]],
    'col_2': [[0.833, 0.1], [0.833, 0.9]],
    'diag': [[0.1, 0.1], [0.9, 0.9]],
    'diag_2': [[0.9, 0.1], [0.1, 0.9]],
}


class TicTocGame:
    def __init__(self):
        self.fied = np.zeros((3, 3), dtype=np.uint8)
        self.player_state = 0
        self.winner = -1
        self.where = None

    def reset(self):
        self.fied = np.zeros((3, 3), dtype=np.uint8)
        self.player_state = 0
        self.winner = -1
        self.where = None

    def is_game_over(self):
        if self.winner != -1:
            return True
        if np.all(self.fied != 0):
            return True
        return False

    def step(self, image, pos):
        if self.check_if_step_available(image, pos):
            h, w, _ = image.shape
            x_coord = int(np.clip(pos[0], 0, w-1) / w * 3)
            y_coord = int(np.clip(pos[1], 0, h-1) / h * 3)

            if self.player_state == 0:
                self.fied[y_coord, x_coord] = 2
            else:
                self.fied[y_coord, x_coord] = 1
            self.player_state = abs(self.player_state - 1)

            res, where = self.check_win_combination()
            if res != 0:
                self.winner = res
                self.where = where

    def check_win_combination(self):
        for i in range(3):
            if np.all(self.fied[i, :] == 2):
                return 2, f'row_{i}'
            if np.all(self.fied[i, :] == 1):
                return 1,  f'row_{i}'
            if np.all(self.fied[:, i] == 2):
                return 2, f'col_{i}'
            if np.all(self.fied[:, i] == 1):
                return 1, f'col_{i}'
        if self.fied[0, 0] == self.fied[1, 1] == self.fied[2, 2] == 2:
            return 2, 'diag'
        if self.fied[2, 0] == self.fied[1, 1] == self.fied[0, 2] == 2:
            return 2, 'diag_2'
        if self.fied[0, 0] == self.fied[1, 1] == self.fied[2, 2] == 1:
            return 1, 'diag'
        if self.fied[2, 0] == self.fied[1, 1] == self.fied[0, 2] == 1:
            return 1, 'diag_2'
        return 0, None

    def check_if_step_available(self, image, pos):
        if self.winner != -1:
            return False
        h, w, _ = image.shape
        x_coord = int(np.clip(pos[0], 0, w - 1) / w * 3)
        y_coord = int(np.clip(pos[1], 0, h - 1) / h * 3)
        if self.fied[y_coord, x_coord] == 0:
            return True
        return False

    def draw_circle(self, image, x, y):
        h, w, _ = image.shape
        cv2.circle(
            image,
            (x, y),
            radius= min(h, w) // 8,
            color=(0, 0, 255),
            thickness=5
        )
        return image

    def draw_cross(self, image, x, y):
        h, w, _ = image.shape
        radius = min(h, w) // 8
        cv2.line(image, (x - radius, y - radius),
                 (x + radius, y + radius), color=(255, 0, 0), thickness=5)
        cv2.line(image, (x - radius, y + radius),
                 (x + radius, y - radius), color=(255, 0, 0), thickness=5)

        return image

    def draw_on_image(self, image, pos):
        h, w, _ = image.shape
        image = self.draw_grid(image)
        for i in range(3):
            for j in range(3):
                x = i * w // 3 + w // 6
                y = j * h // 3 + h // 6
                if self.fied[j, i] == 1:
                    image = self.draw_circle(image, x, y)
                elif self.fied[j, i] == 2:
                    image = self.draw_cross(image, x, y)
        if pos is not None:
            x_coord = int(np.clip(pos[0], 0, w - 1) / w * 3)
            y_coord = int(np.clip(pos[1], 0, h - 1) / h * 3)
            mask = np.zeros_like(image)
            if self.check_if_step_available(image, pos):
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            cv2.rectangle(mask, (x_coord * w // 3, y_coord * h// 3),
                          ((x_coord + 1) * w // 3, ((y_coord + 1) * h// 3)),
                          color, -1)
            image = cv2.addWeighted(image, 1, mask, 0.5, 0)

        if self.winner != -1:
            coords = draw_coordinates[self.where]
            color = (255, 0, 0) if self.winner == 2 else (0, 0, 255)
            cv2.line(image, (int(coords[0][0] * w), int(coords[0][1] * h)),
                     (int(coords[1][0] * w), int(coords[1][1] * h)),
                     color=color, thickness=15)

        return image

    def draw_grid(self, image, color=(0, 255, 0), thickness=5):
        h, w, _ = image.shape
        rows, cols = 3, 3
        dy, dx = h / rows, w / cols

        for x in np.linspace(start=dx, stop=w-dx, num=cols-1):
            x = int(round(x))
            cv2.line(image, (x, 0), (x, h), color=color, thickness=thickness)

        # draw horizontal lines
        for y in np.linspace(start=dy, stop=h-dy, num=rows-1):
            y = int(round(y))
            cv2.line(image, (0, y), (w, y), color=color, thickness=thickness)

        return image


def main():
    base_options = python.BaseOptions(
        model_asset_path='gesture_recognizer.task')
    options = vision.GestureRecognizerOptions(base_options=base_options)
    recognizer = vision.GestureRecognizer.create_from_options(options)

    cap = cv2.VideoCapture(0)
    game = TicTocGame()

    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame_shape = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format =mp.ImageFormat.SRGB, data=rgb)
        recognition_result = recognizer.recognize(image)
        if len(recognition_result.gestures):
            top_gesture = recognition_result.gestures[0][0]
            hand_landmarks = recognition_result.hand_landmarks
            x = hand_landmarks[0][9].x * frame_shape[1]
            y = hand_landmarks[0][9].y * frame_shape[0]
            pos = int(x), int(y)

            if top_gesture.category_name == 'Closed_Fist':
                game.step(frame, pos)
            frame = game.draw_on_image(frame, pos)
            if top_gesture.category_name == 'Thumb_Up' and game.is_game_over():
                game.reset()

            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y,
                                                z=landmark.z) for landmark in
                hand_landmarks[0]
            ])
            solutions.drawing_utils.draw_landmarks(
                frame,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style())
        frame = game.draw_on_image(frame, pos=None)
        frame = cv2.resize(frame, SCREEN_SHAPE)
        cv2.imshow('Game', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()


if __name__ == '__main__':
    main()
