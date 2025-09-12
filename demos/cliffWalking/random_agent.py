import cv2
import gymnasium as gym
import numpy as np

# Create the Environment
cliffEnv = gym.make("CliffWalking-v1", render_mode="ansi")  # render_mode added to avoid warnings

# Handy functions for visuals
def initialize_frame():
    width, height = 600, 200
    img = np.ones(shape=(height, width, 3), dtype=np.uint8) * 255  # Use uint8 for proper image data

    margin_horizontal = 6
    margin_vertical = 2

    # Vertical Lines
    for i in range(13):
        img = cv2.line(img, (49 * i + margin_horizontal, margin_vertical),
                       (49 * i + margin_horizontal, 200 - margin_vertical), color=(0, 0, 0), thickness=1)

    # Horizontal Lines
    for i in range(5):
        img = cv2.line(img, (margin_horizontal, 49 * i + margin_vertical),
                       (600 - margin_horizontal, 49 * i + margin_vertical), color=(0, 0, 0), thickness=1)

    # Cliff Box
    img = cv2.rectangle(img, (49 * 1 + margin_horizontal + 2, 49 * 3 + margin_vertical + 2),
                        (49 * 11 + margin_horizontal - 2, 49 * 4 + margin_vertical - 2), color=(255, 0, 255),
                        thickness=-1)
    img = cv2.putText(img, text="Cliff", org=(49 * 5 + margin_horizontal, 49 * 4 + margin_vertical - 10),
                      fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255, 255, 255), thickness=2)

    # Goal
    frame = cv2.putText(img, text="G", org=(49 * 11 + margin_horizontal + 10, 49 * 4 + margin_vertical - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)

    return frame


def put_agent(img, state):
    margin_horizontal = 6
    margin_vertical = 2
    row, column = np.unravel_index(indices=state, shape=(4, 12))
    cv2.putText(img, text="A", org=(49 * column + margin_horizontal + 10, 49 * (row + 1) + margin_vertical - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)  # Agent in red
    return img


# Initialize environment
done = False
frame = initialize_frame()
state, _ = cliffEnv.reset() 

while not done:
    # Show the current state of the environment
    frame2 = put_agent(frame.copy(), state)
    cv2.imshow("Cliff Walking", frame2)
    key = cv2.waitKey(250)

    # Select a random action
    action = np.random.randint(0, 4)

    # Take the action in the environment
    state, reward, terminated, truncated, info = cliffEnv.step(action)
    done = terminated or truncated

cliffEnv.close()
cv2.destroyAllWindows()
