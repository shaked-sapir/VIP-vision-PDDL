from typing import List, Callable, Optional

from ..types.image import Image
import gym
from gym.utils.save_video import save_video as gym_save_video
import cv2
import os


class VideoHandler:
    supported_extensions = [".mp4"] #TODO: in the future this list may be expanded

    def _validate_video(self, video_path: str) -> bool:
        self._validate_extension(video_path)

    def _validate_extension(self, video_path: str) -> bool:
        return any(ext in video_path for ext in self.supported_extensions)

    def extract_frames_from_video(self, video_path: str, frame_rate: int=1) -> List[Image]:
        #TODO: test this one on the photoshop course video...
        """
        this function extracts images from a video, in a pre-determined fps to represent the video with proper images
        containing much data to preserve video's information.
        :param video_path: the path to the video file.
        :param frame_rate: the number of frames per second to capture
        """
        self._validate_video(video_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return []

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / frame_rate)

        frames = []
        frame_count = 0

        while True:
            # Read frame-by-frame
            ret, frame = cap.read()

            if not ret:
                break

            # Check if the current frame is one we want to capture
            if frame_count % frame_interval == 0:
                frames.append(frame)

            frame_count += 1

        # Release the video capture object
        cap.release()
        cv2.destroyAllWindows()

        return frames

    @staticmethod
    def save_video(
        frames: list, # this should be of any 'renderable' type
        video_folder: str,
        episode_trigger: Callable[[int], bool] = None,
        step_trigger: Callable[[int], bool] = None,
        video_length: Optional[int] = None,
        name_prefix: str = "VIP-video",
        episode_index: int = 0,
        step_starting_index: int = 0,
        **kwargs,
    ):
        gym_save_video(frames, video_folder, episode_trigger, step_trigger, video_length, name_prefix, episode_index, step_starting_index, **kwargs)
