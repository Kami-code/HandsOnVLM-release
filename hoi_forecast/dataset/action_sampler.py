import numpy as np

from hoi_forecast.dataset.epic_action import EpicAction


class ActionAnticipationSampler(object):
    def __init__(self, observation_seconds, anticipation_seconds=1.0, fps=4.0, origin_fps=60.0):
        self.observation_seconds = observation_seconds
        self.anticipation_seconds = anticipation_seconds
        self.fps = fps
        self.origin_fps = origin_fps

    def __call__(self, action: EpicAction):
        action_start_frame_idx = action.start_frame
        frame_aligned_observation_times, observation_frame_idxs = self.sample_history_frames(action_start_frame_idx)
        return frame_aligned_observation_times, observation_frame_idxs

    def sample_history_frames(self, action_start_frame_idx: int):
        action_start_time = (action_start_frame_idx - 1) / self.origin_fps  # eg: frame_start = 7000, then time_start = 7000/60 = 116.66666666666667
        num_frames = int(np.floor(self.observation_seconds * self.fps))  # self.observation_seconds is 2.5, self.fps is 4.0 then num_frames is 10

        # Calculate the time before the start of the action
        # 116.65 - 1.0 = 115.65s
        anticipation_time = action_start_time - self.anticipation_seconds

        # Generate the uniform time points for the frames before the time_ant
        # eg: [-2.25, -2, -1.75, ..., -0.25] + 115.65
        observation_times = (np.arange(1, num_frames + 1) - num_frames) / self.fps + anticipation_time
        observation_times = np.clip(observation_times, 0, np.inf).astype(np.float32)

        # Convert the times to frame indexes
        observation_frame_idxs = np.floor(observation_times * self.origin_fps).astype(np.int32) + 1

        # Recalculate the times based on the frame indexes
        frame_aligned_observation_times = (observation_frame_idxs - 1) / self.origin_fps
        return frame_aligned_observation_times, observation_frame_idxs