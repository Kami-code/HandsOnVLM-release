class EpicAction(object):
    def __init__(self, uid, participant_id, video_id, verb, verb_class,
                 noun, noun_class, all_nouns, all_noun_classes, start_frame,
                 stop_frame, start_time, stop_time, ori_fps, partition, action, action_class, narration):
        self.uid = uid
        self.participant_id = participant_id
        self.video_id = video_id
        self.verb = verb
        self.verb_class = verb_class
        self.noun = noun
        self.noun_class = noun_class
        self.all_nouns = all_nouns
        self.all_noun_classes = all_noun_classes
        self.start_frame = start_frame
        self.stop_frame = stop_frame
        self.start_time = start_time
        self.stop_time = stop_time
        self.ori_fps = ori_fps
        self.partition = partition
        self.action = action
        self.action_class = action_class
        self.narration = narration
        self.duration = self.stop_time - self.start_time

        self.dict = {
            'uid': self.uid,
            'participant_id': self.participant_id,
            'video_id': self.video_id,
            'verb': self.verb,
            'verb_class': self.verb_class,
            'noun': self.noun,
            'noun_class': self.noun_class,
            'all_nouns': self.all_nouns,
            'all_noun_classes': self.all_noun_classes,
            'start_frame': self.start_frame,
            'stop_frame': self.stop_frame,
            'start_time': self.start_time,
            'stop_time': self.stop_time,
            'ori_fps': self.ori_fps,
            'partition': self.partition,
            'action': self.action,
            'action_class': self.action_class,
            'narration': self.narration,
            'duration': self.duration,
        }

    def __repr__(self):
        return self.__dict__

    def set_previous_actions(self, actions):
        self.actions_prev = actions