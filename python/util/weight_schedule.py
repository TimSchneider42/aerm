from typing import Callable, Union, Sequence, Optional

import torch.nn


class WeightScheduleSegment(torch.nn.Module):
    __constants__ = ["end_point"]

    def __init__(self, end_point: float):
        super(WeightScheduleSegment, self).__init__()
        self.end_point = end_point

    def forward(self, progress: float) -> float:
        raise NotImplementedError()


class ConstantWeightScheduleSegment(WeightScheduleSegment):
    def __init__(self, end_point: float, value: float):
        super(ConstantWeightScheduleSegment, self).__init__(end_point)
        self._value = value

    def forward(self, progress: float) -> float:
        return self._value

    def __repr__(self):
        return "ConstantWeightScheduleSegment({}, {})".format(self.end_point, self._value)


class LinearWeightScheduleSegment(WeightScheduleSegment):
    def __init__(self, end_point: float, start_value: float, end_value: float):
        super(LinearWeightScheduleSegment, self).__init__(end_point)
        self._start_value = start_value
        self._end_value = end_value

    def forward(self, progress: float) -> float:
        return self._start_value + progress * (self._end_value - self._start_value)

    def __repr__(self):
        return "LinearWeightScheduleSegment({}, {}, {})".format(self.end_point, self._start_value, self._end_value)


class ExponentialWeightScheduleSegment(WeightScheduleSegment):
    def __init__(self, end_point: float, start_value: float, end_value: float):
        super(ExponentialWeightScheduleSegment, self).__init__(end_point)
        self._start_value = start_value
        self._end_value = end_value

    def forward(self, progress: float) -> float:
        return self._start_value * (self._end_value / self._start_value) ** progress

    def __repr__(self):
        return "ExponentialWeightScheduleSegment({}, {}, {})".format(
            self.end_point, self._start_value, self._end_value)


class WeightSchedule(torch.nn.Module):
    __constants__ = ["_starting_progress"]

    def __init__(self, segments: Sequence[WeightScheduleSegment], starting_progress: float):
        super(WeightSchedule, self).__init__()
        assert all(
            [s1.end_point <= s2.end_point for s1, s2 in zip(segments[:-1], segments[1:])]), "Segments are not ordered."
        self._segments = torch.nn.ModuleList(segments)
        self._starting_progress = starting_progress

    def forward(self, progress: float) -> float:
        prev_end_point = self._starting_progress
        value = 0.0
        value_set = False
        for i, s in enumerate(self._segments):
            if not value_set:
                assert hasattr(s, "end_point")
                if progress < s.end_point or i == len(self._segments) - 1:
                    r = s.end_point - prev_end_point
                    if r == 0:
                        relative_progress = 0.0
                    else:
                        relative_progress = (progress - prev_end_point) / r
                    value = s(relative_progress)
                    value_set = True
                else:
                    prev_end_point = s.end_point
        return value

    def __repr__(self):
        return list(self._segments).__repr__()


def make_segment_module(type_str: str, start_value: float, end_value: float, end_point: float) -> WeightScheduleSegment:
    if type_str == "c":
        return ConstantWeightScheduleSegment(end_point, start_value)
    elif type_str == "l":
        return LinearWeightScheduleSegment(end_point, start_value, end_value)
    elif type_str == "e":
        return ExponentialWeightScheduleSegment(end_point, start_value, end_value)
    raise ValueError("Unknown segment type \"{}\"".format(type_str))


def parse_segment(segment: str):
    try:
        return 0.0, float(segment)
    except ValueError:
        segment_split = segment.split(":")
        if len(segment_split) == 2:
            return float(segment_split[0]), float(segment_split[1])
        else:
            return segment


def parse_weight_schedule(schedule_str: str) -> Union[float, WeightSchedule]:
    try:
        return float(schedule_str)
    except ValueError:
        schedule_split = schedule_str.split(",")
        schedule_parsed = [parse_segment(s) for s in schedule_split]
        insert_indices = [
            i for i, (s1, s2) in enumerate(zip(schedule_parsed[:-1], schedule_parsed[1:]))
            if isinstance(s1, tuple) and isinstance(s2, tuple)
        ]
        for i in reversed(insert_indices):
            schedule_parsed.insert(i + 1, "c")
        schedule_mods = [
            make_segment_module(t, s, e, ep)
            for (sp, s), t, (ep, e) in zip(schedule_parsed[:-2:2], schedule_parsed[1:-1:2], schedule_parsed[2::2])
        ]
        start_point = schedule_parsed[0][0]
        start_value = schedule_parsed[0][1]
        end_point = schedule_parsed[-1][0]
        end_value = schedule_parsed[-1][1]
        schedule_mods.append(ConstantWeightScheduleSegment(end_point, end_value))
        schedule_mods.insert(0, ConstantWeightScheduleSegment(start_point, start_value))
        return WeightSchedule(schedule_mods, start_point)
