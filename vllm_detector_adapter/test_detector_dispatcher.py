

from vllm_detector_adapter.detector_dispatcher import detector_dispatcher


@detector_dispatcher(types=["foo"])
def preprocessor(data):
    return data + 1

@detector_dispatcher(types=["bar"])
def preprocessor(data):
    return str(data) + " 1"

@detector_dispatcher(types=["foo", "bar"])
def diff_processor(data):
    return data - 1

# Error case
# @detector_dispatcher(types=["foo", "bar"])
# def diff_processor(data):
#     return data - 1

class Foo():
    def __init__(self, x):
        self.x = x

    @detector_dispatcher(types=["foo"])
    def class_processor(self, data):
        return data + 2

    @detector_dispatcher(types=["bar"])
    def class_processor(self, data):
        return str(data) + " 2"

# Testing same method names across different classes
class Foo2():
    def __init__(self, x):
        self.x = x

    @detector_dispatcher(types=["bar"])
    def class_processor(self, data):
        return data + 2

    @detector_dispatcher(types=["foo"])
    def class_processor(self, data):
        return str(data) + " 2"