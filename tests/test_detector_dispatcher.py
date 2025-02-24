
# Standard
import pytest

# Local
from vllm_detector_adapter.detector_dispatcher import detector_dispatcher


### Global Declarations ########################################################

@detector_dispatcher(types=["foo"])
def add(data):
    return data + 1

@detector_dispatcher(types=["bar"])
def add(data):
    return str(data) + " 1"

@detector_dispatcher(types=["foo", "bar"])
def diff(data):
    return data - 1


class Foo():
    def __init__(self, x):
        self.x = x

    @detector_dispatcher(types=["foo"])
    def add(self, data):
        return data + 2

    @detector_dispatcher(types=["bar"])
    def add(self, data):
        return str(data) + " 2"

### Tests #####################################################################

def test_detector_dispatching_to_module_functions():
    "Test that correct function gets called when each are of different type"
    assert add(1, fn_type="foo") == 2
    assert add(1, fn_type="bar") == "1 1"

def test_detector_dispatching_to_module_functions_2_types():
    "Test that correct function gets called when function is of 2 types"
    assert diff(1, fn_type="foo") == 0
    assert diff(1, fn_type="bar") == 0


def test_detector_dispatching_for_methods_works():
    "Test that correct instance method gets called when each are of different type"
    instance = Foo(1)
    assert instance.add(1, fn_type="foo") == 3
    assert instance.add(1, fn_type="bar") == "1 2"

def test_detector_dispatching_same_name_method_across_classes():
    "Test that correct instance method gets called when each are of different type"

    # Create duplicate class with same method as Foo
    class Foo2():
        def __init__(self, x):
            self.x = x

        @detector_dispatcher(types=["foo"])
        def add(self, data):
            return data + 20

    instance_1 = Foo(1)
    instance_2 = Foo2(1)

    assert instance_1.add(1, fn_type="foo") == 3
    assert instance_2.add(1, fn_type="foo") == 21

### Error Tests #############################################################

def test_decorator_erroring_with_no_type_available_fn():
    "Test that an error is raised when function to given type is not available"
    with pytest.raises(ValueError):
        add(1, fn_type="baz")


def test_decorator_error_with_no_fn_type_provided():
    "Test that an error is raised when no fn_type is provided when calling the function"
    with pytest.raises(ValueError):
        add(1)

def test_decorator_erroring_with_duplicate_type_assignment():
    """Test that an error is raised when same type is assigned to multiple functions
     of same name in same module
    """
    with pytest.raises(ValueError):
        @detector_dispatcher()
        def add(data, fn_type="foo"):
            pass