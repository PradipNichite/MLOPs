import pytest


class NotInRange(Exception) :
    def __init__(self, message="Value not in given range") :
        self.message = message
        super().__init__(self.message)

def test_generic() :
    a = 10
    b = 10
    assert a == b


def test_generic_error() :
    a = 10
    b = 100
    assert a != b


def test_exception_case() :
    a = 10
    with pytest.raises(NotInRange) :
        if a not in range(20, 30) :
            raise NotInRange