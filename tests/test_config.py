
def test_generic() :
    a = 10
    b = 10
    assert a == b


def test_generic_error() :
    a = 10
    b = 100
    assert a != b