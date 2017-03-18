

def test_precision():
    n = 1000000000
    for z in xrange(0, 1000000):
        n += 0.000001
    n -= 1000000000
    print('n is: ', n)
    # n would be 1.0.
    assert n == 0.95367431640625