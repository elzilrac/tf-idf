
def test_can_run_tests():
    assert 3 + 4 == 7

def test_fixtures(bootstrap_fixture):
    assert bootstrap_fixture == "Hello World"

