import requests

def test_startup(api_base_url):
    """Smoke-test: test that the servers starts and health endpoint returns a 200 status code"""
    r = requests.get(f"{api_base_url}/health", timeout=5)
    assert r.status_code == 200
