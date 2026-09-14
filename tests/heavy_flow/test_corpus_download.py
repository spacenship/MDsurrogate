"""No real network requests or sleeps in download policy tests."""
import io
from email.utils import formatdate
import urllib.error
import pytest
from tests.heavy_flow.test_corpus_plan import runner


@pytest.fixture
def setup(monkeypatch, tmp_path):
    module = runner()
    clock = [1000.0]
    waits = []
    def sleep(seconds):
        waits.append(seconds)
        clock[0] += seconds
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(module.time, 'time', lambda: clock[0])
    monkeypatch.setattr(module.time, 'sleep', sleep)
    return module, tmp_path / 'file.h5.part', tmp_path / 'file.h5', waits


def request(module, temporary, target, **options):
    downloader = module.ShardDownloader(**options)
    downloader.download('https://example.invalid/shard', temporary, target, 4, chunk=0, domain='d0')
    return downloader


def test_pacing_between_successful_files(setup, monkeypatch):
    m, part, target, waits = setup
    monkeypatch.setattr(m.urllib.request, 'urlopen', lambda *a, **kw: io.BytesIO(b'data'))
    downloader = request(m, part, target)
    downloader.download('unused', part, target, 4, chunk=1, domain='d1')
    assert waits == [5.0]
    assert target.read_bytes() == b'data' and not part.exists()


def test_429_retry_after_and_503_backoff(setup, monkeypatch, capsys):
    m, part, target, waits = setup
    outcomes = iter([urllib.error.HTTPError('url', 429, 'limited', {'Retry-After': '90'}, None),
                     urllib.error.HTTPError('url', 503, 'busy', {}, None), io.BytesIO(b'data')])
    def open_url(*a, **kw):
        result = next(outcomes)
        if isinstance(result, Exception):
            raise result
        return result
    monkeypatch.setattr(m.urllib.request, 'urlopen', open_url)
    request(m, part, target)
    assert waits == [90.0, 60.0]
    assert 'download_retry' in capsys.readouterr().out
    assert target.read_bytes() == b'data'


def test_retry_after_http_date_and_invalid_values(setup):
    m, *_ = setup
    assert m.retry_after_seconds(formatdate(1120, usegmt=True)) == 120
    for value in (None, 'invalid', '-1', 'nan', 'inf'):
        assert m.retry_after_seconds(value) == 0


def test_timeout_exhaustion_and_capped_backoff(setup, monkeypatch):
    m, part, target, waits = setup
    def fail(*a, **kw):
        raise TimeoutError('timeout')
    monkeypatch.setattr(m.urllib.request, 'urlopen', fail)
    with pytest.raises(RuntimeError, match='after 4 attempts'):
        request(m, part, target, retries=3, backoff_max=40)
    assert waits == [30, 40, 40]
    assert not target.exists()


def test_non_retryable_http_and_disk_errors(setup, monkeypatch):
    m, part, target, waits = setup
    for error in (urllib.error.HTTPError('url', 404, 'missing', {}, None), OSError(28, 'disk full')):
        def fail(*a, **kw):
            raise error
        monkeypatch.setattr(m.urllib.request, 'urlopen', fail)
        with pytest.raises(type(error)):
            request(m, part, target)
    assert waits == [] and not target.exists()


def test_short_transfer_restarts_partial_file(setup, monkeypatch):
    m, part, target, waits = setup
    outcomes = iter([io.BytesIO(b'xx'), io.BytesIO(b'data')])
    monkeypatch.setattr(m.urllib.request, 'urlopen', lambda *a, **kw: next(outcomes))
    request(m, part, target)
    assert waits == [30]
    assert target.read_bytes() == b'data' and not part.exists()


@pytest.mark.parametrize('options', [{'interval': 0}, {'interval': float('nan')},
                                  {'retries': -1}, {'backoff_max': 1}])
def test_invalid_download_settings(options):
    with pytest.raises(ValueError):
        runner().ShardDownloader(**options)
