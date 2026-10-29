"""kaggle_csv(): retries on a broken download, never trusts a half-downloaded file, and skips
downloading entirely once a `.complete` marker exists (or on Kaggle, where the dataset is
already attached as an input)."""
import os

import pytest

from taobao_ctr.config import Platform
from taobao_ctr.data.download import kaggle_csv


def test_kaggle_platform_finds_attached_input(monkeypatch, tmp_path):
    monkeypatch.setattr("taobao_ctr.data.download.glob.glob", lambda *a, **k: ["/kaggle/input/x/raw_sample.csv"])
    monkeypatch.setattr("taobao_ctr.data.download.os.path.isfile", lambda p: True)
    path = kaggle_csv("raw_sample.csv", data_dir=tmp_path, kaggle_dataset="d/d", platform=Platform.KAGGLE)
    assert path == "/kaggle/input/x/raw_sample.csv"


def test_kaggle_platform_missing_input_raises(monkeypatch, tmp_path):
    monkeypatch.setattr("taobao_ctr.data.download.glob.glob", lambda *a, **k: [])
    with pytest.raises(FileNotFoundError):
        kaggle_csv("raw_sample.csv", data_dir=tmp_path, kaggle_dataset="d/d", platform=Platform.KAGGLE)


def test_skips_download_when_already_complete(monkeypatch, tmp_path):
    folder = tmp_path / "raw_sample"
    folder.mkdir()
    (folder / "raw_sample.csv").write_text("user,clk\n1,0\n")
    (folder / ".complete").touch()

    def _should_not_be_called(*a, **k):
        raise AssertionError("kagglehub should not be called when a .complete marker exists")

    monkeypatch.setattr("taobao_ctr.data.download._kagglehub_download", _should_not_be_called)
    path = kaggle_csv("raw_sample.csv", data_dir=tmp_path, kaggle_dataset="d/d", platform=Platform.LOCAL)
    assert path == str(folder / "raw_sample.csv")


def test_half_downloaded_file_without_marker_is_redownloaded(monkeypatch, tmp_path):
    folder = tmp_path / "raw_sample"
    folder.mkdir()
    (folder / "raw_sample.csv").write_text("incomplete")  # no .complete marker next to it

    calls = []

    def _fake_download(kaggle_dataset, fname, out_folder):
        calls.append((kaggle_dataset, fname, out_folder))
        return str(folder / "raw_sample.csv")

    monkeypatch.setattr("taobao_ctr.data.download._kagglehub_download", _fake_download)
    path = kaggle_csv("raw_sample.csv", data_dir=tmp_path, kaggle_dataset="d/d", platform=Platform.LOCAL)
    assert len(calls) == 1, "a file without a .complete marker must be redownloaded, not reused"
    assert path == str(folder / "raw_sample.csv")
    assert (folder / ".complete").exists()


def test_retries_then_succeeds(monkeypatch, tmp_path):
    attempts = []

    def _flaky_download(kaggle_dataset, fname, out_folder):
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("Connection broken: IncompleteRead")
        os.makedirs(out_folder, exist_ok=True)  # a real download would create this
        path = os.path.join(out_folder, fname)
        open(path, "w").close()
        return path

    monkeypatch.setattr("taobao_ctr.data.download._kagglehub_download", _flaky_download)
    sleeps = []
    path = kaggle_csv(
        "raw_sample.csv", data_dir=tmp_path, kaggle_dataset="d/d", platform=Platform.LOCAL,
        retries=4, sleep=sleeps.append,
    )
    assert len(attempts) == 3
    assert sleeps == [30, 60]  # 30 * attempt, for the 2 failed attempts
    assert path.endswith("raw_sample.csv")
    assert os.path.exists(os.path.join(tmp_path, "raw_sample", ".complete"))


def test_raises_after_exhausting_retries(monkeypatch, tmp_path):
    def _always_fails(kaggle_dataset, fname, out_folder):
        raise RuntimeError("Connection broken: IncompleteRead")

    monkeypatch.setattr("taobao_ctr.data.download._kagglehub_download", _always_fails)
    with pytest.raises(RuntimeError):
        kaggle_csv(
            "raw_sample.csv", data_dir=tmp_path, kaggle_dataset="d/d", platform=Platform.LOCAL,
            retries=3, sleep=lambda s: None,
        )
    assert not os.path.exists(os.path.join(tmp_path, "raw_sample", ".complete"))
