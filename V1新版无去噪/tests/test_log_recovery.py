"""Exercise locked CSV recovery without touching any actual experiment files."""
from pathlib import Path
import os

import pytest

from utils.io import read_csv, write_csv


def test_locked_csv_retains_new_snapshot_and_commits_after_unlock(tmp_path, monkeypatch):
    path = tmp_path/'history.csv'
    write_csv(path,[dict(completed_epoch=1,total=.2)])
    original = path.read_bytes()
    replace = Path.replace

    def locked_replace(self, target):
        if Path(target) == path:
            raise PermissionError('simulated spreadsheet sharing lock')
        return replace(self,target)

    with monkeypatch.context() as patch:
        patch.setattr(Path,'replace',locked_replace)
        with pytest.warns(RuntimeWarning,match='locked'):
            write_csv(path,[dict(completed_epoch=1,total=.2),dict(completed_epoch=2,total=.1)],allow_locked=True)
        assert path.read_bytes() == original
        assert read_csv(path)[-1]['completed_epoch'] == '2'
    write_csv(path,read_csv(path),allow_locked=True)
    assert not path.with_name('history.csv.pending.csv').exists()
    assert read_csv(path)[-1]['completed_epoch'] == '2'


def test_locked_non_log_destination_fails_explicitly(tmp_path,monkeypatch):
    path=tmp_path/'metadata.csv'
    replace=Path.replace

    def fail(self,target):
        if Path(target)==path:
            raise PermissionError('locked')
        return replace(self,target)

    monkeypatch.setattr(Path,'replace',fail)
    with pytest.raises(PermissionError):
        write_csv(path,[dict(filename='x.png')])
    assert path.with_name('metadata.csv.pending.csv').exists()


@pytest.mark.skipif(os.name!='nt',reason='Windows file sharing API required')
def test_real_windows_file_sharing_lock(tmp_path):
    import ctypes
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,
                                wintypes.LPVOID,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    kernel.CreateFileW.restype=wintypes.HANDLE
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    path=tmp_path/'history.csv'
    write_csv(path,[dict(completed_epoch=1)])
    # Deny writes and deletion, as a spreadsheet reader can do on Windows.
    handle=kernel.CreateFileW(str(path),0x80000000,1,None,3,128,None)
    assert handle not in (None,ctypes.c_void_p(-1).value),ctypes.get_last_error()
    try:
        with pytest.warns(RuntimeWarning,match='locked'):
            write_csv(path,[dict(completed_epoch=1),dict(completed_epoch=2)],allow_locked=True)
        assert read_csv(path)[-1]['completed_epoch']=='2'
        assert '2' not in path.read_text(encoding='utf-8-sig')
    finally:
        kernel.CloseHandle(handle)
    write_csv(path,read_csv(path),allow_locked=True)
    assert not path.with_name('history.csv.pending.csv').exists()
