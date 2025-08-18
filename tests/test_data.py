import os

def test_data_files_exist_after_build(tmp_path=None):
    assert os.path.exists('data/raw') and os.path.exists('data/processed')
