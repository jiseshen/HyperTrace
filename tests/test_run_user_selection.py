from pathlib import Path

import pytest

from data import UserData
from run import select_users_by_id_file


def test_select_users_by_id_file_preserves_requested_order(tmp_path: Path):
    users = [UserData(user_id="u1"), UserData(user_id="u2"), UserData(user_id="u3")]
    user_ids_path = tmp_path / "ids.txt"
    user_ids_path.write_text("# fixed cohort\nu3\nu1\n")

    selected = select_users_by_id_file(users, user_ids_path)

    assert [user.user_id for user in selected] == ["u3", "u1"]


def test_select_users_by_id_file_rejects_missing_ids(tmp_path: Path):
    users = [UserData(user_id="u1")]
    user_ids_path = tmp_path / "ids.txt"
    user_ids_path.write_text("u2\n")

    with pytest.raises(ValueError, match="Missing users"):
        select_users_by_id_file(users, user_ids_path)
