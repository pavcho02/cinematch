import hashlib
import os
import sqlite3

from services.database_service import get_connection


PBKDF2_ITERATIONS = 100_000
CINEMATCH_USER_ID_START = 100_001


def hash_password(password: str) -> str:
    """
    Hashes a password using PBKDF2-HMAC-SHA256 with a random salt.

    The stored format is:
        salt_hex$hash_hex
    """
    salt = os.urandom(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS
    )

    return f"{salt.hex()}${password_hash.hex()}"


def verify_password(password: str, stored_password_hash: str) -> bool:
    """
    Verifies a password against the stored password hash.
    """
    try:
        salt_hex, hash_hex = stored_password_hash.split("$")
        salt = bytes.fromhex(salt_hex)
        expected_hash = bytes.fromhex(hash_hex)

        actual_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            PBKDF2_ITERATIONS
        )

        return actual_hash == expected_hash

    except ValueError:
        return False


def get_next_recommender_user_id(connection: sqlite3.Connection) -> int:
    """
    Generates a CineMatch user id that will not conflict with MovieLens users.

    MovieLens users are 1..610.
    CineMatch users start from 100001.
    """
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT MAX(recommender_user_id)
        FROM users
        """
    )

    max_user_id = cursor.fetchone()[0]

    if max_user_id is None:
        return CINEMATCH_USER_ID_START

    return max_user_id + 1


def get_user_by_username(username: str):
    """
    Returns a user by username or None.
    """
    with get_connection() as connection:
        connection.row_factory = sqlite3.Row
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                recommender_user_id,
                username,
                password_hash,
                created_at
            FROM users
            WHERE username = ?
            """,
            (username,)
        )

        row = cursor.fetchone()

        if row is None:
            return None

        return dict(row)


def get_user_by_recommender_id(recommender_user_id: int):
    """
    Returns a user by recommender_user_id or None.
    """
    with get_connection() as connection:
        connection.row_factory = sqlite3.Row
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                recommender_user_id,
                username,
                created_at
            FROM users
            WHERE recommender_user_id = ?
            """,
            (recommender_user_id,)
        )

        row = cursor.fetchone()

        if row is None:
            return None

        return dict(row)


def username_exists(username: str) -> bool:
    """
    Checks whether a username already exists.
    """
    return get_user_by_username(username) is not None


def create_user(username: str, password: str):
    """
    Creates a new CineMatch user.

    Returns:
        dict with success status and message.
    """
    username = username.strip()

    if not username:
        return {
            "success": False,
            "message": "Username cannot be empty."
        }

    if len(username) < 3:
        return {
            "success": False,
            "message": "Username must be at least 3 characters long."
        }

    if len(password) < 6:
        return {
            "success": False,
            "message": "Password must be at least 6 characters long."
        }

    if username_exists(username):
        return {
            "success": False,
            "message": "Username already exists."
        }

    password_hash = hash_password(password)

    with get_connection() as connection:
        cursor = connection.cursor()

        recommender_user_id = get_next_recommender_user_id(connection)

        cursor.execute(
            """
            INSERT INTO users (
                recommender_user_id,
                username,
                password_hash
            )
            VALUES (?, ?, ?)
            """,
            (
                recommender_user_id,
                username,
                password_hash
            )
        )

        connection.commit()

    return {
        "success": True,
        "message": "User created successfully.",
        "user": {
            "recommender_user_id": recommender_user_id,
            "username": username
        }
    }


def authenticate_user(username: str, password: str):
    """
    Authenticates a user by username and password.

    Returns:
        dict with success status and user data if successful.
    """
    username = username.strip()

    user = get_user_by_username(username)

    if user is None:
        return {
            "success": False,
            "message": "Invalid username or password."
        }

    is_valid_password = verify_password(
        password,
        user["password_hash"]
    )

    if not is_valid_password:
        return {
            "success": False,
            "message": "Invalid username or password."
        }

    return {
        "success": True,
        "message": "Login successful.",
        "user": {
            "id": user["id"],
            "recommender_user_id": user["recommender_user_id"],
            "username": user["username"],
            "created_at": user["created_at"]
        }
    }