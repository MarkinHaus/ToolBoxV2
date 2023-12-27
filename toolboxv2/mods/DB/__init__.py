from toolboxv2 import get_logger

try:
    from .tb_adapter import Tools
    DB_ACTIVE = True
except ImportError as e:
    DB_ACTIVE = e

get_logger().info(f"DB STATE : {DB_ACTIVE}")
Name = "DB"
# private = True
