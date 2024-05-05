from toolboxv2 import get_logger

try:
    from .quickNote import Tools
    from .QnWidget import get_widget

    quickNote_ACTIVE = True
except ImportError as e:
    quickNote_ACTIVE = e

get_logger().info(f"quickNote STATE : {quickNote_ACTIVE}")
Name = "quickNote"
# private = True
