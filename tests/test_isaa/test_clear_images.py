"""Tests fuer purge_images_from_messages (icli /clear_images).

Hintergrund: Discord/Vision-Bilder persistieren als Base64-Data-URIs in der
Session-History und reiten in jedem LLM-Call mit (Context-Bloat, 75 Bilder
~= 40-150 MB Payload pro Call). /clear_images entfernt image_url-Parts und
erhaelt Text.
"""
import sys

sys.path.insert(0, "/root/ToolBoxV2")

from toolboxv2.flows.isaa.icli import purge_images_from_messages


def _img(data_len=1000):
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * data_len}}


def test_multipart_text_preserved():
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "Analysiere das"},
        _img(),
        {"type": "text", "text": "bitte genau"},
    ]}]
    cleaned, removed, freed = purge_images_from_messages(msgs)
    assert removed == 1
    assert freed > 0
    # Zwei Text-Parts bleiben als Liste erhalten
    assert cleaned[0]["content"] == [
        {"type": "text", "text": "Analysiere das"},
        {"type": "text", "text": "bitte genau"},
    ]


def test_single_text_part_collapses_to_string():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "Hi"}, _img()]}]
    cleaned, removed, _ = purge_images_from_messages(msgs)
    assert removed == 1
    assert cleaned[0]["content"] == "Hi"


def test_image_only_message_gets_placeholder():
    msgs = [{"role": "user", "content": [_img(), _img()]}]
    cleaned, removed, _ = purge_images_from_messages(msgs)
    assert removed == 2
    assert cleaned[0]["content"] == "[Bild entfernt]"


def test_plain_string_messages_untouched():
    msgs = [{"role": "assistant", "content": "Alles klar"}]
    cleaned, removed, freed = purge_images_from_messages(msgs)
    assert removed == 0
    assert freed == 0
    assert cleaned == msgs


def test_multiple_images_across_messages():
    msgs = [
        {"role": "user", "content": [_img()]},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": [_img(), {"type": "text", "text": "und?"}, _img()]},
    ]
    cleaned, removed, _ = purge_images_from_messages(msgs)
    assert removed == 3
    assert cleaned[1]["content"] == "ok"
    # 2 Bilder entfernt, 1 Text-Part bleibt -> kollabiert zu String
    assert cleaned[2]["content"] == "und?"


def test_75_image_history_simulates_user_report():
    """Real-Szenario: 75 Bilder in der History -> alle weg, Text bleibt."""
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": f"Nachricht {i}"},
        _img(20000),
    ]} for i in range(75)]
    cleaned, removed, freed = purge_images_from_messages(msgs)
    assert removed == 75
    assert freed > 75 * 20000  # Base64-Bytes tatsaechlich gefreut
    assert all(m["content"] == f"Nachricht {i}" for i, m in enumerate(cleaned))


def test_non_dict_entries_pass_through():
    msgs = ["raw", {"role": "user", "content": "text"}, 42]
    cleaned, removed, _ = purge_images_from_messages(msgs)
    assert removed == 0
    assert cleaned == msgs
