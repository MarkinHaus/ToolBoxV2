import os
import sys


# Beispiel Usage:
if __name__ == "__main__":
    # Normale Ausgaben werden automatisch gespeichert
    print("Zeile 1: Gespeichert")
    print("Zeile 2: Auch gespeichert")
    print("Zeile 3: Wird gespeichert")

    # Live print wird nicht gespeichert
    live_print("Diese Zeile wird NICHT gespeichert")

    # Terminal clearen und content speichern
    input("\nDrücke Enter um Terminal zu clearen...")
    save_and_clear()

    # Neuer content nach clear
    print("Neuer Content nach Clear")
    live_print("Temporärer Text (nicht gespeichert)")

    # Original content wiederherstellen
    input("\nDrücke Enter um ursprünglichen Content wiederherzustellen...")
    save_and_clear()
    restore_content()

    print("\n--- Original Content wurde wiederhergestellt ---")
