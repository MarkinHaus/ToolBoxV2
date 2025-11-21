"""
WhatsApp Advanced Tools for ProA Kernel
Version: 1.0.0

Bietet dem Agenten Werkzeuge für interaktive Nachrichten, Kontaktmanagement
und Gruppen-Funktionen (Broadcasts).
"""

from typing import List, Dict, Any, Optional
import json


class WhatsAppKernelTools:
    """WhatsApp-spezifische Tools für die Agenten-Integration"""

    def __init__(self, messenger, kernel, output_router):
        self.messenger = messenger
        self.kernel = kernel
        self.output_router = output_router
        # Simulierter Speicher für Gruppen (Broadcast-Listen)
        # In Produktion: Datenbank nutzen!
        self.broadcast_lists: Dict[str, List[str]] = {}

        # ===== INTERACTIVE MESSAGES =====

    async def send_buttons(
        self,
        user_id: str,
        text: str,
        buttons: List[Dict[str, str]],
        header: Optional[str] = None,
        footer: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Sendet eine Nachricht mit bis zu 3 Buttons.

        Args:
            user_id: Telefonnummer des Empfängers
            text: Nachrichtentext
            buttons: Liste von Dictionaries [{"id": "yes_btn", "title": "Ja"}, ...]
            header: Optionaler Header-Text
            footer: Optionaler Footer-Text
        """
        # Formatierung für whatsapp-python Wrapper vorbereiten
        formatted_buttons = []
        for btn in buttons:
            formatted_buttons.append({
                "type": "reply",
                "reply": {
                    "id": btn.get("id", "btn_id"),
                    "title": btn.get("title", "Button")
                }
            })

        try:
            # Über OutputRouter, damit es im Kernel-Flow bleibt
            # Wir nutzen hier metadata injection, um dem Router zu sagen: Mach interaktiv!
            metadata = {
                "interactive": {
                    "type": "button",
                    "buttons": formatted_buttons,
                    "header": header,
                    "footer": footer
                }
            }
            await self.output_router.send_response(user_id, text, metadata=metadata)
            return {"success": True, "type": "buttons_sent"}
        except Exception as e:
            return {"error": str(e)}

    async def send_menu_list(
        self,
        user_id: str,
        text: str,
        button_text: str,
        sections: List[Dict[str, Any]],
        title: str = "Menü"
    ) -> Dict[str, Any]:
        """
        Sendet ein Listen-Menü (bis zu 10 Optionen).

        Args:
            sections: Liste von Sektionen [{"title": "Sektion 1", "rows": [{"id": "1", "title": "Option A", "description": "Details"}]}]
        """
        try:
            # Datenstruktur anpassen
            formatted_rows = []
            for section in sections:
                # whatsapp-python erwartet oft flache Struktur oder spezifische API-Formate
                # Wir bauen hier die Standard Cloud API Struktur nach
                sec_data = {
                    "title": section.get("title", "Optionen"),
                    "rows": section.get("rows", [])
                }
                formatted_rows.append(sec_data)

            metadata = {
                "interactive": {
                    "type": "list",
                    "button_text": button_text,
                    "rows": formatted_rows,
                    "title": title
                }
            }
            await self.output_router.send_response(user_id, text, metadata=metadata)
            return {"success": True, "type": "list_sent"}
        except Exception as e:
            return {"error": str(e)}

    # ===== BROADCAST / GROUP SIMULATION =====

    async def create_broadcast_list(self, name: str, user_ids: List[str]) -> Dict[str, Any]:
        """Erstellt eine neue Broadcast-Liste (Simulierte Gruppe)"""
        self.broadcast_lists[name] = user_ids
        return {"success": True, "list_name": name, "members": len(user_ids)}

    async def add_to_broadcast(self, list_name: str, user_id: str) -> Dict[str, Any]:
        """Fügt User zur Liste hinzu"""
        if list_name not in self.broadcast_lists:
            self.broadcast_lists[list_name] = []

        if user_id not in self.broadcast_lists[list_name]:
            self.broadcast_lists[list_name].append(user_id)

        return {"success": True, "list_name": list_name, "total_members": len(self.broadcast_lists[list_name])}

    async def send_broadcast(self, list_name: str, content: str, is_interactive: bool = False) -> Dict[str, Any]:
        """
        Sendet eine Nachricht an alle in der Liste.
        """
        if list_name not in self.broadcast_lists:
            return {"error": f"List {list_name} not found"}

        members = self.broadcast_lists[list_name]
        count = 0

        for user_id in members:
            try:
                # Kurze Pause um Rate-Limits zu vermeiden
                import asyncio
                await asyncio.sleep(0.1)
                await self.output_router.send_response(user_id, content)
                count += 1
            except Exception as e:
                print(f"Failed to send to {user_id}: {e}")

        return {"success": True, "sent_count": count}

    # ===== CONTACT MANAGEMENT =====

    async def send_contact(self, user_id: str, contact_name: str, contact_phone: str) -> Dict[str, Any]:
        """Sendet eine vCard / Kontaktkarte"""
        try:
            # Muss direkt über Messenger gehen, da Router meist auf Text/Media spezialisiert
            data = {
                "name": {"formatted_name": contact_name, "first_name": contact_name},
                "phones": [{"phone": contact_phone, "type": "MOBILE"}]
            }
            self.messenger.send_contacts(data, user_id)
            return {"success": True}
        except Exception as e:
            return {"error": str(e)}

    async def mark_as_read(self, message_id: str) -> Dict[str, Any]:
        """Markiert eine Nachricht explizit als gelesen"""
        try:
            self.messenger.mark_as_read(message_id)
            return {"success": True}
        except Exception as e:
            return {"error": str(e)}

    # ===== EXPORT =====

    async def export_to_agent(self):
        """Exportiert die Tools zum Agenten"""
        agent = self.kernel.agent

        # Buttons
        await agent.add_tool(
            self.send_buttons,
            "whatsapp_send_buttons",
            description="Sendet eine Nachricht mit bis zu 3 Buttons. Args: user_id, text, buttons=[{'id': '1', 'title': 'Yes'}]."
        )

        # Listen
        await agent.add_tool(
            self.send_menu_list,
            "whatsapp_send_list",
            description="Sendet ein Auswahlmenü. Args: user_id, text, button_text, sections=[{'title': 'Main', 'rows': [{'id': '1', 'title': 'Option'}]}]."
        )

        # Broadcasts
        await agent.add_tool(
            self.create_broadcast_list,
            "whatsapp_create_group",
            description="Erstellt eine Broadcast-Gruppe. Args: name, user_ids list."
        )

        await agent.add_tool(
            self.add_to_broadcast,
            "whatsapp_add_to_group",
            description="Fügt User zur Gruppe hinzu. Args: list_name, user_id."
        )

        await agent.add_tool(
            self.send_broadcast,
            "whatsapp_send_to_group",
            description="Sendet Nachricht an alle in der Gruppe. Args: list_name, content."
        )

        # Kontakt
        await agent.add_tool(
            self.send_contact,
            "whatsapp_send_contact",
            description="Teilt einen Kontakt. Args: user_id, contact_name, contact_phone."
        )

        print("✓ WhatsApp Advanced Tools exported to agent")
