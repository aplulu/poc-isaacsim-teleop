"""Isaac Sim window that displays the 6-digit pairing codes and the connected sessions."""

from __future__ import annotations

import time
from typing import Callable

import omni.ui as ui

from .auth import AuthManager

CODE_STYLE = {"font_size": 54, "color": 0xFF7FE0FF}
TITLE_STYLE = {"font_size": 18, "color": 0xFFFFFFFF}
DIM_STYLE = {"font_size": 14, "color": 0xFFA0A0A0}
CARD_STYLE = {"background_color": 0xFF2A2A2A, "border_radius": 6}


class PairingWindow:
    def __init__(
        self,
        auth: AuthManager,
        urls: list[str],
        on_revoke: Callable[[str], None],
        get_status: Callable[[], dict],
    ) -> None:
        self.auth = auth
        self.urls = urls
        self.on_revoke = on_revoke
        self.get_status = get_status
        self._version = -1
        self._last_tick = 0.0
        self._countdowns: list[tuple[ui.Label, float]] = []
        self.window = ui.Window("VR Teleop - Pairing", width=460, height=620, visible=True)
        self.window.deferred_dock_in("Property", ui.DockPolicy.DO_NOTHING)
        self._status_label: ui.Label | None = None
        self._rebuild()

    def _rebuild(self) -> None:
        pending, sessions = self.auth.snapshot()
        self._countdowns = []
        with self.window.frame:
            with ui.ScrollingFrame() as self._scroll:
                with ui.VStack(spacing=8, height=0):
                    ui.Spacer(height=4)
                    ui.Label("Isaac Sim VR Teleop", style={"font_size": 22}, height=0)
                    ui.Label(f"Pairing requests ({len(pending)})", style=TITLE_STYLE, height=0)
                    if not pending:
                        ui.Label(
                            "No pending requests. Press 'Connect' in the web app to get a code.",
                            style=DIM_STYLE,
                            height=0,
                            word_wrap=True,
                        )
                    for idx, req in enumerate(pending):  # newest first
                        with ui.ZStack(height=0):
                            ui.Rectangle(style=CARD_STYLE if idx else {**CARD_STYLE, "background_color": 0xFF3A3020})
                            with ui.VStack(spacing=2, height=0):
                                ui.Spacer(height=6)
                                with ui.HStack(height=0):
                                    ui.Spacer(width=10)
                                    tag = "NEW  " if idx == 0 and len(pending) > 1 else ""
                                    ui.Label(f"{tag}{req.device_name}  ({req.remote_ip})", style=TITLE_STYLE)
                                with ui.HStack(height=0):
                                    ui.Spacer(width=10)
                                    ui.Label(f"{req.code[:3]} {req.code[3:]}", style=CODE_STYLE, height=64)
                                with ui.HStack(height=0):
                                    ui.Spacer(width=10)
                                    cd = ui.Label("", style=DIM_STYLE)
                                    self._countdowns.append((cd, req.expires_at))
                                    ui.Label(f"attempts: {req.attempts}", style=DIM_STYLE, width=90)
                                    ui.Button(
                                        "Reject",
                                        width=70,
                                        clicked_fn=lambda rid=req.request_id: self.auth.cancel_pairing(rid),
                                    )
                                    ui.Spacer(width=10)
                                ui.Spacer(height=6)
                    ui.Separator(height=8)
                    # order: pairing requests (a new code is always visible at the top) -> sessions -> URLs / status
                    ui.Label(f"Connected sessions ({len(sessions)})", style=TITLE_STYLE, height=0)
                    for s in sessions:
                        with ui.HStack(height=24, spacing=6):
                            dot = 0xFF50D050 if s.connected_peers else 0xFF808080
                            ui.Circle(radius=5, width=14, style={"background_color": dot})
                            ui.Label(f"{s.device_name} ({s.remote_ip})")
                            exp = time.strftime("%H:%M", time.localtime(s.expires_at))
                            ui.Label(f"until {exp}", style=DIM_STYLE, width=80)
                            ui.Button("Revoke", width=70, clicked_fn=lambda sid=s.session_id: self.on_revoke(sid))
                    ui.Separator(height=8)
                    ui.Label("Open one of these URLs in the headset browser:", style=DIM_STYLE, height=0)
                    for url in self.urls:
                        ui.StringField(height=22).model.set_value(url)
                    self._status_label = ui.Label("", style=DIM_STYLE, height=0, word_wrap=True)
        self._version = self.auth.version
        self._scroll.scroll_y = 0  # show the newest request (top)
        self._tick(force=True)

    def _tick(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_tick < 0.5:
            return
        self._last_tick = now
        for label, exp in self._countdowns:
            label.text = f"expires in {max(0, int(exp - now))} s"
        if self._status_label is not None:
            st = self.get_status()
            self._status_label.text = (
                f"sim fps: {st.get('fps', 0)}   camera: {st.get('camera', '-')}   "
                f"gripper: {'closed' if st.get('gripper', {}).get('closed') else 'open'}"
                f"{' (' + st['gripper']['holding'] + ')' if st.get('gripper', {}).get('holding') else ''}"
            )

    def update(self) -> None:
        """Call every frame from the main loop."""
        if self.auth.version != self._version:
            self._rebuild()
            # Bring the window to front when a new request arrives
            self.window.visible = True
            self.window.focus()
        else:
            self._tick()

    def destroy(self) -> None:
        self.window.destroy()
