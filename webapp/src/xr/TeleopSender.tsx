import { useFrame } from '@react-three/fiber'
import { useRef } from 'react'
import { getConnection, useApp } from '../store'
import type { XRHandSample, XRSample } from '../types'
import { uiHover } from './uiHover'
import { motionFormat } from '../vmc'

const SEND_INTERVAL_MS = 1000 / 60
const HOLD_RESET_MS = 800

type Edge = { prev: boolean[]; holdStart: number | null }

/**
 * Reads the XR controllers every frame and streams their poses / analog values to Isaac Sim over the
 * unreliable "teleop" DataChannel. Discrete buttons are handled here and turned into commands.
 *
 * Control hand (default right):  grip = clutch (move the arm while held), trigger = gripper,
 *                                 A/X = next camera, B/Y = randomize, stick click = home
 * Other hand:                     X/A (hold) = reset, Y/B = menu placement (hand / fixed / hidden),
 *                                 stick up/down = screen distance (free viewpoint: stick = walk),
 *                                 control-hand stick left/right = snap turn (free viewpoint),
 *                                 stick click = start recording / save the episode as success
 */
export function TeleopSender() {
  const lastSend = useRef(0)
  const edges = useRef<Record<string, Edge>>({})
  const frozenTrigger = useRef<number | null>(null)
  const sentIpd = useRef<number | null>(null)

  useFrame((state, delta, frame?: XRFrame) => {
    if (!frame) return
    const refSpace = state.gl.xr.getReferenceSpace()
    if (!refSpace) return
    const session = frame.session
    const app = useApp.getState()
    const controlHand = app.sim?.settings.hand ?? 'right'
    const sample: XRSample = { t: 'xr', ts: performance.now() }

    const viewer = frame.getViewerPose(refSpace)
    if (viewer && viewer.views.length === 2) {
      // measured IPD -> stereo camera baseline (send once, and again if it changes by > 1 mm)
      const a = viewer.views[0].transform.position
      const b = viewer.views[1].transform.position
      const ipd = Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z)
      if (ipd > 0.045 && ipd < 0.085 && (sentIpd.current == null || Math.abs(ipd - sentIpd.current) > 0.001)) {
        sentIpd.current = ipd
        app.sendIpd(ipd)
      }
    }
    if (viewer) {
      const { position: p, orientation: q } = viewer.transform
      sample.head = { p: [p.x, p.y, p.z], q: [q.x, q.y, q.z, q.w] }
    }

    for (const src of session.inputSources) {
      if (!src.gripSpace || (src.handedness !== 'left' && src.handedness !== 'right')) continue
      const pose = frame.getPose(src.gripSpace, refSpace)
      const gp = src.gamepad
      if (!pose || !gp) continue
      const { position: p, orientation: q } = pose.transform
      const hand: XRHandSample = {
        p: [p.x, p.y, p.z],
        q: [q.x, q.y, q.z, q.w],
        trigger: gp.buttons[0]?.value ?? 0,
        grip: gp.buttons[1]?.value ?? 0,
        axes: [gp.axes[2] ?? 0, gp.axes[3] ?? 0],
      }
      if (src.handedness === controlHand) {
        // keep the gripper state while the ray is on a menu button
        if (uiHover.active) {
          frozenTrigger.current ??= hand.trigger
          hand.trigger = frozenTrigger.current
        } else frozenTrigger.current = null
      }
      sample[src.handedness] = hand
      handleButtons(src.handedness, src.handedness === controlHand, gp, edges.current, delta)
    }

    const now = performance.now()
    if (now - lastSend.current >= SEND_INTERVAL_MS) {
      lastSend.current = now
      getConnection()?.sendXR(sample, motionFormat(app.info))
    }
  })

  return null
}

function handleButtons(
  handedness: 'left' | 'right',
  isControl: boolean,
  gp: Gamepad,
  edges: Record<string, Edge>,
  delta: number,
) {
  const e = (edges[handedness] ??= { prev: [], holdStart: null })
  const pressed = gp.buttons.map((b) => b.pressed)
  const down = (i: number) => pressed[i] && !e.prev[i]
  const app = useApp.getState()

  if (isControl) {
    if (down(4)) void app.cycleCamera()
    if (down(5)) void app.randomize()
    if (down(3)) void app.command('go_home', {}, 'ホーム')
  } else {
    // reset requires a short hold to avoid accidents
    if (pressed[4]) {
      e.holdStart ??= performance.now()
      if (performance.now() - e.holdStart > HOLD_RESET_MS) {
        e.holdStart = Number.POSITIVE_INFINITY
        void app.reset()
      }
    } else e.holdStart = null
    if (down(5)) app.cycleVrMenu()
    if (down(3)) void app.recToggle()
    // screen view: stick = screen distance. Free viewpoint: the stick walks (handled by the server).
    const stickY = gp.axes[3] ?? 0
    if (app.sim?.view !== 'immersive' && Math.abs(stickY) > 0.3)
      app.setScreenDistance(app.screenDistance + stickY * delta * 1.2)
  }
  e.prev = pressed
}
