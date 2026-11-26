/**
 * XR samples as VMC Protocol (OSC) bundles, the alternative to the JSON motion messages.
 *
 * VMC Protocol is OSC normally carried over UDP; here the same binary OSC bundle goes over the "teleop" DataChannel
 * (one bundle per sample). Device poses use the Performer -> Marionette tracking messages:
 *
 *   /VMC/Ext/OK        i       1 (loaded)
 *   /VMC/Ext/T         f       time [s] (performance.now() / 1000)
 *   /VMC/Ext/Hmd/Pos   s fffffff   "HMD", p.xyz, q.xyzw
 *   /VMC/Ext/Con/Pos   s fffffff   "LeftController" / "RightController", p.xyz, q.xyzw (grip pose)
 *   /VMC/Ext/Con       i s i i i fff   active(2 = axis changed), name, isLeft, isTouch, isAxis, axis.xyz
 *                      names: "Trigger" (axis.x = value), "Grip" (axis.x = value), "Stick" (axis.xy)
 *
 * Coordinates follow VMC (Unity, left-handed: x right, y up, z forward). WebXR is right-handed (-z forward), so
 * p = (x, y, -z) and q = (-x, -y, z, w). The Gamepad API's stick y is positive down; VMC's (SteamVR's) is positive up.
 */
import type { MotionFormat, ServerInfo, XRHandSample, XRSample } from './types'

/**
 * The server decides the format (env TELEOP_MOTION_FORMAT, default "vmc", sent in /api/info). A server that does
 * not say (older than the VMC support, e.g. one still running while the web app was rebuilt) only reads JSON.
 */
export function motionFormat(info: ServerInfo | null | undefined): MotionFormat {
  return info?.motionFormat === 'vmc' ? 'vmc' : 'json'
}

export const VMC_SERIAL = { head: 'HMD', left: 'LeftController', right: 'RightController' } as const

type Arg = { t: 'i'; v: number } | { t: 'f'; v: number } | { t: 's'; v: string }

const enc = new TextEncoder()

function pad4(n: number) {
  return (n + 4) & ~3 // OSC strings: NUL terminated, padded to 4 bytes
}

function message(addr: string, args: Arg[]): Uint8Array {
  const tags = ',' + args.map((a) => a.t).join('')
  const strs = [addr, tags, ...args.filter((a) => a.t === 's').map((a) => a.v as string)].map((s) => enc.encode(s))
  let size = pad4(strs[0].length) + pad4(strs[1].length)
  let si = 2
  for (const a of args) size += a.t === 's' ? pad4(strs[si++].length) : 4
  const out = new Uint8Array(size)
  const dv = new DataView(out.buffer)
  let o = 0
  const putStr = (b: Uint8Array) => {
    out.set(b, o)
    o += pad4(b.length)
  }
  putStr(strs[0])
  putStr(strs[1])
  si = 2
  for (const a of args) {
    if (a.t === 's') putStr(strs[si++])
    else if (a.t === 'i') (dv.setInt32(o, a.v), (o += 4))
    else (dv.setFloat32(o, a.v), (o += 4))
  }
  return out
}

function bundle(msgs: Uint8Array[]): Uint8Array {
  const size = 16 + msgs.reduce((n, m) => n + 4 + m.length, 0)
  const out = new Uint8Array(size)
  const dv = new DataView(out.buffer)
  out.set(enc.encode('#bundle'), 0)
  dv.setUint32(12, 1) // time tag "immediately"
  let o = 16
  for (const m of msgs) {
    dv.setInt32(o, m.length)
    out.set(m, o + 4)
    o += 4 + m.length
  }
  return out
}

const f = (v: number): Arg => ({ t: 'f', v })
const i = (v: number): Arg => ({ t: 'i', v })
const s = (v: string): Arg => ({ t: 's', v })

function pose(serial: string, p: number[], q: number[]): Arg[] {
  return [s(serial), f(p[0]), f(p[1]), f(-p[2]), f(-q[0]), f(-q[1]), f(q[2]), f(q[3])]
}

function input(name: string, left: boolean, x: number, y = 0): Uint8Array {
  return message('/VMC/Ext/Con', [i(2), s(name), i(left ? 1 : 0), i(0), i(1), f(x), f(y), f(0)])
}

export function encodeVmc(sample: XRSample): Uint8Array {
  const msgs = [message('/VMC/Ext/OK', [i(1)]), message('/VMC/Ext/T', [f(sample.ts / 1000)])]
  if (sample.head) msgs.push(message('/VMC/Ext/Hmd/Pos', pose(VMC_SERIAL.head, sample.head.p, sample.head.q)))
  for (const side of ['left', 'right'] as const) {
    const h: XRHandSample | undefined = sample[side]
    if (!h) continue
    const left = side === 'left'
    msgs.push(message('/VMC/Ext/Con/Pos', pose(VMC_SERIAL[side], h.p, h.q)))
    msgs.push(input('Trigger', left, h.trigger))
    msgs.push(input('Grip', left, h.grip))
    if (h.axes) msgs.push(input('Stick', left, h.axes[0], -h.axes[1])) // VMC / SteamVR: stick up = +y
  }
  return bundle(msgs)
}
