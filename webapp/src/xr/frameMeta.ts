/**
 * Per-frame metadata from the server ("fmeta": the XR-space head pose a frame was rendered with), matched to the
 * frame actually on screen via the WebRTC RTP timestamp.
 *
 * aiortc sends RTP timestamp = random 32-bit origin + frame pts (90 kHz). We receive (pts, pose) over the
 * DataChannel and (rtpTimestamp) from requestVideoFrameCallback, so the origin is the value of
 * (rtp - pts) mod 2^32 that keeps re-appearing.
 */
export interface FrameMeta {
  pts: number
  view: 'immersive' | 'screen'
  spub?: number // server wall clock [ms] when the frame was published
  // immersive only: the head pose the frame was rendered with
  head?: { p: [number, number, number]; q: [number, number, number, number] }
  ipd?: number
  fov?: number
  ts?: number | null // performance.now of the XR sample this pose came from
}

const buffer: FrameMeta[] = []
const originVotes = new Map<number, number>()
let origin: number | null = null

const mod32 = (v: number) => ((v % 4294967296) + 4294967296) % 4294967296

export function pushFrameMeta(m: FrameMeta) {
  buffer.push(m)
  if (buffer.length > 120) buffer.shift()
}

/** Newest metadata that carries a render pose (immersive view). */
export function latestFrameMeta(): ImmersiveMeta | undefined {
  for (let i = buffer.length - 1; i >= 0; i--) if (buffer[i].head) return buffer[i] as ImmersiveMeta
  return undefined
}

export type ImmersiveMeta = FrameMeta & Required<Pick<FrameMeta, 'head' | 'ipd' | 'fov'>>

/** Metadata of the frame with this RTP timestamp (or undefined while the origin is still unknown). */
export function frameMetaForRtp(rtp: number): FrameMeta | undefined {
  if (origin !== null) {
    const pts = mod32(rtp - origin)
    for (let i = buffer.length - 1; i >= 0; i--) if (buffer[i].pts === pts) return buffer[i]
  }
  // vote for the origin using the recent metadata
  for (let i = Math.max(0, buffer.length - 30); i < buffer.length; i++) {
    const o = mod32(rtp - buffer[i].pts)
    const n = (originVotes.get(o) ?? 0) + 1
    originVotes.set(o, n)
    if (n >= 3 && origin !== o) origin = o
  }
  if (originVotes.size > 4000) originVotes.clear()
  return undefined
}

/** Call when a new WebRTC connection starts (new random RTP origin). */
export function resetFrameMeta() {
  buffer.length = 0
  originVotes.clear()
  origin = null
}

export const frameSync = {
  get synced() {
    return origin !== null
  },
}
