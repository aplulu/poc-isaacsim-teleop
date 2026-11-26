/**
 * Receive-side video timings, reported to the server once a second ("perf" message) and shown in the panels.
 *
 *   fps            frames shown by the <video> element (requestVideoFrameCallback)
 *   decodeMs       average decode time per frame (inbound-rtp totalDecodeTime)
 *   jitterMs       average time a frame waits in the jitter buffer (jitterBufferDelay / emitted)
 *   recvToShowMs   packet received -> frame composited (rVFC receiveTime -> expectedDisplayTime)
 *   poseToShowMs   XR sample sent -> frame rendered with that head pose is composited (immersive only;
 *                  both ends are this page's performance.now clock, so this is the full round trip)
 *   rttMs          ICE round trip
 *   kbps, size, decoder, dropped/s, freezes
 */
import { frameMetaForRtp } from './xr/frameMeta'

export interface ClientPerf {
  fps: number
  decodeMs: number | null
  jitterMs: number | null
  recvToShowMs: number | null
  poseToShowMs: number | null
  poseToShowP95: number | null
  pubToShowMs: number | null // server published -> composited here (clock offset from ping, +-RTT/2)
  rttMs: number | null
  kbps: number | null
  size: string | null
  decoder: string | null
  droppedPerSec: number | null
  freezes: number | null
  xrFps: number | null
}

const shown: number[] = [] // display timestamps of the last second
const recvToShow: number[] = []
const poseToShow: number[] = []
const xrFrames: number[] = []
const pubToShow: number[] = []
/** server wall clock - (performance.timeOrigin + performance.now()), from ping/pong (ms) */
let clockOffset: number | null = null
let bestRtt = Infinity

export function onPong(sentPerf: number, serverWallSec: number) {
  const now = performance.now()
  const rtt = now - sentPerf
  // keep the estimate of the fastest round trip (least queuing)
  if (rtt <= bestRtt * 1.5 || clockOffset === null) {
    bestRtt = Math.min(bestRtt, rtt)
    clockOffset = serverWallSec * 1000 - (performance.timeOrigin + (sentPerf + now) / 2)
  }
}

export const clientPerf: { current: ClientPerf | null } = { current: null }

type Rvfc = HTMLVideoElement & {
  requestVideoFrameCallback?: (cb: (now: number, md: VideoFrameCallbackMetadata) => void) => number
  cancelVideoFrameCallback?: (id: number) => void
}
interface VideoFrameCallbackMetadata {
  expectedDisplayTime: number
  receiveTime?: number
  rtpTimestamp?: number
}

function push(a: number[], v: number, max = 240) {
  a.push(v)
  if (a.length > max) a.shift()
}

const avg = (a: number[]) => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : null)
const p95 = (a: number[]) => {
  if (!a.length) return null
  const s = [...a].sort((x, y) => x - y)
  return s[Math.min(s.length - 1, Math.floor(s.length * 0.95))]
}

/** Watch the frames the <video> element presents. Returns a stop function. */
export function watchVideo(video: HTMLVideoElement): () => void {
  const v = video as Rvfc
  if (!v.requestVideoFrameCallback) return () => undefined
  let alive = true
  let id = 0
  const onFrame = (cbNow: number, md: VideoFrameCallbackMetadata) => {
    if (!alive) return
    // The callback's timestamps are not always on this page's performance.now clock: during a WebXR session
    // (Quest Browser) they run ~100 s behind it. Use them only for differences among themselves and place the
    // display time on the performance.now clock as "now + (expected display - callback time)".
    const lead = Math.max(0, Math.min(md.expectedDisplayTime - cbNow, 100))
    const t = performance.now() + lead
    push(shown, t)
    if (md.receiveTime != null) push(recvToShow, md.expectedDisplayTime - md.receiveTime)
    if (md.rtpTimestamp != null) {
      const meta = frameMetaForRtp(md.rtpTimestamp)
      if (meta?.ts != null) push(poseToShow, t - meta.ts)
      if (meta?.spub != null && clockOffset !== null) push(pubToShow, performance.timeOrigin + t + clockOffset - meta.spub)
    }
    id = v.requestVideoFrameCallback!(onFrame)
  }
  id = v.requestVideoFrameCallback(onFrame)
  return () => {
    alive = false
    v.cancelVideoFrameCallback?.(id)
  }
}

/** Call once per XR animation frame (headset refresh actually achieved). */
export function markXRFrame() {
  push(xrFrames, performance.now(), 200)
}

let prev: { t: number; decoded: number; decodeTime: number; jbDelay: number; jbCount: number; bytes: number; dropped: number } | null = null

/** Sample the WebRTC stats (call about once a second). */
export async function sampleStats(pc: RTCPeerConnection): Promise<ClientPerf> {
  const now = performance.now()
  let inbound: Record<string, number | string> | null = null
  let rtt: number | null = null
  const report = await pc.getStats()
  report.forEach((s: Record<string, unknown>) => {
    if (s.type === 'inbound-rtp' && s.kind === 'video') inbound = s as Record<string, number | string>
    if (s.type === 'candidate-pair' && s.nominated && typeof s.currentRoundTripTime === 'number') {
      rtt = (s.currentRoundTripTime as number) * 1000
    }
  })
  const out: ClientPerf = {
    fps: 0,
    decodeMs: null,
    jitterMs: null,
    recvToShowMs: null,
    poseToShowMs: null,
    poseToShowP95: null,
    pubToShowMs: null,
    rttMs: rtt,
    kbps: null,
    size: null,
    decoder: null,
    droppedPerSec: null,
    freezes: null,
    xrFps: null,
  }
  while (shown.length && shown[0] < now - 1000) shown.shift()
  out.fps = shown.length
  while (xrFrames.length && xrFrames[0] < now - 1000) xrFrames.shift()
  out.xrFps = xrFrames.length || null
  out.recvToShowMs = avg(recvToShow)
  out.poseToShowMs = avg(poseToShow)
  out.poseToShowP95 = p95(poseToShow)
  out.pubToShowMs = avg(pubToShow)
  recvToShow.length = 0
  poseToShow.length = 0
  pubToShow.length = 0
  const s = inbound as Record<string, number | string> | null
  if (s) {
    const cur = {
      t: now,
      decoded: Number(s.framesDecoded ?? 0),
      decodeTime: Number(s.totalDecodeTime ?? 0),
      jbDelay: Number(s.jitterBufferDelay ?? 0),
      jbCount: Number(s.jitterBufferEmittedCount ?? 0),
      bytes: Number(s.bytesReceived ?? 0),
      dropped: Number(s.framesDropped ?? 0),
    }
    if (prev) {
      const dn = cur.decoded - prev.decoded
      const dt = (cur.t - prev.t) / 1000
      if (dn > 0) out.decodeMs = ((cur.decodeTime - prev.decodeTime) / dn) * 1000
      const dj = cur.jbCount - prev.jbCount
      if (dj > 0) out.jitterMs = ((cur.jbDelay - prev.jbDelay) / dj) * 1000
      if (dt > 0) {
        out.kbps = ((cur.bytes - prev.bytes) * 8) / 1000 / dt
        out.droppedPerSec = (cur.dropped - prev.dropped) / dt
      }
    }
    prev = cur
    if (s.frameWidth) out.size = `${s.frameWidth}x${s.frameHeight}`
    out.decoder = (s.decoderImplementation as string) ?? null
    out.freezes = s.freezeCount != null ? Number(s.freezeCount) : null
  }
  clientPerf.current = out
  return out
}

export function resetStats() {
  prev = null
  clockOffset = null
  bestRtt = Infinity
  pubToShow.length = 0
  shown.length = 0
  recvToShow.length = 0
  poseToShow.length = 0
}

/** Rounded copy for the server log. */
export function perfForServer(p: ClientPerf): Record<string, number | string | null> {
  const r = (v: number | null, d = 1) => (v == null ? null : Math.round(v * 10 ** d) / 10 ** d)
  return {
    fps: p.fps,
    xrFps: p.xrFps,
    decodeMs: r(p.decodeMs),
    jitterMs: r(p.jitterMs),
    recvToShowMs: r(p.recvToShowMs),
    poseToShowMs: r(p.poseToShowMs),
    poseToShowP95: r(p.poseToShowP95),
    pubToShowMs: r(p.pubToShowMs),
    rttMs: r(p.rttMs),
    kbps: r(p.kbps, 0),
    size: p.size,
    decoder: p.decoder,
    droppedPerSec: r(p.droppedPerSec),
    freezes: p.freezes,
  }
}
