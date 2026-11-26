import { api } from './api'
import type { MotionFormat, SimState, XRSample } from './types'
import { encodeVmc } from './vmc'
import { type FrameMeta, pushFrameMeta, resetFrameMeta } from './xr/frameMeta'
import { type ClientPerf, onPong, perfForServer, resetStats, sampleStats } from './perfStats'

type Listener = {
  onState: (s: SimState) => void
  onStream: (s: MediaStream) => void
  onStatus: (s: RTCPeerConnectionState | 'connecting') => void
  onLatency: (ms: number) => void
  onPerf?: (p: ClientPerf) => void
}

/** WebRTC session with the Isaac Sim server: video (recvonly) + two DataChannels. */
export class TeleopConnection {
  private pc: RTCPeerConnection | null = null
  private teleop: RTCDataChannel | null = null
  private events: RTCDataChannel | null = null
  private pingTimer: number | null = null
  private statsTimer: number | null = null
  private pending = new Map<string, { resolve: (v: unknown) => void; reject: (e: Error) => void }>()
  private seq = 0
  peerId: string | null = null
  private listener: Listener

  constructor(listener: Listener) {
    this.listener = listener
  }

  async connect(): Promise<void> {
    this.close()
    resetFrameMeta()
    resetStats()
    this.listener.onStatus('connecting')
    // LAN only: no STUN/TURN needed (host candidates)
    const pc = new RTCPeerConnection({ iceServers: [] })
    this.pc = pc
    pc.addTransceiver('video', { direction: 'recvonly' })
    // XR samples: latest-wins, never retransmit stale poses
    this.teleop = pc.createDataChannel('teleop', { ordered: false, maxRetransmits: 0 })
    this.events = pc.createDataChannel('events')
    this.events.onmessage = (ev) => this.onEvent(ev.data)
    this.events.onopen = () => {
      this.pingTimer = window.setInterval(() => this.ping(), 2000)
      this.statsTimer = window.setInterval(() => void this.reportStats(), 1000)
    }
    pc.ontrack = (ev) => {
      const stream = ev.streams[0] ?? new MediaStream([ev.track])
      this.listener.onStream(stream)
    }
    pc.onconnectionstatechange = () => this.listener.onStatus(pc.connectionState)

    await pc.setLocalDescription(await pc.createOffer())
    await waitForIce(pc, 2500)
    const answer = await api.offer(pc.localDescription!.sdp)
    this.peerId = answer.peerId
    await pc.setRemoteDescription({ type: answer.type, sdp: answer.sdp })
  }

  private onEvent(data: string) {
    let msg: { t?: string; id?: string; ts?: number; ok?: boolean; result?: unknown; error?: string }
    try {
      msg = JSON.parse(data)
    } catch {
      return
    }
    if (msg.t === 'state') this.listener.onState(msg as unknown as SimState)
    else if (msg.t === 'fmeta') pushFrameMeta(msg as unknown as FrameMeta)
    else if (msg.t === 'pong' && msg.ts) {
      this.listener.onLatency(performance.now() - msg.ts)
      const server = (msg as { server?: number }).server
      if (typeof server === 'number') onPong(msg.ts, server)
    }
    else if (msg.t === 'cmd_result' && msg.id) {
      const p = this.pending.get(msg.id)
      if (p) {
        this.pending.delete(msg.id)
        if (msg.ok) p.resolve(msg.result)
        else p.reject(new Error(msg.error ?? 'command failed'))
      }
    }
  }

  private async reportStats() {
    const pc = this.pc
    if (!pc) return
    try {
      const p = await sampleStats(pc)
      this.listener.onPerf?.(p)
      if (this.events?.readyState === 'open') this.events.send(JSON.stringify({ t: 'perf', data: perfForServer(p) }))
    } catch {
      /* stats are best effort */
    }
  }

  private ping() {
    if (this.events?.readyState === 'open') {
      this.events.send(JSON.stringify({ t: 'ping', id: String(++this.seq), ts: performance.now() }))
    }
  }

  /** "vmc": a VMC Protocol (OSC) bundle (binary), "json": the JSON message. The server accepts either. */
  sendXR(sample: XRSample, format: MotionFormat = 'vmc') {
    const ch = this.teleop
    // Drop samples if the channel is backed up (keeps latency low)
    if (!ch || ch.readyState !== 'open' || ch.bufferedAmount >= 16 * 1024) return
    if (format === 'vmc') ch.send(encodeVmc(sample) as Uint8Array<ArrayBuffer>)
    else ch.send(JSON.stringify(sample))
  }

  takeControl() {
    if (this.events?.readyState === 'open') this.events.send(JSON.stringify({ t: 'take_control' }))
  }

  /** Command through the reliable DataChannel (low latency, used by in-VR controls). */
  command<T = unknown>(name: string, payload: object = {}): Promise<T> {
    const ch = this.events
    if (!ch || ch.readyState !== 'open') return api.command<T>(name, payload)
    const id = `c${++this.seq}`
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject })
      ch.send(JSON.stringify({ t: 'cmd', id, name, payload }))
      window.setTimeout(() => {
        if (this.pending.delete(id)) reject(new Error('command timeout'))
      }, 15000)
    })
  }

  get connected() {
    return this.pc?.connectionState === 'connected'
  }

  close() {
    if (this.pingTimer) window.clearInterval(this.pingTimer)
    this.pingTimer = null
    if (this.statsTimer) window.clearInterval(this.statsTimer)
    this.statsTimer = null
    this.teleop?.close()
    this.events?.close()
    this.pc?.close()
    this.pc = null
    this.teleop = null
    this.events = null
    for (const p of this.pending.values()) p.reject(new Error('connection closed'))
    this.pending.clear()
  }
}

function waitForIce(pc: RTCPeerConnection, timeoutMs: number): Promise<void> {
  if (pc.iceGatheringState === 'complete') return Promise.resolve()
  return new Promise((resolve) => {
    const t = window.setTimeout(done, timeoutMs)
    function done() {
      window.clearTimeout(t)
      pc.removeEventListener('icegatheringstatechange', check)
      resolve()
    }
    function check() {
      if (pc.iceGatheringState === 'complete') done()
    }
    pc.addEventListener('icegatheringstatechange', check)
  })
}
