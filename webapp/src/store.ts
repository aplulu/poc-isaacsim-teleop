import type { ClientPerf } from './perfStats'
import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { ApiError, api, setAuthToken, setUnauthorizedHandler } from './api'
import type {
  CameraName,
  DrConfig,
  DrTarget,
  RecSettings,
  ServerInfo,
  SimState,
  TaskSpec,
  TaskTemplate,
  TeleopSettings,
  VRMenuMode,
  ViewMode,
} from './types'
import { TeleopConnection } from './webrtc'

export type PairingPhase = 'idle' | 'requesting' | 'awaitingCode' | 'confirming' | 'paired'
export type ConnStatus = RTCPeerConnectionState | 'idle' | 'connecting'

interface Toast {
  id: number
  kind: 'info' | 'error'
  text: string
}

interface AppState {
  // persisted
  token: string | null
  tokenExpiresAt: number | null
  deviceName: string
  deviceNameEdited: boolean
  // pairing
  phase: PairingPhase
  requestId: string | null
  requestExpiresAt: number | null
  pairError: string | null
  // server / connection
  info: ServerInfo | null
  conn: ConnStatus
  stream: MediaStream | null
  latencyMs: number | null
  clientPerf: ClientPerf | null
  sim: SimState | null
  dr: DrConfig | null
  busy: string | null
  toasts: Toast[]
  inXR: boolean
  screenDistance: number
  templates: TaskTemplate[]
  taskSpec: TaskSpec
  vrMenu: VRMenuMode
  swapEyes: boolean

  loadInfo: () => Promise<void>
  setDeviceName: (n: string) => void
  requestPairing: () => Promise<void>
  confirmPairing: (code: string) => Promise<void>
  cancelPairing: () => void
  logout: () => Promise<void>
  connect: () => Promise<void>
  disconnect: () => void
  command: <T = unknown>(name: string, payload?: object, label?: string) => Promise<T | undefined>
  reset: (opts?: { randomize?: boolean; restoreAppearance?: boolean }) => Promise<void>
  randomize: (targets?: DrTarget[]) => Promise<void>
  loadDr: () => Promise<void>
  saveDr: (patch: Partial<DrConfig> | DrConfig) => Promise<void>
  setCamera: (c: CameraName) => Promise<void>
  cycleCamera: () => Promise<void>
  setGripper: (closed?: boolean) => Promise<void>
  setTeleop: (s: Partial<TeleopSettings>) => Promise<void>
  toast: (text: string, kind?: Toast['kind']) => void
  setInXR: (v: boolean) => void
  setScreenDistance: (d: number) => void
  loadTemplates: () => Promise<void>
  setTaskSpec: (t: TaskSpec) => void
  recStart: () => Promise<void>
  recStop: (success: boolean) => Promise<void>
  recDiscard: () => Promise<void>
  recToggle: () => Promise<void>
  recSettings: (s: Partial<RecSettings>) => Promise<void>
  recDeleteLast: () => Promise<void>
  setMirror: (visible: boolean) => Promise<void>
  setAvatar: (opts: { visible?: boolean; hideWhileRecording?: boolean; collision?: 'on' | 'off' }) => Promise<void>
  setResolution: (r: {
    screen?: [number, number]
    stereo?: [number, number]
    immersive?: number
    bitrate?: number | 'auto'
    dlss?: string
  }) => Promise<void>
  setStereo: (enabled: boolean) => Promise<void>
  toggleSwapEyes: () => void
  setView: (mode: ViewMode, recenter?: boolean) => Promise<void>
  recenter: () => Promise<void>
  setObjectSet: (set: string) => Promise<void>
  sendIpd: (ipd: number) => void
  setVrMenu: (m: VRMenuMode) => void
  cycleVrMenu: () => void
}

let connection: TeleopConnection | null = null
let connecting: Promise<void> | null = null
export const getConnection = () => connection
let toastSeq = 0

/**
 * Default device name from the user agent. The Meta Quest browser reports the model, e.g.
 * "Mozilla/5.0 (X11; Linux x86_64; Quest 3) ... OculusBrowser/37.0 ...".
 */
export function defaultDeviceName(ua = navigator.userAgent): string {
  // "Quest 3", "Quest 3S", "Quest 2", "Quest Pro" (older firmware may report just "Quest")
  const quest = ua.match(/\bQuest(?:\s*(3S|3|2|Pro))?\b/i)
  if (quest) {
    const model = quest[1] ? ` ${quest[1].toUpperCase() === 'PRO' ? 'Pro' : quest[1].toUpperCase()}` : ''
    return `Meta Quest${model}`
  }
  if (/OculusBrowser/i.test(ua)) return 'Meta Quest'
  const pico = ua.match(/\bPICO\s*(4\s*Ultra|4|Neo\s*\d)\b/i)
  if (pico) return `PICO ${pico[1]}`
  if (/Pico/i.test(ua)) return 'PICO'
  if (/iPad/i.test(ua)) return 'iPad'
  if (/iPhone/i.test(ua)) return 'iPhone'
  if (/Android/i.test(ua)) return 'Android'
  if (/Windows/i.test(ua)) return 'Windows PC'
  if (/Macintosh/i.test(ua)) return 'Mac'
  if (/Linux/i.test(ua)) return 'Linux PC'
  return 'Browser'
}

export const useApp = create<AppState>()(
  persist(
    (set, get) => ({
      token: null,
      tokenExpiresAt: null,
      deviceName: defaultDeviceName(),
      deviceNameEdited: false,
      phase: 'idle',
      requestId: null,
      requestExpiresAt: null,
      pairError: null,
      info: null,
      conn: 'idle',
      stream: null,
      latencyMs: null,
      clientPerf: null,
      sim: null,
      dr: null,
      busy: null,
      toasts: [],
      inXR: false,
      screenDistance: 1.6,
      templates: [],
      taskSpec: { id: 'put_in_tray', object: 'random' },
      vrMenu: 'hand',
      swapEyes: false,

      loadInfo: async () => {
        try {
          set({ info: await api.info() })
        } catch (e) {
          get().toast(`サーバに接続できません: ${(e as Error).message}`, 'error')
        }
      },

      setDeviceName: (deviceName) => set({ deviceName, deviceNameEdited: deviceName.trim() !== '' }),

      requestPairing: async () => {
        set({ phase: 'requesting', pairError: null })
        try {
          const r = await api.pairRequest(get().deviceName.trim() || defaultDeviceName())
          set({ phase: 'awaitingCode', requestId: r.requestId, requestExpiresAt: Date.now() + r.expiresIn * 1000 })
        } catch (e) {
          set({ phase: 'idle', pairError: (e as Error).message })
        }
      },

      confirmPairing: async (code) => {
        const { requestId } = get()
        if (!requestId) return
        set({ phase: 'confirming', pairError: null })
        try {
          const r = await api.pairConfirm(requestId, code)
          setAuthToken(r.token)
          set({
            token: r.token,
            tokenExpiresAt: r.expiresAt * 1000,
            phase: 'paired',
            requestId: null,
            requestExpiresAt: null,
          })
          await get().connect()
        } catch (e) {
          const err = e as ApiError
          const restart = err.code === 'locked' || err.code === 'not_found'
          set({
            phase: restart ? 'idle' : 'awaitingCode',
            requestId: restart ? null : requestId,
            pairError: err.message,
          })
        }
      },

      cancelPairing: () => set({ phase: 'idle', requestId: null, requestExpiresAt: null, pairError: null }),

      logout: async () => {
        try {
          if (get().token) await api.logout()
        } catch {
          /* token may already be invalid */
        }
        get().disconnect()
        setAuthToken(null)
        set({ token: null, tokenExpiresAt: null, phase: 'idle', sim: null, dr: null })
      },

      connect: async () => {
        if (!get().token) return
        if (connecting) return connecting // a connection attempt is already running
        connection?.close()
        // callbacks of a replaced connection must not touch the state (they could trigger reconnect loops)
        const c: TeleopConnection = new TeleopConnection({
          onState: (sim) => connection === c && set({ sim }),
          onStream: (stream) => connection === c && set({ stream }),
          onStatus: (conn) => connection === c && set({ conn }),
          onLatency: (latencyMs) => connection === c && set({ latencyMs }),
          onPerf: (clientPerf) => connection === c && set({ clientPerf }),
        })
        connection = c
        let done!: () => void
        connecting = new Promise<void>((r) => (done = r))
        try {
          await c.connect()
          void get().loadDr()
          void get().loadTemplates()
        } catch (e) {
          if (connection === c) set({ conn: 'failed' })
          get().toast(`WebRTC接続に失敗しました: ${(e as Error).message}`, 'error')
        } finally {
          connecting = null
          done()
        }
      },

      disconnect: () => {
        connection?.close()
        connection = null
        set({ conn: 'idle', stream: null, sim: null })
      },

      command: async <T,>(name: string, payload: object = {}, label?: string) => {
        set({ busy: label ?? name })
        try {
          const conn = connection
          return conn ? await conn.command<T>(name, payload) : await api.command<T>(name, payload)
        } catch (e) {
          get().toast(`${label ?? name}: ${(e as Error).message}`, 'error')
          return undefined
        } finally {
          set({ busy: null })
        }
      },

      reset: async (opts = {}) => {
        await get().command('reset', opts, 'リセット')
      },

      randomize: async (targets) => {
        const r = await get().command<{ applied: string[] }>('randomize', targets ? { targets } : {}, 'ランダマイズ')
        if (r) get().toast(`ランダマイズ: ${r.applied.join(', ') || '(なし)'}`)
      },

      loadDr: async () => {
        const dr = await get().command<DrConfig>('get_dr_config', {}, 'DR設定取得')
        if (dr) set({ dr })
      },

      saveDr: async (patch) => {
        const dr = await get().command<DrConfig>('set_dr_config', { config: patch }, 'DR設定保存')
        if (dr) set({ dr })
      },

      setCamera: async (camera) => {
        await get().command('set_camera', { camera }, 'カメラ切替')
      },

      cycleCamera: async () => {
        const sim = get().sim
        const cams = sim?.cameras ?? ['operator', 'front', 'top', 'wrist']
        const i = cams.indexOf(sim?.camera ?? 'operator')
        await get().setCamera(cams[(i + 1) % cams.length])
      },

      setGripper: async (closed) => {
        await get().command('set_gripper', closed === undefined ? {} : { closed }, 'グリッパ')
      },

      setTeleop: async (s) => {
        await get().command('set_teleop_settings', s, 'テレオペ設定')
      },

      toast: (text, kind = 'info') => {
        const id = ++toastSeq
        set({ toasts: [...get().toasts, { id, kind, text }].slice(-4) })
        window.setTimeout(() => set({ toasts: get().toasts.filter((t) => t.id !== id) }), 4500)
      },

      loadTemplates: async () => {
        try {
          // silently ignored when the server runs with --no-record
          const r = await api.command<{ templates: TaskTemplate[] }>('task_templates')
          set({ templates: r.templates })
        } catch {
          set({ templates: [] })
        }
      },

      setTaskSpec: (taskSpec) => set({ taskSpec }),

      recStart: async () => {
        await get().command('rec_start', { task: get().taskSpec }, '録画開始')
      },

      recStop: async (success) => {
        await get().command('rec_stop', { success }, success ? '成功で保存' : '失敗で終了')
      },

      recDiscard: async () => {
        await get().command('rec_discard', {}, '破棄')
      },

      recToggle: async () => {
        const st = get().sim?.recording?.state ?? 'idle'
        if (st === 'idle') await get().recStart()
        else if (st === 'recording') await get().recStop(true)
        else await get().recDiscard()
      },

      recSettings: async (s) => {
        await get().command('rec_settings', s, '録画設定')
      },

      recDeleteLast: async () => {
        const r = await get().command<{ deleted: number }>('rec_delete_last', {}, '最後のエピソード削除')
        if (r) get().toast(`エピソード ${r.deleted} を削除しました`)
      },

      setMirror: async (visible) => {
        await get().command('set_mirror', { visible }, '鏡')
      },

      setAvatar: async (opts) => {
        await get().command('set_avatar', opts, 'アバター')
      },

      setResolution: async (r) => {
        await get().command('set_resolution', r, '解像度')
      },

      setStereo: async (enabled) => {
        await get().command('set_stereo', { enabled }, enabled ? '立体視 ON' : '立体視 OFF')
      },

      toggleSwapEyes: () => set({ swapEyes: !get().swapEyes }),

      setView: async (mode, recenter = mode === 'immersive') => {
        await get().command('set_view', { mode, recenter }, mode === 'immersive' ? '没入表示' : 'モニター表示')
      },

      setObjectSet: async (objectSet) => {
        const r = await get().command('set_object_set', { set: objectSet }, '対象セット')
        if (r) {
          set({ taskSpec: { ...get().taskSpec, object: 'random' } })
          await get().loadTemplates()
        }
      },

      recenter: async () => {
        await get().command('set_view', { recenter: true }, '正面リセット')
      },

      sendIpd: (ipd) => {
        // the headset's measured eye distance becomes the stereo camera baseline (orthostereo)
        const conn = getConnection()
        if (conn) void conn.command('set_stereo', { ipd: Math.round(ipd * 10000) / 10000 }).catch(() => undefined)
      },

      setVrMenu: (vrMenu) => set({ vrMenu }),
      cycleVrMenu: () => {
        const order: VRMenuMode[] = ['hand', 'fixed', 'hidden']
        set({ vrMenu: order[(order.indexOf(get().vrMenu) + 1) % order.length] })
      },

      setInXR: (inXR) => set({ inXR }),
      setScreenDistance: (d) => set({ screenDistance: Math.min(4, Math.max(0.6, d)) }),
    }),
    {
      name: 'isaacsimvr',
      partialize: (s) => ({
        token: s.token,
        tokenExpiresAt: s.tokenExpiresAt,
        deviceName: s.deviceName,
        deviceNameEdited: s.deviceNameEdited,
        screenDistance: s.screenDistance,
        taskSpec: s.taskSpec,
        vrMenu: s.vrMenu,
        swapEyes: s.swapEyes,
      }),
      onRehydrateStorage: () => (state) => {
        if (!state) return
        if (!state.deviceNameEdited) state.deviceName = defaultDeviceName()
        if (state.token && state.tokenExpiresAt && state.tokenExpiresAt > Date.now()) {
          setAuthToken(state.token)
          state.phase = 'paired'
        } else {
          state.token = null
          state.tokenExpiresAt = null
        }
      },
    },
  ),
)

// Any 401 (expired / revoked / server restarted) drops back to the pairing screen
setUnauthorizedHandler(() => {
  const s = useApp.getState()
  if (!s.token) return
  s.disconnect()
  setAuthToken(null)
  useApp.setState({ token: null, tokenExpiresAt: null, phase: 'idle' })
  s.toast('認証の有効期限が切れたか、取り消されました。再度ペアリングしてください。', 'error')
})
