export interface ServerInfo {
  name: string
  robot: string
  codeLength: number
  pairingTtl: number
  video: { width: number; height: number; fps: number; stereo: boolean }
  /** How to send the XR samples (server env TELEOP_MOTION_FORMAT; missing on older servers, which read JSON only) */
  motionFormat?: MotionFormat
  serverTime: number
}

export interface PairRequestResponse {
  requestId: string
  expiresAt: number
  expiresIn: number
}

export interface PairConfirmResponse {
  token: string
  tokenType: 'Bearer'
  expiresAt: number
  sessionId: string
  deviceName: string
}

export type CameraName = 'operator' | 'front' | 'top' | 'wrist' | 'dataset'
export type Hand = 'left' | 'right'
export type RotationMode = 'yaw' | 'full' | 'off'
export type VRMenuMode = 'hand' | 'fixed' | 'hidden'
export type MappingMode = 'camera' | 'operator' | 'robot'
export type ViewMode = 'screen' | 'immersive'

export interface TeleopSettings {
  positionScale: number
  rotationMode: RotationMode
  smoothing: number
  mapping: MappingMode
  graspAssist: boolean
  hand: Hand
  /** free viewpoint: "robot" = control hand drives the robot, other hand grabs; "grab" = both hands grab */
  handMode: HandMode
}

export type HandMode = 'robot' | 'grab'

export interface StreamResolution {
  screen: [number, number]
  stereo: [number, number]
  immersive: number
  bitrate: number
  bitrateAuto?: boolean
  dlss?: DlssMode | string
}

export type DlssMode = 'performance' | 'balanced' | 'quality' | 'auto' | 'native'

export interface ServerPerf {
  sections: Record<string, { avg: number; p95: number }>
  rates: Record<string, number>
  encoder?: string
  client?: Record<string, number | string | null>
}

export interface SimState {
  ready: boolean
  playing: boolean
  simTime: number
  fps: number
  camera: CameraName
  cameras: CameraName[]
  settings: TeleopSettings
  clutch: boolean
  input?: { age: number | null; hand: boolean; grip: number | null; trigger: number | null }
  gripper: { closed: boolean; holding: string | null; position?: number; type?: 'robotiq_2f85' | 'virtual' }
  mirror?: boolean
  avatar?: AvatarStatus | null
  handGrab?: { left: string | null; right: string | null }
  resolution?: StreamResolution
  perf?: ServerPerf
  recording?: RecStatus | null
  props?: { name: string; p: [number, number, number] }[]
  stereo: boolean
  ipd?: number
  view?: ViewMode
  objectSet?: string
  objectSets?: { id: string; label: string; count: number }[]
  immersive?: { fov: number; size: [number, number]; anchor: [number, number, number] } | null
  joints?: number[]
  jointNames?: string[]
  tcp?: { p: [number, number, number]; q: [number, number, number, number] }
  targetError?: number
  hfov?: number
  controller?: boolean
  peerId?: string
}

export interface DrConfig {
  randomizeOnReset: boolean
  autoInterval: number
  seed: number | null
  objects: { enabled: boolean; randomYaw: boolean; minSpacing: number }
  lighting: {
    enabled: boolean
    intensityScaleMin: number
    intensityScaleMax: number
    colorTempMin: number
    colorTempMax: number
    randomDirection: boolean
  }
  materials: { enabled: boolean; table: boolean; floor: boolean }
  camera: { enabled: boolean; positionJitter: number; rotationJitterDeg: number }
  mirror: { enabled: boolean; probability: number }
  physics: {
    enabled: boolean
    massScaleMin: number
    massScaleMax: number
    frictionMin: number
    frictionMax: number
  }
}

export type DrTarget = 'objects' | 'lighting' | 'materials' | 'camera' | 'mirror' | 'physics'

export interface XRHandSample {
  p: [number, number, number]
  q: [number, number, number, number]
  grip: number
  trigger: number
  axes?: [number, number]
}

/** How the XR samples go to the server: JSON text or VMC Protocol (OSC) bundles. */
export type MotionFormat = 'json' | 'vmc'

export interface XRSample {
  t: 'xr'
  ts: number
  right?: XRHandSample
  left?: XRHandSample
  head?: { p: [number, number, number]; q: [number, number, number, number] }
}

export interface TaskTemplate {
  id: string
  label: string
  template: string
  objects: { id: string; name: string }[]
  autoSuccess: boolean
}

export interface TaskSpec {
  id: string
  object?: string
  text?: string
}

export interface RecSettings {
  waitForMotion: boolean
  saveFailed: boolean
  autoNext: boolean
  autoSuccess: boolean
  maxDuration: number
  randomize: boolean
}

export interface RecStatus {
  state: 'idle' | 'preparing' | 'armed' | 'recording'
  task: { id: string; object: string | null; text: string; autoSuccess: boolean } | null
  episode: number | null
  frames: number
  duration: number
  message: string
  settings: RecSettings
  dataset: {
    root: string
    episodes: number
    frames: number
    tasks: number
    fps: number
    codec: string
    pending: number
    lastError: string | null
    recent: { episode: number; task: string; length: number; success: boolean; duration: number }[]
  }
}

export type AvatarStatus = {
  source: string
  /** requested by the user */
  visible: boolean
  /** shown right now (false while hidden for recording) */
  shown: boolean
  scale: number
  autoScale: boolean
  ok: boolean
  hideWhileRecording: boolean
  collision: 'on' | 'off'
  collisionActive: boolean
  bodies: number
}
