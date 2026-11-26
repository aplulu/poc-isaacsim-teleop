import { useFrame } from '@react-three/fiber'
import { XRSpace, useXRInputSourceState } from '@react-three/xr'
import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { useApp } from '../store'
import type { MappingMode, RotationMode, TaskTemplate } from '../types'
import { IMMERSIVE_PRESETS, SCREEN_PRESETS, STEREO_PRESETS } from '../components/ControlPanel'
import { immersiveStatus } from './ImmersiveView'
import { LabelTexture } from './labelTexture'
import { SCREEN_HEIGHT, stereoRender } from './VideoScreen'
import { uiHover } from './uiHover'
import { xrStore } from './xrStore'

function Button({
  label,
  position,
  onClick,
  active = false,
  width = 0.2,
}: {
  label: string
  position: [number, number, number]
  onClick: () => void
  active?: boolean
  width?: number
}) {
  const [hover, setHover] = useState(false)
  // canvas width follows the button width so wide buttons keep the same height and font size
  const tex = useMemo(() => new LabelTexture(Math.round((320 * width) / 0.2), 112), [width])
  useEffect(() => () => tex.dispose(), [tex])
  useEffect(() => () => uiHover.clear(), [])
  tex.draw([label], {
    bg: active ? '#1f6feb' : hover ? '#30363d' : 'rgba(22,27,34,0.92)',
    border: hover ? '#58a6ff' : '#30363d',
    size: 40,
  })
  return (
    <mesh
      position={position}
      onClick={(e) => {
        e.stopPropagation()
        onClick()
      }}
      onPointerOver={(e) => {
        setHover(true)
        uiHover.enter(e.pointerId, e.pointerType)
      }}
      onPointerOut={(e) => {
        setHover(false)
        uiHover.leave(e.pointerId)
      }}
    >
      <planeGeometry args={[width, 0.2 * (112 / 320)]} />
      <meshBasicMaterial map={tex.texture} transparent toneMapped={false} />
    </mesh>
  )
}

const OBJECT_LABEL: Record<string, string> = {
  random: 'ランダム',
  apple_red: '赤りんご',
  banana: 'バナナ',
  socket: 'ソケット',
  pulley: 'プーリー',
  relay: 'リレー',
  injector: 'インジェクター',
  cube: 'キューブ',
  canned_peach: '桃缶',
  tray: 'トレイ',
  golf_ball: 'ゴルフボール',
  candle: 'キャンドル',
  rubiks_cube: 'ルービックキューブ',
  tape_roll: 'テープ',
  screwdriver: 'ドライバー',
  torx_driver: 'トルクスドライバー',
  extension_bar: 'エクステンションバー',
  socket_small: 'ソケット小',
  mega_fuse: 'ヒューズ大',
  midi_fuse: 'ヒューズ中',
  micro_fuse: 'ヒューズ小',
  bolt_m20: 'ボルトM20',
  nut_m20: 'ナットM20',
  bolt_m16: 'ボルトM16',
  nut_m16: 'ナットM16',
  bearing: 'ベアリング',
  bearing_small: 'ベアリング小',
  valve_lifter: 'バルブリフター',
  tape: 'テープ',
  coffee_cup: 'カップ',
}

function objectLabel(id: string, tmpl?: TaskTemplate): string {
  if (tmpl?.id === 'free') return '(自由入力)'
  return OBJECT_LABEL[id] ?? tmpl?.objects.find((o) => o.id === id)?.name ?? id
}

const MAPPING_LABEL: Record<MappingMode, string> = { camera: '表示カメラ', operator: 'オペレータ視点', robot: 'ロボット' }
const NEXT_MAPPING: Record<MappingMode, MappingMode> = { camera: 'operator', operator: 'robot', robot: 'camera' }

/** Compact status line on top of the hand menu (the only status display in the free-viewpoint view). */
function MenuStatus() {
  const tex = useMemo(() => new LabelTexture(1024, 72), [])
  useEffect(() => () => tex.dispose(), [tex])
  useFrame(() => {
    const s = useApp.getState()
    const warn = statusWarning()
    const imm = s.sim?.view === 'immersive'
    const parts = [
      warn,
      imm ? `没入表示${immersiveStatus.synced ? '' : ' (同期待ち)'}` : '',
      s.sim?.clutch ? '● 操作中' : '',
      s.sim?.handGrab?.left ? `左手: ${OBJECT_LABEL[s.sim.handGrab.left] ?? s.sim.handGrab.left}` : '',
      s.sim?.handGrab?.right ? `右手: ${OBJECT_LABEL[s.sim.handGrab.right] ?? s.sim.handGrab.right}` : '',
      perfLabel(),
      s.latencyMs != null ? `RTT ${s.latencyMs.toFixed(0)} ms` : '',
    ]
    tex.draw([parts.filter(Boolean).join('   ')], { size: 30, bg: warn ? 'rgba(110,70,0,0.9)' : 'rgba(13,17,23,0.8)' })
  })
  return (
    <mesh position={[0, 0.085, 0]}>
      <planeGeometry args={[0.86, 0.86 * (72 / 1024)]} />
      <meshBasicMaterial map={tex.texture} transparent toneMapped={false} />
    </mesh>
  )
}

/** sim fps / video fps shown, plus head motion -> display latency in the immersive view. */
function perfLabel(): string {
  const s = useApp.getState()
  const cp = s.clientPerf
  const base = `sim ${s.sim?.fps ?? 0} / 映像 ${cp?.fps ?? 0} fps`
  return cp?.poseToShowMs != null && s.sim?.view === 'immersive' ? `${base}  遅延 ${cp.poseToShowMs.toFixed(0)} ms` : base
}

function statusWarning(): string {
  const sm = useApp.getState().sim
  const inp = sm?.input
  if (!sm) return '未接続'
  if (sm.controller === false) return '⚠ 操作権なし'
  if (inp?.age == null || inp.age > 1) return '⚠ コントローラ入力なし'
  if (!inp.hand) return `⚠ ${sm.settings.hand === 'right' ? '右' : '左'}手のコントローラが見えません`
  return ''
}

const ROTATION_LABEL: Record<RotationMode, string> = { yaw: '回転: 水平のみ', full: '回転: 全軸', off: '回転: OFF' }
const NEXT_ROTATION: Record<RotationMode, RotationMode> = { yaw: 'full', full: 'off', off: 'yaw' }

/** Status bar + current task, drawn just above the video screen (out of the way of the image). */
export function ScreenStatus() {
  const distance = useApp((s) => s.screenDistance)
  const hfov = useApp((s) => s.sim?.hfov ?? 73.6)
  const status = useMemo(() => new LabelTexture(1024, 72), [])
  useEffect(() => () => status.dispose(), [status])

  useFrame(() => {
    const s = useApp.getState()
    const sm = s.sim
    const rec = sm?.recording
    const recLabel =
      rec && rec.state !== 'idle'
        ? `${rec.state === 'recording' ? '● REC' : rec.state === 'armed' ? '◎ 操作待ち' : '… 準備'} ${rec.duration.toFixed(1)}s`
        : ''
    const warn = statusWarning()
    const parts = [
      warn,
      recLabel,
      rec?.task?.text && rec.state !== 'idle' ? rec.task.text : '',
      sm?.clutch ? '● 操作中' : '',
      `グリッパ ${sm?.gripper.closed ? '閉' : '開'}${sm?.gripper.holding ? ` (${sm.gripper.holding})` : ''}`,
      `${sm?.camera ?? '-'}`,
      stereoRender.active
        ? `立体視 ${stereoRender.frameW}x${stereoRender.frameH}${s.swapEyes ? ' (左右入替)' : ''}`
        : sm?.stereo
          ? '立体視 待機中'
          : 'モノラル',
      perfLabel(),
      s.latencyMs != null ? `RTT ${s.latencyMs.toFixed(0)} ms` : '',
      s.busy ? `… ${s.busy}` : '',
    ]
    status.draw([parts.filter(Boolean).join('   ')], {
      size: 30,
      bg: warn ? 'rgba(110,70,0,0.9)' : rec?.state === 'recording' ? 'rgba(90,20,20,0.85)' : 'rgba(13,17,23,0.75)',
    })
  })

  const aspect = 16 / 9 // mono 1280x720 and stereo 960x540 per eye
  const width = 2 * distance * Math.tan(((hfov * Math.PI) / 180) / 2)
  const height = width / aspect
  const barW = Math.min(width, 1.6)
  return (
    <mesh position={[0, SCREEN_HEIGHT + height / 2 + 0.06, -distance]}>
      <planeGeometry args={[barW, barW * (72 / 1024)]} />
      <meshBasicMaterial map={status.texture} transparent toneMapped={false} />
    </mesh>
  )
}

/** The button grid. Its own coordinates are ~0.9 m wide; the parent scales it. */
function MenuButtons() {
  const sim = useApp((s) => s.sim)
  const app = useApp.getState
  const recState = sim?.recording?.state ?? 'idle'
  const rotation = sim?.settings.rotationMode ?? 'yaw'
  const swapEyes = useApp((s) => s.swapEyes)
  const immersive = sim?.view === 'immersive'
  const mapping = sim?.settings.mapping ?? 'camera'
  const templates = useApp((s) => s.templates)
  const sets = (sim?.objectSets ?? []).filter((x) => x.count > 0) // sets without models are skipped
  const spec = useApp((s) => s.taskSpec)
  const tmpl = templates.find((t) => t.id === spec.id) ?? templates[0]
  return (
    <group>
      <Button label="リセット" position={[-0.33, 0, 0]} onClick={() => void app().reset()} />
      <Button label="ランダム化" position={[-0.11, 0, 0]} onClick={() => void app().randomize()} />
      <Button label="カメラ切替" position={[0.11, 0, 0]} onClick={() => void app().cycleCamera()} />
      <Button
        label={sim?.gripper.closed ? 'グリッパ開' : 'グリッパ閉'}
        position={[0.33, 0, 0]}
        active={!!sim?.gripper.closed}
        onClick={() => void app().setGripper()}
      />
      <Button label="ホーム" position={[-0.33, -0.085, 0]} onClick={() => void app().command('go_home', {}, 'ホーム')} />
      <Button
        label={ROTATION_LABEL[rotation]}
        position={[-0.11, -0.085, 0]}
        active={rotation !== 'off'}
        onClick={() => void app().setTeleop({ rotationMode: NEXT_ROTATION[rotation] })}
      />
      <Button
        label={`倍率 x${(sim?.settings.positionScale ?? 1).toFixed(1)}`}
        position={[0.11, -0.085, 0]}
        onClick={() => {
          const cur = sim?.settings.positionScale ?? 1
          const next = cur >= 2 ? 0.5 : cur >= 1 ? 2 : 1
          void app().setTeleop({ positionScale: next })
        }}
      />
      <Button
        label={sim?.mirror ? '鏡: ON' : '鏡: OFF'}
        position={[0.33, -0.085, 0]}
        active={!!sim?.mirror}
        onClick={() => void app().setMirror(!sim?.mirror)}
      />
      {/* row 3: view */}
      <Button
        label={immersive ? '表示: 没入' : '表示: モニター'}
        position={[-0.33, -0.17, 0]}
        active={immersive}
        onClick={() => void app().setView(immersive ? 'screen' : 'immersive')}
      />
      <Button label="正面リセット" position={[-0.11, -0.17, 0]} onClick={() => void app().recenter()} />
      {immersive ? (
        <Button
          label={sim?.settings.handMode === 'grab' ? '手: 両手で掴む' : '手: ロボ+掴む'}
          position={[0.11, -0.17, 0]}
          active={sim?.settings.handMode === 'grab'}
          onClick={() => void app().setTeleop({ handMode: sim?.settings.handMode === 'grab' ? 'robot' : 'grab' })}
        />
      ) : (
        <Button
          label={sim?.stereo ? '立体視: ON' : '立体視: OFF'}
          position={[0.11, -0.17, 0]}
          active={!!sim?.stereo}
          onClick={() => void app().setStereo(!sim?.stereo)}
        />
      )}
      <Button label="左右入替" position={[0.33, -0.17, 0]} active={swapEyes} onClick={() => app().toggleSwapEyes()} />
      {/* row 4: recording + exit */}
      {sim?.recording && (
        <>
          <Button
            label={recState === 'idle' ? '● 録画開始' : recState === 'recording' ? '成功で保存' : '準備中…'}
            position={[-0.33, -0.255, 0]}
            active={recState === 'recording'}
            onClick={() => void app().recToggle()}
          />
          <Button label="失敗で終了" position={[-0.11, -0.255, 0]} onClick={() => void app().recStop(false)} />
          <Button label="破棄" position={[0.11, -0.255, 0]} onClick={() => void app().recDiscard()} />
        </>
      )}
      <Button label="VR終了" position={[0.33, -0.255, 0]} onClick={() => void xrStore.getState().session?.end()} />
      {/* row 5: hand -> arm direction (screen view only; the free viewpoint is always exact) */}
      <Button
        label={`セット: ${sets.find((x) => x.id === sim?.objectSet)?.label ?? '-'}`}
        position={[0.22, -0.34, 0]}
        width={0.42}
        onClick={() => {
          if (recState !== 'idle' || sets.length === 0) return
          const i = sets.findIndex((x) => x.id === sim?.objectSet)
          void app().setObjectSet(sets[(i + 1) % sets.length].id)
        }}
      />
      {!immersive && (
        <Button
          label={`基準: ${MAPPING_LABEL[mapping]}`}
          position={[-0.33, -0.34, 0]}
          onClick={() => void app().setTeleop({ mapping: NEXT_MAPPING[mapping] })}
        />
      )}
      {immersive && sim?.avatar && (
        <Button
          label={sim.avatar.visible ? 'アバター: ON' : 'アバター: OFF'}
          position={[-0.33, -0.34, 0]}
          active={sim.avatar.visible}
          onClick={() => void app().setAvatar({ visible: !sim.avatar?.visible })}
        />
      )}
      {sim?.resolution && (
        <Button
          label={resolutionLabel(sim.resolution, immersive, !!sim.stereo)}
          position={[-0.11, -0.34, 0]}
          onClick={() => {
            const r = sim.resolution
            if (!r) return
            if (immersive) {
              const i = IMMERSIVE_PRESETS.indexOf(r.immersive)
              void app().setResolution({ immersive: IMMERSIVE_PRESETS[(i + 1) % IMMERSIVE_PRESETS.length] })
            } else {
              const list = sim.stereo ? STEREO_PRESETS : SCREEN_PRESETS
              const cur = sim.stereo ? r.stereo : r.screen
              const i = list.findIndex((v) => v[0] === cur[0] && v[1] === cur[1])
              const next = list[(i + 1) % list.length]
              void app().setResolution(sim.stereo ? { stereo: next } : { screen: next })
            }
          }}
        />
      )}
      {/* row 6: task and target object for recording (cycled; locked while an episode is in progress) */}
      {sim?.recording && templates.length > 0 && (
        <>
          <Button
            label={`タスク: ${tmpl?.label.replace('◯◯', '…') ?? '-'}`}
            position={[-0.22, -0.425, 0]}
            width={0.42}
            onClick={() => {
              if (recState !== 'idle') return
              const i = templates.findIndex((t) => t.id === spec.id)
              const next = templates[(i + 1) % templates.length]
              app().setTaskSpec({ id: next.id, object: 'random', text: spec.text })
            }}
          />
          <Button
            label={`対象: ${objectLabel(spec.object ?? 'random', tmpl)}`}
            position={[0.22, -0.425, 0]}
            width={0.42}
            onClick={() => {
              if (recState !== 'idle' || !tmpl || tmpl.objects.length === 0) return
              const options = ['random', ...tmpl.objects.map((o) => o.id)]
              const i = options.indexOf(spec.object ?? 'random')
              app().setTaskSpec({ ...spec, object: options[(i + 1) % options.length] })
            }}
          />
        </>
      )}
      <MenuStatus />
    </group>
  )
}

/** Menu following the non-controlling hand (like a wrist palette): look at your hand to use it. */
function HandAttached({ children }: { children: ReactNode }) {
  const controlHand = useApp((s) => s.sim?.settings.hand ?? 'right')
  const other = controlHand === 'right' ? 'left' : 'right'
  const controller = useXRInputSourceState('controller', other)
  const space = controller?.inputSource.gripSpace
  if (!space) return null
  // ~35 cm wide, floating just above the controller and tilted towards the face
  return (
    <XRSpace space={space}>
      <group position={[0, 0.09, -0.03]} rotation={[-0.75, 0, 0]} scale={0.38}>
        {children}
      </group>
    </XRSpace>
  )
}

/**
 * In-VR controls. Placement (cycled with the non-controlling hand's Y/B button or the 2D panel):
 *   hand   - attached to the other controller (default; never blocks the view)
 *   fixed  - low on the left side of the user
 *   hidden - no menu (status bar above the screen remains)
 */
export function VRPanel() {
  const mode = useApp((s) => s.vrMenu)
  if (mode === 'hidden') return null
  if (mode === 'hand') {
    return (
      <HandAttached>
        <MenuButtons />
      </HandAttached>
    )
  }
  return (
    <group position={[-0.62, 0.95, -0.45]} rotation={[-0.55, 0.85, 0.3]} scale={0.6}>
      <MenuButtons />
    </group>
  )
}

function resolutionLabel(r: { screen: [number, number]; stereo: [number, number]; immersive: number }, immersive: boolean, stereo: boolean) {
  if (immersive) return `解像度 ${r.immersive}`
  const v = stereo ? r.stereo : r.screen
  return `解像度 ${v[1]}p`
}
