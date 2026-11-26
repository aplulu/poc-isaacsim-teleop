import { useEffect, useState, type ReactNode } from 'react'
import { getConnection, useApp } from '../store'
import type { CameraName, DrConfig, DrTarget, HandMode } from '../types'
import { motionFormat } from '../vmc'
import { xrStore } from '../xr/xrStore'
import { RecordingPanel } from './RecordingPanel'

const CAMERA_LABELS: Record<CameraName, string> = {
  operator: 'オペレータ',
  front: '正面',
  top: '真上',
  wrist: 'リスト',
  dataset: '学習用',
}

const DR_LABELS: Record<DrTarget, string> = {
  objects: '物体配置',
  lighting: '照明',
  materials: '素材/色',
  camera: 'カメラ揺らぎ',
  mirror: '鏡の有無',
  physics: '物理(質量/摩擦)',
}

function Section({ title, children, right }: { title: string; children: ReactNode; right?: ReactNode }) {
  return (
    <section className="section">
      <header>
        <h2>{title}</h2>
        {right}
      </header>
      {children}
    </section>
  )
}

function Num({
  label,
  value,
  onChange,
  step = 0.1,
  min,
  max,
}: {
  label: string
  value: number
  onChange: (v: number) => void
  step?: number
  min?: number
  max?: number
}) {
  return (
    <label className="num">
      <span>{label}</span>
      <input
        type="number"
        value={Number.isFinite(value) ? value : 0}
        step={step}
        min={min}
        max={max}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
    </label>
  )
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="toggle">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span>{label}</span>
    </label>
  )
}

function useXRSupported() {
  const [ok, setOk] = useState<boolean | null>(null)
  useEffect(() => {
    if (new URLSearchParams(window.location.search).has('xremulate')) return setOk(true)
    if (!navigator.xr) return setOk(false)
    navigator.xr.isSessionSupported('immersive-vr').then(setOk, () => setOk(false))
  }, [])
  return ok
}

export function ControlPanel() {
  const { sim, conn, latencyMs, busy, dr, deviceName, inXR, vrMenu, info } = useApp()
  const app = useApp.getState()
  const xrOk = useXRSupported()
  const [draft, setDraft] = useState<DrConfig | null>(dr)
  const [restoreAppearance, setRestoreAppearance] = useState(false)
  const [jog, setJog] = useState(0.02)

  useEffect(() => setDraft(dr), [dr])

  const upd = <K extends keyof DrConfig>(k: K, v: Partial<DrConfig[K]> | DrConfig[K]) =>
    setDraft((d) =>
      d ? { ...d, [k]: typeof v === 'object' && v !== null ? { ...(d[k] as object), ...v } : v } : d,
    )

  const connected = conn === 'connected'
  const move = (delta: [number, number, number]) => void app.command('move_tcp', { delta }, 'ジョグ')

  return (
    <div className="panel">
      <Section
        title="接続"
        right={<span className={`badge ${connected ? 'ok' : conn === 'connecting' ? 'warn' : 'bad'}`}>{conn}</span>}
      >
        <div className="kv">
          <span>デバイス</span>
          <b>{deviceName}</b>
          <span>RTT</span>
          <b>{latencyMs != null ? `${latencyMs.toFixed(0)} ms` : '-'}</b>
          <span>Sim FPS</span>
          <b>{sim?.fps ?? '-'}</b>
          <span>操作権</span>
          <b>{sim?.controller ? 'あり' : 'なし'}</b>
          <span>モーション送信</span>
          <b>{motionFormat(info) === 'vmc' ? 'VMC Protocol' : 'JSON'}</b>
        </div>
        <div className="row wrap">
          <button
            className="primary"
            disabled={!xrOk || !connected || inXR}
            title={xrOk === false ? 'このブラウザは WebXR (immersive-vr) に対応していません' : ''}
            onClick={() => void xrStore.enterVR()}
          >
            VR を開始
          </button>
          {!sim?.controller && connected && (
            <button onClick={() => getConnection()?.takeControl()}>操作権を取得</button>
          )}
          <button onClick={() => void app.connect()}>再接続</button>
          <button className="danger" onClick={() => void app.logout()}>
            ログアウト
          </button>
        </div>
        {xrOk === false && <p className="muted small">WebXR 非対応のため 2D プレビューのみ表示しています。</p>}
      </Section>

      <RecordingPanel />

      <Section title="ロボット" right={busy ? <span className="badge warn">{busy}…</span> : null}>
        <div className="row wrap">
          <button className="primary" onClick={() => void app.reset({ restoreAppearance })}>
            リセット
          </button>
          <button onClick={() => void app.reset({ randomize: true })}>リセット + ランダム化</button>
          <button onClick={() => void app.command('go_home', {}, 'ホーム')}>ホーム姿勢</button>
          <button className={sim?.gripper.closed ? 'active' : ''} onClick={() => void app.setGripper()}>
            グリッパ {sim?.gripper.closed ? '開く' : '閉じる'}
          </button>
        </div>
        <Toggle label="リセット時に見た目(DR)を元に戻す" checked={restoreAppearance} onChange={setRestoreAppearance} />
        <div className="jog">
          <span className="muted small">TCP ジョグ (world)</span>
          <div className="jog-grid">
            <button onClick={() => move([jog, 0, 0])}>+X</button>
            <button onClick={() => move([-jog, 0, 0])}>-X</button>
            <button onClick={() => move([0, jog, 0])}>+Y</button>
            <button onClick={() => move([0, -jog, 0])}>-Y</button>
            <button onClick={() => move([0, 0, jog])}>+Z</button>
            <button onClick={() => move([0, 0, -jog])}>-Z</button>
          </div>
          <select value={jog} onChange={(e) => setJog(parseFloat(e.target.value))}>
            {[0.005, 0.02, 0.05, 0.1].map((v) => (
              <option key={v} value={v}>
                {v * 100} cm
              </option>
            ))}
          </select>
        </div>
        <div className="kv mono">
          <span>TCP</span>
          <b>{sim?.tcp ? sim.tcp.p.map((v) => v.toFixed(3)).join(', ') : '-'}</b>
          <span>関節[deg]</span>
          <b>{sim?.joints ? sim.joints.map((v) => v.toFixed(1)).join(', ') : '-'}</b>
          <span>保持物</span>
          <b>{sim?.gripper.holding ?? '-'}</b>
          <span>グリッパ</span>
          <b>
            {sim?.gripper.type === 'robotiq_2f85' ? 'Robotiq 2F-85' : '仮想'} ·{' '}
            {((sim?.gripper.position ?? (sim?.gripper.closed ? 1 : 0)) * 100).toFixed(0)}% 閉
          </b>
        </div>
      </Section>

      <Section title="カメラ">
        <div className="row wrap">
          {(sim?.cameras ?? (['operator', 'front', 'top', 'wrist'] as CameraName[])).map((c) => (
            <button key={c} className={sim?.camera === c ? 'active' : ''} onClick={() => void app.setCamera(c)}>
              {CAMERA_LABELS[c] ?? c}
            </button>
          ))}
        </div>
        <div className="row wrap">
          <span className="muted small">VR 内の表示</span>
          <button className={sim?.view !== 'immersive' ? 'active' : ''} onClick={() => void app.setView('screen')}>
            モニター
          </button>
          <button className={sim?.view === 'immersive' ? 'active' : ''} onClick={() => void app.setView('immersive')}>
            没入（シーンに入る）
          </button>
          <button disabled={sim?.view !== 'immersive'} onClick={() => void app.recenter()}>
            正面リセット
          </button>
        </div>
        <p className="muted small">
          {sim?.view === 'immersive'
            ? '没入表示: 自分が Isaac Sim のシーンの中に入り、頭の動きに視界が追従します（VR 中のみ）。左スティックで移動、右スティック左右で 30° 回転。操作方向は見たままに一致し、アバターの表示や手で物を掴むのもこの表示です。'
            : 'モニター表示: 空間に浮かぶ仮想モニターで、選んだカメラの映像を見ます。'}
        </p>
        <Toggle label="鏡を表示" checked={sim?.mirror ?? true} onChange={(v) => void app.setMirror(v)} />
        {sim?.avatar && (
          <>
            <Toggle
              label={`アバターを表示（${sim.avatar.source}）`}
              checked={sim.avatar.visible}
              onChange={(v) => void app.setAvatar({ visible: v })}
            />
            <Toggle
              label="録画中はアバターを隠す（データセットに写さない）"
              checked={sim.avatar.hideWhileRecording}
              onChange={(v) => void app.setAvatar({ hideWhileRecording: v })}
            />
            {sim.avatar.bodies > 0 && (
              <Toggle
                label="アバターの当たり判定（手で物を押す・ロボットが止まる）"
                checked={sim.avatar.collision !== 'off'}
                onChange={(v) => void app.setAvatar({ collision: v ? 'on' : 'off' })}
              />
            )}
            <p className="muted small">アバターは没入表示のときだけ表示されます。</p>
            {!sim.avatar.ok && <p className="error small">アバターの姿勢計算でエラーが発生したため停止しました（サーバのログを確認してください）。</p>}
          </>
        )}
        <ResolutionControls />
        <Toggle
          label={`ステレオ配信（立体視, IPD ${((sim?.ipd ?? 0.064) * 1000).toFixed(1)} mm）`}
          checked={sim?.stereo ?? false}
          onChange={(v) => void app.setStereo(v)}
        />
      </Section>

      <Section title="テレオペ設定">
        <div className="row wrap">
          <span className="muted small">操作する手</span>
          {(['right', 'left'] as const).map((h) => (
            <button key={h} className={sim?.settings.hand === h ? 'active' : ''} onClick={() => void app.setTeleop({ hand: h })}>
              {h === 'right' ? '右' : '左'}
            </button>
          ))}
        </div>
        <div className="row wrap">
          <span className="muted small">回転の追従</span>
          {(
            [
              ['yaw', '水平のみ'],
              ['full', '全軸'],
              ['off', 'なし'],
            ] as const
          ).map(([m, label]) => (
            <button
              key={m}
              className={(sim?.settings.rotationMode ?? 'yaw') === m ? 'active' : ''}
              onClick={() => void app.setTeleop({ rotationMode: m })}
            >
              {label}
            </button>
          ))}
        </div>
        <label className="range">
          <span>移動倍率 x{(sim?.settings.positionScale ?? 1).toFixed(2)}</span>
          <input
            type="range"
            min={0.1}
            max={3}
            step={0.05}
            value={sim?.settings.positionScale ?? 1}
            onChange={(e) => void app.setTeleop({ positionScale: parseFloat(e.target.value) })}
          />
        </label>
        <div className="row wrap">
          <span className="muted small">操作基準（モニター表示時）</span>
          {(
            [
              ['camera', '表示カメラ'],
              ['operator', 'オペレータ視点'],
              ['robot', 'ロボット'],
            ] as const
          ).map(([m, label]) => (
            <button
              key={m}
              className={(sim?.settings.mapping ?? 'camera') === m ? 'active' : ''}
              onClick={() => void app.setTeleop({ mapping: m })}
            >
              {label}
            </button>
          ))}
        </div>
        <p className="muted small">
          表示カメラ: 見ている映像の向きに合わせる（視点切替で変わる） / オペレータ視点: 常にオペレータカメラ基準 /
          ロボット: 手を前に出すとロボットの正面方向へ。没入表示では常に見たまま。
        </p>
        <label className="num">
          <span>没入表示での手</span>
          <select
            value={sim?.settings.handMode ?? 'robot'}
            onChange={(e) => void app.setTeleop({ handMode: e.target.value as HandMode })}
          >
            <option value="robot">操作する手でロボット、反対の手で物を掴む</option>
            <option value="grab">両手で物を掴む（ロボット操作は停止）</option>
          </select>
        </label>
        <p className="muted small">没入表示ではグリップボタンで手元の物体を直接掴めます（離すと手の速さで投げられます）。</p>
        <Toggle
          label="把持アシスト（挟んだ部品を確実に保持）"
          checked={sim?.settings.graspAssist ?? true}
          onChange={(graspAssist) => void app.setTeleop({ graspAssist })}
        />
        <label className="range">
          <span>手ぶれ補正 {((sim?.settings.smoothing ?? 0.5) * 100).toFixed(0)}%</span>
          <input
            type="range"
            min={0}
            max={0.95}
            step={0.05}
            value={sim?.settings.smoothing ?? 0.5}
            onChange={(e) => void app.setTeleop({ smoothing: parseFloat(e.target.value) })}
          />
        </label>
        <div className="row wrap">
          <span className="muted small">VR メニュー</span>
          {(
            [
              ['hand', '手元'],
              ['fixed', '左下に固定'],
              ['hidden', '非表示'],
            ] as const
          ).map(([m, label]) => (
            <button key={m} className={vrMenu === m ? 'active' : ''} onClick={() => app.setVrMenu(m)}>
              {label}
            </button>
          ))}
        </div>
        <details className="help">
          <summary>VR コントローラ操作</summary>
          <ul>
            <li>操作する手: グリップ長押し = クラッチ(押している間だけアームが追従) / トリガー = グリッパ</li>
            <li>操作する手: A/X = カメラ切替, B/Y = ランダム化, スティック押込 = ホーム</li>
            <li>反対の手: X/A 長押し = リセット, Y/B = メニュー位置切替（手元 / 左下 / 非表示）, スティック上下 = モニター距離（没入表示では歩行）</li>
            <li>メニューは反対の手のコントローラの上に表示されます。操作する手のレイで押してください。</li>
            <li>反対の手: スティック押込 = 録画開始 / 成功で保存</li>
          </ul>
        </details>
      </Section>

      <Section
        title="ドメインランダマイゼーション"
        right={
          <button className="primary" onClick={() => void app.randomize()}>
            有効な項目を実行
          </button>
        }
      >
        {!draft ? (
          <p className="muted">読み込み中…</p>
        ) : (
          <>
            <div className="row wrap">
              {(Object.keys(DR_LABELS) as DrTarget[]).map((t) => (
                <button key={t} onClick={() => void app.randomize([t])}>
                  {DR_LABELS[t]}のみ
                </button>
              ))}
            </div>
            <div className="dr-grid">
              <fieldset>
                <legend>
                  <Toggle label={DR_LABELS.objects} checked={draft.objects.enabled} onChange={(enabled) => upd('objects', { enabled })} />
                </legend>
                <Toggle label="ランダム回転(yaw)" checked={draft.objects.randomYaw} onChange={(randomYaw) => upd('objects', { randomYaw })} />
                <Num label="最小間隔[m]" step={0.01} value={draft.objects.minSpacing} onChange={(minSpacing) => upd('objects', { minSpacing })} />
              </fieldset>
              <fieldset>
                <legend>
                  <Toggle label={DR_LABELS.lighting} checked={draft.lighting.enabled} onChange={(enabled) => upd('lighting', { enabled })} />
                </legend>
                <Num label="強度倍率 min" value={draft.lighting.intensityScaleMin} onChange={(v) => upd('lighting', { intensityScaleMin: v })} />
                <Num label="強度倍率 max" value={draft.lighting.intensityScaleMax} onChange={(v) => upd('lighting', { intensityScaleMax: v })} />
                <Num label="色温度 min[K]" step={100} value={draft.lighting.colorTempMin} onChange={(v) => upd('lighting', { colorTempMin: v })} />
                <Num label="色温度 max[K]" step={100} value={draft.lighting.colorTempMax} onChange={(v) => upd('lighting', { colorTempMax: v })} />
                <Toggle label="光源の向き" checked={draft.lighting.randomDirection} onChange={(v) => upd('lighting', { randomDirection: v })} />
              </fieldset>
              <fieldset>
                <legend>
                  <Toggle label={DR_LABELS.materials} checked={draft.materials.enabled} onChange={(enabled) => upd('materials', { enabled })} />
                </legend>
                <Toggle label="テーブル/台座" checked={draft.materials.table} onChange={(table) => upd('materials', { table })} />
                <Toggle label="床" checked={draft.materials.floor} onChange={(floor) => upd('materials', { floor })} />
              </fieldset>
              <fieldset>
                <legend>
                  <Toggle label={DR_LABELS.camera} checked={draft.camera.enabled} onChange={(enabled) => upd('camera', { enabled })} />
                </legend>
                <Num label="位置[m]" step={0.01} value={draft.camera.positionJitter} onChange={(v) => upd('camera', { positionJitter: v })} />
                <Num label="回転[deg]" step={0.5} value={draft.camera.rotationJitterDeg} onChange={(v) => upd('camera', { rotationJitterDeg: v })} />
              </fieldset>
              <fieldset>
                <legend>
                  <Toggle label={DR_LABELS.mirror} checked={draft.mirror.enabled} onChange={(enabled) => upd('mirror', { enabled })} />
                </legend>
                <Num label="表示確率" step={0.1} min={0} max={1} value={draft.mirror.probability} onChange={(v) => upd('mirror', { probability: v })} />
              </fieldset>
              <fieldset>
                <legend>
                  <Toggle label={DR_LABELS.physics} checked={draft.physics.enabled} onChange={(enabled) => upd('physics', { enabled })} />
                </legend>
                <Num label="質量倍率 min" value={draft.physics.massScaleMin} onChange={(v) => upd('physics', { massScaleMin: v })} />
                <Num label="質量倍率 max" value={draft.physics.massScaleMax} onChange={(v) => upd('physics', { massScaleMax: v })} />
                <Num label="摩擦 min" value={draft.physics.frictionMin} onChange={(v) => upd('physics', { frictionMin: v })} />
                <Num label="摩擦 max" value={draft.physics.frictionMax} onChange={(v) => upd('physics', { frictionMax: v })} />
              </fieldset>
              <fieldset>
                <legend>実行タイミング</legend>
                <Toggle label="リセット時に実行" checked={draft.randomizeOnReset} onChange={(v) => upd('randomizeOnReset', v)} />
                <Num label="自動実行間隔[s] (0=無効)" step={1} min={0} value={draft.autoInterval} onChange={(v) => upd('autoInterval', v)} />
                <label className="num">
                  <span>シード (空=ランダム)</span>
                  <input
                    type="number"
                    value={draft.seed ?? ''}
                    onChange={(e) => upd('seed', e.target.value === '' ? null : parseInt(e.target.value, 10))}
                  />
                </label>
              </fieldset>
            </div>
            <div className="row">
              <span className="spacer" />
              <button onClick={() => setDraft(dr)}>元に戻す</button>
              <button className="primary" onClick={() => draft && void app.saveDr(draft)}>
                設定を保存
              </button>
            </div>
          </>
        )}
      </Section>

      <PerfSection />
    </div>
  )
}

const PERF_ROWS: [string, string][] = [
  ['loop.kit', 'Kit 更新（物理 + 全カメラの描画）'],
  ['loop.pre', '前処理（IK・入力・アバター）'],
  ['sim.avatar', '　うちアバター'],
  ['loop.post', '後処理（取得・録画・状態）'],
  ['grab.read', '画像取得（GPU→CPU）'],
  ['grab.concat', '左右の連結'],
  ['pose.to_publish', '頭の姿勢反映 → 配信'],
  ['track.queue', '配信待ち'],
  ['track.convert', 'フレーム変換'],
  ['enc.encode', 'エンコード'],
]

const fmt = (v: number | string | null | undefined, unit = '') =>
  v == null ? '-' : typeof v === 'number' ? `${v.toFixed(1)}${unit}` : `${v}${unit}`

/** Where the video time goes: server sections (avg / p95 ms) and this browser's receive side. */
function PerfSection() {
  const perf = useApp((s) => s.sim?.perf)
  const cp = useApp((s) => s.clientPerf)
  const fps = useApp((s) => s.sim?.fps)
  if (!perf) return null
  const r = perf.rates
  const kbps = r['enc.bytes'] != null ? (r['enc.bytes'] * 8) / 1000 : null
  return (
    <Section title="映像の計測">
      <table className="perf-table">
        <tbody>
          <tr>
            <th>シミュレーション</th>
            <td colSpan={2}>{fmt(fps, ' fps')}</td>
          </tr>
          <tr>
            <th>配信 / エンコード</th>
            <td colSpan={2}>
              {fmt(r['frames.published'], ' fps')} / {fmt(r['enc.frames'], ' fps')}（{perf.encoder || '-'}、
              {fmt(kbps != null ? kbps / 1000 : null, ' Mbps')}）
            </td>
          </tr>
          <tr className="muted small">
            <th>サーバ区間</th>
            <td>平均 ms</td>
            <td>p95 ms</td>
          </tr>
          {PERF_ROWS.filter(([k]) => perf.sections[k]).map(([k, label]) => (
            <tr key={k}>
              <th>{label}</th>
              <td>{perf.sections[k].avg.toFixed(1)}</td>
              <td>{perf.sections[k].p95.toFixed(1)}</td>
            </tr>
          ))}
          <tr className="muted small">
            <th>このブラウザ</th>
            <td colSpan={2}>{cp?.decoder ?? ''} {cp?.size ?? ''}</td>
          </tr>
          <tr>
            <th>表示 fps（VR 描画 fps）</th>
            <td colSpan={2}>
              {fmt(cp?.fps)}（{fmt(cp?.xrFps)}）
            </td>
          </tr>
          <tr>
            <th>デコード</th>
            <td colSpan={2}>{fmt(cp?.decodeMs, ' ms')}</td>
          </tr>
          <tr>
            <th>受信バッファ待ち</th>
            <td colSpan={2}>{fmt(cp?.jitterMs, ' ms')}</td>
          </tr>
          <tr>
            <th>受信 → 表示</th>
            <td colSpan={2}>{fmt(cp?.recvToShowMs, ' ms')}</td>
          </tr>
          <tr>
            <th>サーバ配信 → 表示</th>
            <td colSpan={2}>{fmt(cp?.pubToShowMs, ' ms')}</td>
          </tr>
          <tr>
            <th>頭の動き → 表示（没入）</th>
            <td colSpan={2}>
              {fmt(cp?.poseToShowMs, ' ms')}（p95 {fmt(cp?.poseToShowP95, ' ms')}）
            </td>
          </tr>
          <tr>
            <th>往復（ICE RTT）</th>
            <td colSpan={2}>{fmt(cp?.rttMs, ' ms')}</td>
          </tr>
          <tr>
            <th>受信ビットレート / 欠落</th>
            <td colSpan={2}>
              {fmt(cp?.kbps != null ? cp.kbps / 1000 : null, ' Mbps')} / {fmt(cp?.droppedPerSec, ' 枚/s')}
            </td>
          </tr>
        </tbody>
      </table>
      <p className="muted small">
        Quest で VR 中の値はサーバのログ（teleop.perf）にも 5 秒ごとに記録されます。
      </p>
    </Section>
  )
}

export const SCREEN_PRESETS: [number, number][] = [[960, 540], [1280, 720], [1600, 900], [1920, 1080]]
export const STEREO_PRESETS: [number, number][] = [[640, 360], [960, 540], [1280, 720], [1600, 900]]
export const IMMERSIVE_PRESETS = [1024, 1280, 1536, 1792, 2048]

/** Stream resolution per view (the render products are recreated on the server). */
function ResolutionControls() {
  const res = useApp((s) => s.sim?.resolution)
  const app = useApp.getState()
  if (!res) return null
  const opt = (v: [number, number]) => `${v[0]}x${v[1]}`
  const parse = (v: string) => v.split('x').map(Number) as [number, number]
  const withCurrent = <T,>(list: T[], cur: T, eq: (a: T, b: T) => boolean) => (list.some((x) => eq(x, cur)) ? list : [...list, cur])
  const same = (a: [number, number], b: [number, number]) => a[0] === b[0] && a[1] === b[1]
  return (
    <div className="res-grid">
      <label className="num">
        <span>解像度: モニター</span>
        <select value={opt(res.screen)} onChange={(e) => void app.setResolution({ screen: parse(e.target.value) })}>
          {withCurrent(SCREEN_PRESETS, res.screen, same).map((v) => (
            <option key={opt(v)} value={opt(v)}>
              {opt(v)}
            </option>
          ))}
        </select>
      </label>
      <label className="num">
        <span>モニター立体視（片目）</span>
        <select value={opt(res.stereo)} onChange={(e) => void app.setResolution({ stereo: parse(e.target.value) })}>
          {withCurrent(STEREO_PRESETS, res.stereo, same).map((v) => (
            <option key={opt(v)} value={opt(v)}>
              {opt(v)}
            </option>
          ))}
        </select>
      </label>
      <label className="num">
        <span>没入（片目）</span>
        <select value={res.immersive} onChange={(e) => void app.setResolution({ immersive: Number(e.target.value) })}>
          {withCurrent(IMMERSIVE_PRESETS, res.immersive, (a, b) => a === b).map((v) => (
            <option key={v} value={v}>
              {v}x{v}
            </option>
          ))}
        </select>
      </label>
      <label className="num">
        <span>描画（DLSS）</span>
        <select value={res.dlss ?? 'performance'} onChange={(e) => void app.setResolution({ dlss: e.target.value })}>
          {DLSS_OPTIONS.map(([v, label]) => (
            <option key={v} value={v}>
              {label}
            </option>
          ))}
        </select>
      </label>
      <BitrateControl bitrate={res.bitrate} auto={res.bitrateAuto ?? true} />
    </div>
  )
}

const DLSS_OPTIONS: [string, string][] = [
  ['performance', '性能（軽い）'],
  ['balanced', 'バランス'],
  ['quality', '画質'],
  ['native', 'ネイティブ（DLSS なし・重い）'],
]

/** Upper limit of the video bitrate (the encoder aims at a constant quality and stays below it). */
function BitrateControl({ bitrate, auto }: { bitrate: number; auto: boolean }) {
  const app = useApp.getState()
  const mbps = Math.round(bitrate / 1e6)
  const [draft, setDraft] = useState<number | null>(null)
  // send once the slider rests (dragging / arrow keys would otherwise send a command per step)
  useEffect(() => {
    if (draft == null) return
    const id = window.setTimeout(() => {
      void useApp
        .getState()
        .setResolution({ bitrate: draft * 1e6 })
        .finally(() => setDraft((d) => (d === draft ? null : d)))
    }, 500)
    return () => window.clearTimeout(id)
  }, [draft])
  const shown = draft ?? mbps
  return (
    <div className="num bitrate">
      <span>
        ビットレート上限 <b>{shown} Mbps</b>
        {auto && draft == null ? '（解像度に合わせて自動）' : ''}
      </span>
      <input
        type="range"
        min={2}
        max={100}
        step={1}
        value={shown}
        onChange={(e) => setDraft(Number(e.target.value))}
      />
      <label className="inline">
        <input type="checkbox" checked={auto} onChange={(e) => void app.setResolution({ bitrate: e.target.checked ? 'auto' : mbps * 1e6 })} />
        自動
      </label>
    </div>
  )
}
