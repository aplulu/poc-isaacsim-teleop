import { useApp } from '../store'
import type { RecSettings } from '../types'

const STATE_LABEL: Record<string, string> = {
  idle: '待機',
  preparing: '準備中',
  armed: '操作待ち',
  recording: '録画中',
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="toggle">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span>{label}</span>
    </label>
  )
}

/** Episode recording for VLA fine-tuning (LeRobot dataset). */
export function RecordingPanel() {
  const rec = useApp((s) => s.sim?.recording)
  const templates = useApp((s) => s.templates)
  const spec = useApp((s) => s.taskSpec)
  const app = useApp.getState()

  if (rec === undefined || rec === null) {
    return (
      <section className="section">
        <header>
          <h2>データ収集 (LeRobot)</h2>
        </header>
        <p className="muted small">サーバで録画が無効になっています（--no-record）。</p>
      </section>
    )
  }

  const tmpl = templates.find((t) => t.id === spec.id) ?? templates[0]
  const st = rec.state
  const busy = st !== 'idle'
  const set = (s: Partial<RecSettings>) => void app.recSettings(s)
  const ds = rec.dataset

  return (
    <section className="section rec">
      <header>
        <h2>データ収集 (LeRobot)</h2>
        <span className={`badge ${st === 'recording' ? 'bad' : st === 'idle' ? '' : 'warn'}`}>
          {st === 'recording' ? '● ' : ''}
          {STATE_LABEL[st] ?? st}
        </span>
      </header>

      <div className="task-row">
        <select
          value={useApp.getState().sim?.objectSet ?? ''}
          disabled={busy}
          onChange={(e) => void app.setObjectSet(e.target.value)}
        >
          {(useApp.getState().sim?.objectSets ?? []).map((o) => (
            <option key={o.id} value={o.id} disabled={o.count === 0}>
              {o.label}
              {o.count === 0 ? '（未登録）' : ''}
            </option>
          ))}
        </select>
        <select
          value={spec.id}
          disabled={busy}
          onChange={(e) => app.setTaskSpec({ id: e.target.value, object: 'random', text: spec.text })}
        >
          {templates.map((t) => (
            <option key={t.id} value={t.id}>
              {t.label}
            </option>
          ))}
        </select>
        {tmpl && tmpl.id !== 'free' && (
          <select
            value={spec.object ?? 'random'}
            disabled={busy}
            onChange={(e) => app.setTaskSpec({ ...spec, object: e.target.value })}
          >
            <option value="random">ランダム</option>
            {tmpl.objects.map((o) => (
              <option key={o.id} value={o.id}>
                {o.name}
              </option>
            ))}
          </select>
        )}
      </div>
      {tmpl?.id === 'free' && (
        <input
          className="wide"
          placeholder="英語の指示文 (例: stack the cube on the can)"
          value={spec.text ?? ''}
          disabled={busy}
          onChange={(e) => app.setTaskSpec({ ...spec, text: e.target.value })}
        />
      )}

      <div className="rec-status">
        <b>{rec.task?.text ?? (tmpl ? tmpl.template.replace('{obj}', '…') : '')}</b>
        <span className="muted small">
          {rec.episode != null && busy ? `ep ${rec.episode} · ` : ''}
          {rec.frames} frames · {rec.duration.toFixed(1)} s
        </span>
        {rec.message && <span className="small">{rec.message}</span>}
      </div>

      <div className="row wrap">
        {!busy ? (
          <button className="primary" onClick={() => void app.recStart()}>
            ● 録画開始
          </button>
        ) : (
          <>
            <button className="primary" disabled={st !== 'recording'} onClick={() => void app.recStop(true)}>
              成功で保存
            </button>
            <button disabled={st !== 'recording'} onClick={() => void app.recStop(false)}>
              失敗で終了
            </button>
            <button className="danger" onClick={() => void app.recDiscard()}>
              破棄
            </button>
          </>
        )}
      </div>

      <div className="rec-settings">
        <Toggle label="操作開始で録画" checked={rec.settings.waitForMotion} onChange={(v) => set({ waitForMotion: v })} />
        <Toggle label="成功を自動判定" checked={rec.settings.autoSuccess} onChange={(v) => set({ autoSuccess: v })} />
        <Toggle label="開始時にDR" checked={rec.settings.randomize} onChange={(v) => set({ randomize: v })} />
        <Toggle label="保存後に次を自動開始" checked={rec.settings.autoNext} onChange={(v) => set({ autoNext: v })} />
        <Toggle label="失敗も保存" checked={rec.settings.saveFailed} onChange={(v) => set({ saveFailed: v })} />
        <label className="num">
          <span>最大長[s]</span>
          <input
            type="number"
            min={5}
            step={5}
            value={rec.settings.maxDuration}
            onChange={(e) => set({ maxDuration: parseFloat(e.target.value) })}
          />
        </label>
      </div>

      <div className="kv">
        <span>データセット</span>
        <b className="mono-path" title={ds.root}>
          {ds.root.split(/[\\/]/).slice(-1)[0]}
        </b>
        <span>エピソード</span>
        <b>
          {ds.episodes} 本 / {ds.frames} frames ({ds.fps} fps, {ds.codec})
          {ds.pending ? ` · 保存中 ${ds.pending}` : ''}
        </b>
        <span>タスク数</span>
        <b>{ds.tasks}</b>
      </div>
      {ds.lastError && <p className="error small">書き込みエラー: {ds.lastError}</p>}
      {ds.recent.length > 0 && (
        <table className="recent">
          <tbody>
            {ds.recent.map((r) => (
              <tr key={r.episode}>
                <td>#{r.episode}</td>
                <td className={r.success ? 'ok' : 'bad'}>{r.success ? '成功' : '失敗'}</td>
                <td className="task">{r.task}</td>
                <td>{r.duration?.toFixed(1)}s</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <div className="row">
        <span className="spacer" />
        <button
          className="danger"
          disabled={busy || ds.episodes === 0 || ds.pending > 0}
          onClick={() => {
            if (window.confirm('最後に保存したエピソードを削除しますか？（元に戻せません）')) void app.recDeleteLast()
          }}
        >
          最後のエピソードを削除
        </button>
      </div>
    </section>
  )
}
