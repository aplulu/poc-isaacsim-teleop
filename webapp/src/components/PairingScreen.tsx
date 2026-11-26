import { useEffect, useRef, useState } from 'react'
import { useApp } from '../store'

const KEYS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'C', '0', '⌫']

export function PairingScreen() {
  const { phase, deviceName, pairError, requestExpiresAt, info } = useApp()
  const { setDeviceName, requestPairing, confirmPairing, cancelPairing } = useApp.getState()
  const [code, setCode] = useState('')
  const [now, setNow] = useState(Date.now())
  const inputRef = useRef<HTMLInputElement>(null)
  const len = info?.codeLength ?? 6

  useEffect(() => {
    const t = window.setInterval(() => setNow(Date.now()), 500)
    return () => window.clearInterval(t)
  }, [])

  useEffect(() => {
    if (phase === 'awaitingCode') {
      setCode('')
      inputRef.current?.focus()
    }
  }, [phase])

  const remaining = requestExpiresAt ? Math.max(0, Math.round((requestExpiresAt - now) / 1000)) : 0
  useEffect(() => {
    if (phase === 'awaitingCode' && requestExpiresAt && remaining === 0) cancelPairing()
  }, [phase, remaining, requestExpiresAt, cancelPairing])

  const submit = (c = code) => {
    if (c.length === len && phase === 'awaitingCode') void confirmPairing(c)
  }

  const press = (k: string) => {
    if (k === 'C') return setCode('')
    if (k === '⌫') return setCode((c) => c.slice(0, -1))
    setCode((c) => {
      const next = (c + k).slice(0, len)
      if (next.length === len) window.setTimeout(() => submit(next), 50)
      return next
    })
  }

  return (
    <div className="pairing">
      <div className="card">
        <h1>Isaac Sim VR Teleop</h1>
        <p className="muted">{info ? `${info.robot} を WebXR + WebRTC で遠隔操作` : 'サーバ情報を取得中…'}</p>

        {(phase === 'idle' || phase === 'requesting') && (
          <>
            <label className="field">
              <span>デバイス名</span>
              <input value={deviceName} maxLength={64} onChange={(e) => setDeviceName(e.target.value)} />
            </label>
            <button className="primary big" disabled={phase === 'requesting'} onClick={() => void requestPairing()}>
              {phase === 'requesting' ? '要求中…' : '接続を要求する'}
            </button>
            <p className="muted small">
              要求すると Isaac Sim の「VR Teleop - Pairing」ウィンドウに 6 桁の認証コードが表示されます。
            </p>
          </>
        )}

        {(phase === 'awaitingCode' || phase === 'confirming') && (
          <>
            <p>Isaac Sim のウィンドウに表示された {len} 桁のコードを入力してください</p>
            <input
              ref={inputRef}
              className="code-input"
              inputMode="numeric"
              autoComplete="one-time-code"
              pattern="[0-9]*"
              value={code}
              maxLength={len}
              onChange={(e) => {
                const v = e.target.value.replace(/\D/g, '').slice(0, len)
                setCode(v)
                if (v.length === len) submit(v)
              }}
              onKeyDown={(e) => e.key === 'Enter' && submit()}
            />
            <div className="code-boxes" aria-hidden>
              {Array.from({ length: len }, (_, i) => (
                <span key={i} className={i < code.length ? 'filled' : ''}>
                  {code[i] ?? ''}
                </span>
              ))}
            </div>
            <div className="keypad">
              {KEYS.map((k) => (
                <button key={k} onClick={() => press(k)} disabled={phase === 'confirming'}>
                  {k}
                </button>
              ))}
            </div>
            <div className="row">
              <span className="muted small">有効期限 残り {remaining} 秒</span>
              <span className="spacer" />
              <button onClick={cancelPairing}>キャンセル</button>
              <button className="primary" disabled={code.length !== len || phase === 'confirming'} onClick={() => submit()}>
                {phase === 'confirming' ? '確認中…' : '接続'}
              </button>
            </div>
          </>
        )}
        {pairError && <p className="error">{pairError}</p>}
      </div>
    </div>
  )
}
