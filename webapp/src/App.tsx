import { useEffect, useState } from 'react'
import { ControlPanel } from './components/ControlPanel'
import { PairingScreen } from './components/PairingScreen'
import { useApp } from './store'
import { XRScene } from './xr/XRScene'
import { watchVideo } from './perfStats'

/** Hidden <video> element fed by the WebRTC stream; used as a texture source by the 3D scene. */
function useVideoElement(stream: MediaStream | null) {
  const [video] = useState(() => {
    const v = document.createElement('video')
    v.playsInline = true
    v.muted = true
    v.autoplay = true
    return v
  })
  useEffect(() => {
    video.srcObject = stream
    if (stream) video.play().catch(() => undefined)
  }, [video, stream])
  useEffect(() => watchVideo(video), [video])
  return stream ? video : null
}

function Toasts() {
  const toasts = useApp((s) => s.toasts)
  return (
    <div className="toasts">
      {toasts.map((t) => (
        <div key={t.id} className={`toast ${t.kind}`}>
          {t.text}
        </div>
      ))}
    </div>
  )
}

export default function App() {
  const phase = useApp((s) => s.phase)
  const conn = useApp((s) => s.conn)
  const stream = useApp((s) => s.stream)
  const video = useVideoElement(stream)

  useEffect(() => {
    const s = useApp.getState()
    void s.loadInfo()
    // restored session (token in localStorage): reconnect straight away
    if (s.phase === 'paired' && s.conn === 'idle') void s.connect()
  }, [])

  // auto-reconnect when the peer connection drops while we are still paired
  useEffect(() => {
    if (phase !== 'paired' || (conn !== 'failed' && conn !== 'disconnected')) return
    const t = window.setTimeout(() => void useApp.getState().connect(), 2000)
    return () => window.clearTimeout(t)
  }, [phase, conn])

  if (phase !== 'paired') {
    return (
      <>
        <PairingScreen />
        <Toasts />
      </>
    )
  }

  return (
    <div className="layout">
      <div className="stage">
        <XRScene video={video} />
        {!stream && <div className="overlay">{conn === 'connecting' ? '映像を待っています…' : `接続状態: ${conn}`}</div>}
      </div>
      <ControlPanel />
      <Toasts />
    </div>
  )
}
