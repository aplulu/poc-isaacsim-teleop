import { OrbitControls } from '@react-three/drei'
import { Canvas, useFrame } from '@react-three/fiber'
import { XR } from '@react-three/xr'
import { useEffect } from 'react'
import { useApp } from '../store'
import { xrStore } from './xrStore'
import { TeleopSender } from './TeleopSender'
import { SCREEN_HEIGHT, VideoScreen } from './VideoScreen'
import { ImmersiveView } from './ImmersiveView'
import { ScreenStatus, VRPanel } from './VRPanel'
import { markXRFrame } from '../perfStats'

function FrameCounter() {
  useFrame(() => markXRFrame())
  return null
}

export function XRScene({ video }: { video: HTMLVideoElement | null }) {
  const inXR = useApp((s) => s.inXR)
  const distance = useApp((s) => s.screenDistance)
  // free viewpoint needs head tracking -> only inside an XR session (desktop keeps the flat preview)
  const immersive = useApp((s) => s.sim?.view === 'immersive') && inXR

  useEffect(() => () => void xrStore.getState().session?.end(), [])

  return (
    <Canvas
      camera={{ position: [0, SCREEN_HEIGHT, 0.8], fov: 60, near: 0.01, far: 100 }}
      gl={{ antialias: true, alpha: false }}
      style={{ background: '#0d1117' }}
    >
      <XR store={xrStore}>
        <color attach="background" args={['#0d1117']} />
        {!immersive && <gridHelper args={[10, 20, '#30363d', '#1c2128']} />}
        {video && (immersive ? <ImmersiveView video={video} /> : <VideoScreen video={video} />)}
        {inXR && <VRPanel />}
        {inXR && !immersive && <ScreenStatus />}
        <TeleopSender />
        <FrameCounter />
        {!inXR && <OrbitControls target={[0, SCREEN_HEIGHT, -distance]} enablePan={false} makeDefault />}
      </XR>
    </Canvas>
  )
}
