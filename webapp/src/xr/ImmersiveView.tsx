import { useFrame } from '@react-three/fiber'
import { useEffect, useMemo, useRef } from 'react'
import { ClampToEdgeWrapping, LinearFilter, Mesh, Quaternion, SRGBColorSpace, Vector3, VideoTexture } from 'three'
import { useApp } from '../store'
import { type ImmersiveMeta, frameMetaForRtp, latestFrameMeta } from './frameMeta'

/** Distance of the per-eye image quads. Far away -> head translation since rendering barely matters. */
const QUAD_DISTANCE = 4.0

type VideoWithRvfc = HTMLVideoElement & {
  requestVideoFrameCallback?: (cb: (now: number, md: { rtpTimestamp?: number }) => void) => number
  cancelVideoFrameCallback?: (id: number) => void
}

/** What the free-viewpoint view is currently showing (for the status bar). */
export const immersiveStatus = { synced: false, latencyMs: 0 }
if (new URLSearchParams(window.location.search).has('xremulate'))
  (window as unknown as { __immersiveStatus?: unknown }).__immersiveStatus = immersiveStatus

/**
 * Free-viewpoint view. The Isaac Sim stereo camera follows the head; every received frame is drawn at the head
 * pose it was rendered with (world-locked), one quad per eye covering that camera's FOV. Rendering latency
 * then never makes the image swim when the head turns - newly visible areas just appear a moment later.
 */
export function ImmersiveView({ video }: { video: HTMLVideoElement }) {
  const swapEyes = useApp((s) => s.swapEyes)
  const shown = useRef<ImmersiveMeta | undefined>(undefined)
  const left = useRef<Mesh>(null)
  const right = useRef<Mesh>(null)

  const [texL, texR] = useMemo(() => {
    const make = (offset: number) => {
      const t = new VideoTexture(video)
      t.colorSpace = SRGBColorSpace
      t.minFilter = LinearFilter
      t.magFilter = LinearFilter
      t.wrapS = ClampToEdgeWrapping
      t.generateMipmaps = false
      t.repeat.set(0.5, 1)
      t.offset.set(offset, 0)
      return t
    }
    return swapEyes ? [make(0.5), make(0)] : [make(0), make(0.5)]
  }, [video, swapEyes])
  useEffect(
    () => () => {
      texL.dispose()
      texR.dispose()
    },
    [texL, texR],
  )

  // match each presented video frame to its render pose
  useEffect(() => {
    const v = video as VideoWithRvfc
    if (!v.requestVideoFrameCallback) return
    let id = 0
    let alive = true
    const onFrame = (_now: number, md: { rtpTimestamp?: number }) => {
      if (!alive) return
      const meta = md.rtpTimestamp != null ? frameMetaForRtp(md.rtpTimestamp) : undefined
      const imm = meta?.head ? (meta as ImmersiveMeta) : undefined
      immersiveStatus.synced = !!imm
      shown.current = imm ?? latestFrameMeta()
      id = v.requestVideoFrameCallback!(onFrame)
    }
    id = v.requestVideoFrameCallback(onFrame)
    return () => {
      alive = false
      v.cancelVideoFrameCallback?.(id)
    }
  }, [video])

  const tmpQ = useMemo(() => new Quaternion(), [])
  const tmpP = useMemo(() => new Vector3(), [])
  const tmpO = useMemo(() => new Vector3(), [])

  useFrame(() => {
    // without requestVideoFrameCallback: assume the newest pose belongs to the newest frame
    const m = shown.current ?? latestFrameMeta()
    if (!m || !left.current || !right.current) {
      if (left.current) left.current.visible = false
      if (right.current) right.current.visible = false
      return
    }
    const size = 2 * QUAD_DISTANCE * Math.tan(((m.fov * Math.PI) / 180) / 2)
    tmpQ.set(m.head.q[0], m.head.q[1], m.head.q[2], m.head.q[3])
    for (const [mesh, sign] of [
      [left.current, -1],
      [right.current, 1],
    ] as const) {
      // eye position at render time, then straight ahead along that eye camera's view direction
      tmpO.set((sign * m.ipd) / 2, 0, -QUAD_DISTANCE).applyQuaternion(tmpQ)
      tmpP.set(m.head.p[0], m.head.p[1], m.head.p[2]).add(tmpO)
      mesh.position.copy(tmpP)
      mesh.quaternion.copy(tmpQ)
      mesh.scale.set(size, size, 1)
      mesh.visible = true
    }
    // XR eye cameras: left sees layers {0,1}, right sees {0,2}
    left.current.layers.set(1)
    right.current.layers.set(2)
  })

  return (
    <>
      <mesh ref={left} renderOrder={-10}>
        <planeGeometry args={[1, 1]} />
        <meshBasicMaterial map={texL} toneMapped={false} depthWrite={false} />
      </mesh>
      <mesh ref={right} renderOrder={-10}>
        <planeGeometry args={[1, 1]} />
        <meshBasicMaterial map={texR} toneMapped={false} depthWrite={false} />
      </mesh>
    </>
  )
}
