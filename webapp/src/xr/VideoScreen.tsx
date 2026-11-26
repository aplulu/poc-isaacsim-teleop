import { useFrame } from '@react-three/fiber'
import { useEffect, useMemo, useRef, useState } from 'react'
import { ClampToEdgeWrapping, DoubleSide, LinearFilter, Mesh, SRGBColorSpace, VideoTexture } from 'three'
import { useApp } from '../store'

/** Height of the screen center above the floor (meters) */
export const SCREEN_HEIGHT = 1.45

/** Stereo state actually used for drawing (shown in the VR status bar). */
export const stereoRender = { active: false, frameW: 0, frameH: 0 }

/**
 * Screen showing the Isaac Sim camera stream. Its size is derived from the camera's horizontal FOV so that,
 * seen from the origin, the image covers the same visual angle as the simulated camera.
 *
 * Stereo (side-by-side frame): the left half is shown only to the left eye (three.js XR layer 1) and the right
 * half only to the right eye (layer 2). Each eye's screen is shifted sideways by half the IPD so that it sits
 * straight in front of that eye: with FOV-matched images from parallel cameras this is orthostereoscopic -
 * objects appear at their true distance from the camera regardless of how far away the screen is placed.
 */
export function VideoScreen({ video }: { video: HTMLVideoElement }) {
  const hfov = useApp((s) => s.sim?.hfov ?? 73.6)
  const distance = useApp((s) => s.screenDistance)
  const inXR = useApp((s) => s.inXR)
  const ipd = useApp((s) => s.sim?.ipd ?? 0.064)
  const serverStereo = useApp((s) => s.sim?.stereo ?? false)
  const swapEyes = useApp((s) => s.swapEyes)

  // Poll the decoded frame size every frame: 'resize' events are not reliably delivered for a detached
  // <video> element when the stream switches between mono (16:9) and side-by-side stereo (32:9).
  const [size, setSize] = useState({ w: 0, h: 0 })
  useFrame(() => {
    const w = video.videoWidth
    const h = video.videoHeight
    if (w !== size.w || h !== size.h) setSize({ w, h })
  })
  const known = size.w > 0 && size.h > 0
  // 16:9 = mono, 32:9 = stereo pair, 2:1 = free-viewpoint stereo pair (square eyes)
  const stereo = known ? size.w / size.h > 1.9 : serverStereo
  const aspect = known ? size.w / (stereo ? 2 : 1) / size.h : 16 / 9

  const [texL, texR] = useMemo(() => {
    const make = (offset: number) => {
      const t = new VideoTexture(video)
      t.colorSpace = SRGBColorSpace
      t.minFilter = LinearFilter
      t.magFilter = LinearFilter
      t.wrapS = ClampToEdgeWrapping
      t.generateMipmaps = false
      if (stereo) {
        t.repeat.set(0.5, 1)
        t.offset.set(offset, 0)
      }
      return t
    }
    return swapEyes ? [make(0.5), make(0)] : [make(0), make(0.5)]
  }, [video, stereo, swapEyes])

  useEffect(
    () => () => {
      texL.dispose()
      texR.dispose()
    },
    [texL, texR],
  )

  const width = 2 * distance * Math.tan(((hfov * Math.PI) / 180) / 2)
  const height = width / aspect
  const left = useRef<Mesh>(null)
  const right = useRef<Mesh>(null)
  const stereoXR = stereo && inXR

  useFrame(() => {
    stereoRender.active = stereoXR
    stereoRender.frameW = size.w
    stereoRender.frameH = size.h
    // XR eye cameras: left = layers {0, 1}, right = layers {0, 2}. Mono / desktop: left image on layer 0.
    if (left.current) {
      if (stereoXR) left.current.layers.set(1)
      else left.current.layers.set(0)
    }
    if (right.current) {
      right.current.layers.set(2)
      right.current.visible = stereoXR
    }
  })

  return (
    <group position={[0, SCREEN_HEIGHT, -distance]}>
      <mesh ref={left} position={[stereoXR ? -ipd / 2 : 0, 0, 0]}>
        <planeGeometry args={[width, height]} />
        <meshBasicMaterial map={texL} toneMapped={false} side={DoubleSide} />
      </mesh>
      {stereo && (
        <mesh ref={right} position={[ipd / 2, 0, 0]}>
          <planeGeometry args={[width, height]} />
          <meshBasicMaterial map={texR} toneMapped={false} side={DoubleSide} />
        </mesh>
      )}
      {/* thin frame */}
      <mesh position={[0, 0, -0.01]}>
        <planeGeometry args={[width + 0.04 + (stereoXR ? ipd : 0), height + 0.04]} />
        <meshBasicMaterial color="#20262e" />
      </mesh>
    </group>
  )
}
