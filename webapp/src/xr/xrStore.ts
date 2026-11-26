import { createXRStore } from '@react-three/xr'
import { useApp } from '../store'

// Debug without a headset: open the app with ?xremulate to emulate a Meta Quest 3 (IWER, stereo, 63 mm IPD)
// in a desktop browser. The left/right halves of the canvas then show what each eye sees.
if (new URLSearchParams(window.location.search).has('xremulate')) {
  void import('iwer').then(({ XRDevice, metaQuest3 }) => {
    const device = new XRDevice(metaQuest3)
    device.ipd = 0.063
    device.stereoEnabled = true
    device.installRuntime({ forceInstall: true })
    ;(window as unknown as { __xrDevice?: unknown }).__xrDevice = device // for scripted tests
  })
}

export const xrStore = createXRStore({
  // Teleoperation needs the physical controllers (grip/trigger), so hand tracking is disabled.
  hand: false,
  emulate: false,
  frameRate: 'high',
  foveation: 0,
})

xrStore.subscribe((s, prev) => {
  if (!!s.session !== !!prev.session) useApp.getState().setInXR(!!s.session)
})
