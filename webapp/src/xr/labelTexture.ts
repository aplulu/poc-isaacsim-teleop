import { CanvasTexture, SRGBColorSpace } from 'three'

/**
 * Text rendered into a CanvasTexture. Uses the system font, so it works offline (no font download)
 * and supports Japanese text inside VR.
 */
export class LabelTexture {
  readonly canvas: HTMLCanvasElement
  readonly texture: CanvasTexture
  private ctx: CanvasRenderingContext2D
  private last = ''

  constructor(width = 512, height = 128) {
    this.canvas = document.createElement('canvas')
    this.canvas.width = width
    this.canvas.height = height
    this.ctx = this.canvas.getContext('2d')!
    this.texture = new CanvasTexture(this.canvas)
    this.texture.colorSpace = SRGBColorSpace
    this.texture.anisotropy = 4
  }

  draw(
    lines: string[],
    opts: { bg?: string; fg?: string; size?: number; align?: CanvasTextAlign; border?: string } = {},
  ) {
    const key = JSON.stringify([lines, opts])
    if (key === this.last) return
    this.last = key
    const { width, height } = this.canvas
    const c = this.ctx
    const size = opts.size ?? Math.floor(height / Math.max(lines.length, 1) / 1.6)
    c.clearRect(0, 0, width, height)
    c.fillStyle = opts.bg ?? 'rgba(13,17,23,0.85)'
    roundRect(c, 2, 2, width - 4, height - 4, Math.min(24, height / 4))
    c.fill()
    if (opts.border) {
      c.strokeStyle = opts.border
      c.lineWidth = 4
      c.stroke()
    }
    c.fillStyle = opts.fg ?? '#e6edf3'
    c.font = `600 ${size}px system-ui, -apple-system, "Segoe UI", "Noto Sans JP", sans-serif`
    c.textAlign = opts.align ?? 'center'
    c.textBaseline = 'middle'
    const x = c.textAlign === 'left' ? 24 : width / 2
    const lh = height / lines.length
    lines.forEach((l, i) => c.fillText(l, x, lh * (i + 0.5), width - 32))
    this.texture.needsUpdate = true
  }

  dispose() {
    this.texture.dispose()
  }
}

function roundRect(c: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  c.beginPath()
  c.moveTo(x + r, y)
  c.arcTo(x + w, y, x + w, y + h, r)
  c.arcTo(x + w, y + h, x, y + h, r)
  c.arcTo(x, y + h, x, y, r)
  c.arcTo(x, y, x + w, y, r)
  c.closePath()
}
