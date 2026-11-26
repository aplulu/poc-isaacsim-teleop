import type { PairConfirmResponse, PairRequestResponse, ServerInfo } from './types'

export class ApiError extends Error {
  status: number
  code: string
  constructor(status: number, code: string, message: string) {
    super(message)
    this.status = status
    this.code = code
  }
}

let authToken: string | null = null
let onUnauthorized: (() => void) | null = null

export function setAuthToken(token: string | null) {
  authToken = token
}

export function setUnauthorizedHandler(fn: () => void) {
  onUnauthorized = fn
}

async function request<T>(method: string, path: string, body?: unknown, auth = true): Promise<T> {
  const headers: Record<string, string> = {}
  if (body !== undefined) headers['Content-Type'] = 'application/json'
  if (auth && authToken) headers['Authorization'] = `Bearer ${authToken}`
  const res = await fetch(path, {
    method,
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined,
    cache: 'no-store',
  })
  let data: unknown = null
  const text = await res.text()
  try {
    data = text ? JSON.parse(text) : null
  } catch {
    data = { message: text }
  }
  if (!res.ok) {
    const d = (data ?? {}) as { error?: string; message?: string }
    if (res.status === 401 && auth && onUnauthorized) onUnauthorized()
    throw new ApiError(res.status, d.error ?? 'error', d.message ?? res.statusText)
  }
  return data as T
}

export const api = {
  info: () => request<ServerInfo>('GET', '/api/info', undefined, false),
  pairRequest: (deviceName: string) =>
    request<PairRequestResponse>('POST', '/api/pair/request', { deviceName }, false),
  pairConfirm: (requestId: string, code: string) =>
    request<PairConfirmResponse>('POST', '/api/pair/confirm', { requestId, code }, false),
  session: () => request<{ sessionId: string; deviceName: string; expiresAt: number }>('GET', '/api/session'),
  logout: () => request<{ ok: boolean }>('POST', '/api/session/logout'),
  offer: (sdp: string) => request<{ sdp: string; type: RTCSdpType; peerId: string }>('POST', '/api/webrtc/offer', { sdp, type: 'offer' }),
  command: <T = unknown>(name: string, payload: object = {}) =>
    request<{ ok: boolean; result: T }>('POST', '/api/command', { name, payload }).then((r) => r.result),
}
