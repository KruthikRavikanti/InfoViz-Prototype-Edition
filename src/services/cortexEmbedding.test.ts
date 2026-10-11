import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  getCortexNavigationTarget,
  isCortexEmbedded,
  notifyCortexStatus,
  requestCortexReturnToLab,
  type CortexCompassStatus,
} from './cortexEmbedding';

function frame(search = '?embedded=1', origin = 'https://cortex.example') {
  return { location: { search, origin }, parent: { postMessage: vi.fn() }, self: undefined as unknown };
}

afterEach(() => { vi.unstubAllGlobals(); });

describe('Cortex embedding contract', () => {
  it('requires an explicit flag and an actual parent frame', () => {
    expect(isCortexEmbedded(frame())).toBe(true);
    expect(isCortexEmbedded(frame('?embedded=1&view=clusters'))).toBe(true);
    for (const search of ['', '?embedded=0', '?embedded=true', '?session=embedded=1']) {
      expect(isCortexEmbedded(frame(search))).toBe(false);
    }
    const topLevel = frame();
    topLevel.self = topLevel.parent;
    expect(isCortexEmbedded(topLevel)).toBe(false);
    expect(isCortexEmbedded({ location: { search: '?embedded=1' } })).toBe(false);
    const standalone = { location: { search: '?embedded=1' }, parent: undefined as unknown };
    standalone.parent = standalone;
    expect(isCortexEmbedded(standalone as Parameters<typeof isCortexEmbedded>[0])).toBe(false);
  });

  it('supports the browser environment and incomplete SSR test stubs safely', () => {
    vi.stubGlobal('window', frame());
    expect(isCortexEmbedded()).toBe(true);
    vi.stubGlobal('window', { location: { hash: '' } });
    expect(isCortexEmbedded()).toBe(false);
    vi.stubGlobal('window', undefined);
    expect(isCortexEmbedded()).toBe(false);
    expect(notifyCortexStatus('ready')).toBe(false);
  });

  it.each(['loading', 'ready', 'error'] as const)('sends only the known %s status to the same origin', (status) => {
    const environment = frame();
    expect(notifyCortexStatus(status, environment)).toBe(true);
    expect(environment.parent.postMessage).toHaveBeenCalledExactlyOnceWith(
      { type: 'cortex-compass:status', status }, 'https://cortex.example',
    );
  });

  it('does not send unsupported statuses or messages outside embedded mode', () => {
    const environment = frame('');
    expect(notifyCortexStatus('ready', environment)).toBe(false);
    expect(requestCortexReturnToLab(environment)).toBe(false);
    expect(environment.parent.postMessage).not.toHaveBeenCalled();
    const embedded = frame();
    expect(notifyCortexStatus('session-secret' as CortexCompassStatus, embedded)).toBe(false);
    expect(embedded.parent.postMessage).not.toHaveBeenCalled();
  });

  it.each(['null', '', '*', 'https://cortex.example/path', 'javascript:alert(1)'])('never posts to an opaque or invalid origin: %s', (origin) => {
    const environment = frame('?embedded=1', origin);
    expect(notifyCortexStatus('ready', environment)).toBe(false);
    expect(requestCortexReturnToLab(environment)).toBe(false);
    expect(environment.parent.postMessage).not.toHaveBeenCalled();
  });

  it('requests return without including session tokens or prediction data', () => {
    const environment = frame();
    expect(requestCortexReturnToLab(environment)).toBe(true);
    expect(environment.parent.postMessage).toHaveBeenCalledExactlyOnceWith(
      { type: 'cortex-compass:return-to-lab' }, 'https://cortex.example',
    );
    expect(getCortexNavigationTarget(true)).toBe('_top');
    expect(getCortexNavigationTarget(false)).toBeUndefined();
  });

  it('keeps normal navigation available if parent messaging is unavailable', () => {
    const environment = frame();
    environment.parent.postMessage.mockImplementation(() => { throw new Error('unavailable'); });
    expect(requestCortexReturnToLab(environment)).toBe(false);
  });
});
