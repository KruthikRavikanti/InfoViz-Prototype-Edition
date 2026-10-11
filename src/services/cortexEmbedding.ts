export type CortexCompassStatus = 'loading' | 'ready' | 'error';

type CortexFrameEnvironment = {
  location: { search?: string; origin?: string };
  parent?: { postMessage?: (message: unknown, targetOrigin: string) => void };
  self?: unknown;
};

function browserEnvironment(): CortexFrameEnvironment | undefined {
  return typeof window === 'undefined' ? undefined : window;
}

/** A query parameter alone must not remove standalone navigation. */
export function isCortexEmbedded(environment = browserEnvironment()): boolean {
  if (!environment?.parent || environment.parent === environment || environment.parent === environment.self) {
    return false;
  }
  return new URLSearchParams(environment.location.search ?? '').get('embedded') === '1';
}

function postToCortex(message: unknown, environment = browserEnvironment()): boolean {
  if (!isCortexEmbedded(environment) || !environment?.parent?.postMessage) return false;
  const origin = environment.location.origin;
  // The integrated route is same-origin. Never send session data to another
  // origin, use a wildcard recipient, or send anything from an opaque frame.
  if (!origin || !/^https?:\/\//.test(origin)) return false;
  try {
    if (new URL(origin).origin !== origin) return false;
    environment.parent.postMessage(message, origin);
    return true;
  } catch {
    return false;
  }
}

export function notifyCortexStatus(status: CortexCompassStatus, environment = browserEnvironment()): boolean {
  if (status !== 'loading' && status !== 'ready' && status !== 'error') return false;
  return postToCortex({ type: 'cortex-compass:status', status }, environment);
}

/** The host returns to its existing Lab state rather than nesting Lab in the frame. */
export function requestCortexReturnToLab(environment = browserEnvironment()): boolean {
  return postToCortex({ type: 'cortex-compass:return-to-lab' }, environment);
}

export function getCortexNavigationTarget(embedded: boolean): '_top' | undefined {
  return embedded ? '_top' : undefined;
}
