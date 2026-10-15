import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import { validateComparisonSession } from '../services/cortexSession';
import { rawSessionFixture } from '../test/sessionFixture';
import { SessionBanner } from './SessionBanner';
import { Children, isValidElement, type ReactElement, type ReactNode } from 'react';

function multiModelSession() {
  const session = validateComparisonSession(rawSessionFixture(), 'http://core.example:5556');
  const efficientRois = structuredClone(session.roiResults);
  efficientRois.ffa.stimuli[0].mean = 0.321;
  session.modelResults = {
    efficient_net: { roiResults: efficientRois },
    vgg16_imagenet1kv1: { roiResults: structuredClone(session.roiResults) },
  };
  return session;
}

type ButtonProps = { children?: ReactNode; 'aria-label'?: string; onClick?: () => void };
function elements(tree: ReactNode): ReactElement<ButtonProps>[] {
  return Children.toArray(tree).flatMap((child) => isValidElement<ButtonProps>(child)
    ? [child, ...elements(child.props.children)] : []);
}

describe('comparison session context', () => {
  it('keeps uploads in a compact summary with technical details collapsed', () => {
    const session = validateComparisonSession(rawSessionFixture(), 'http://core.example:5556');
    const markup = renderToStaticMarkup(
      <SessionBanner session={session} focusedStimulusId="stimulus-a" onFocusStimulus={vi.fn()} />,
    );

    expect(markup).toContain('aria-label="Cortex-Compass comparison session"');
    expect(markup).toContain('Comparison active');
    expect(markup).toContain('1 upload');
    expect(markup).toContain('uploaded.png');
    expect(markup).toContain('aria-pressed="true"');
    expect(markup).toContain('<details class="session-context-details">');
    expect(markup).toContain('Comparison details');
    expect(markup).toContain('Mean predicted response by region');
    expect(markup).toContain('Predictions only; no measured fMRI for uploads.');
    expect(markup).toContain('Your original Lab result is unchanged.');
    expect(markup).toContain('Lab result:');
    expect(markup).toContain('Upload comparison:');
    expect(markup).not.toContain('<h2>');
  });

  it('shows deduplicated prepared model buttons and the active model’s region responses', () => {
    const markup = renderToStaticMarkup(
      <SessionBanner session={multiModelSession()} selectedModel="efficient_net"
        focusedStimulusId="stimulus-a" onFocusStimulus={vi.fn()} onSelectModel={vi.fn()} />,
    );
    expect(markup).toContain('3 models');
    expect(markup).toContain('EfficientNet');
    expect(markup).toContain('CLIP RN50');
    expect(markup).toContain('VGG16 (ImageNet)');
    expect(markup).toContain('aria-label="View uploaded stimuli with efficient_net"');
    expect(markup).toContain('<strong>0.321</strong>');
    expect(markup).not.toContain('<strong>0.800</strong>');
    expect(markup).toContain('Lab result: <strong>clip_rn50</strong>');
    expect(markup).toContain('Your original Lab result is unchanged.');
  });

  it('switches models without changing the focused upload identity', () => {
    const onSelectModel = vi.fn();
    const onFocusStimulus = vi.fn();
    const tree = SessionBanner({ session: multiModelSession(), selectedModel: 'clip_rn50',
      focusedStimulusId: 'stimulus-a', onFocusStimulus, onSelectModel });
    const button = elements(tree).find((element) => element.props['aria-label'] === 'View uploaded stimuli with efficient_net');
    button!.props.onClick!();
    expect(onSelectModel).toHaveBeenCalledWith('efficient_net');
    expect(onFocusStimulus).not.toHaveBeenCalled();
  });

  it('does not display another model’s upload means when viewing an unprepared model', () => {
    const markup = renderToStaticMarkup(
      <SessionBanner session={multiModelSession()} selectedModel="resnet50"
        focusedStimulusId="stimulus-a" onFocusStimulus={vi.fn()} onSelectModel={vi.fn()} />,
    );
    expect(markup).toContain('resnet50 · reference images only');
    expect(markup).toContain('Uploads are not predicted for this model.');
    expect(markup).not.toContain('class="session-roi-table"');
    expect(markup).not.toContain('<strong>0.800</strong>');
  });
});
