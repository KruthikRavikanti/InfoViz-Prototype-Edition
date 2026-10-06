const FORM_URL =
  'https://docs.google.com/forms/d/e/1FAIpQLScgwaGWRFbbhhG1cF3bL76moGiCR93AbqD-ZNZXtK8hdtsy6A/viewform?usp=publish-editor';
const EMBED_URL =
  'https://docs.google.com/forms/d/e/1FAIpQLScgwaGWRFbbhhG1cF3bL76moGiCR93AbqD-ZNZXtK8hdtsy6A/viewform?embedded=true';

export function UserStudyView() {
  return (
    <section className="user-study-panel" aria-labelledby="user-study-title">
      <div className="user-study-header">
        <div>
          <p className="eyebrow">User study</p>
          <h2 id="user-study-title">Study form</h2>
          <p>
            Return to the visualization at any time.
          </p>
        </div>
        <a href={FORM_URL} target="_blank" rel="noreferrer">
          Open in new tab
        </a>
      </div>
      <div className="form-frame-shell">
        <iframe
          src={EMBED_URL}
          title="Cortex visualization user study Google Form"
          loading="lazy"
        >
          Loading the user study form.
        </iframe>
      </div>
      <p className="form-fallback-note">
        Form not loading? Open it in a new tab.
      </p>
    </section>
  );
}
