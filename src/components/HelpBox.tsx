export function HelpBox() {
  return (
    <details className="help-box">
      <summary>Quick guide</summary>
      <div>
        <p>
          <strong>Heatmap:</strong> rows are models, columns are brain regions. NSD-trained models evaluated on Murty185.
          Numbers show Pearson r;
          colors show score relative to the noise ceiling. Overall alignment averages FFA, PPA, and EBA.
          Select a cell to explore predictions, not measured activity.
        </p>
        <p>
          <strong>Predicted responses:</strong> explore image rankings and response distributions;
          place uploads among Murty-185 references.
        </p>
        <p>
          <strong>Image similarity:</strong> group images by response patterns or visual features.
          Hover to preview; click to enlarge. Compare FFA · EBA · PPA
          links the same image across regions.
        </p>
        <p>
          <strong>Brain &amp; selectivity:</strong> map image predictions on cortex or group voxel profiles
          across the reference images.
        </p>
        <p>
          <strong>Compare selections:</strong> compare two cells’ image rankings and clusters.
          Enter selects; Escape closes.
        </p>
      </div>
    </details>
  );
}
