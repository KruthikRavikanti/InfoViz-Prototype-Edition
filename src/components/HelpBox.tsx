export function HelpBox() {
  return (
    <details className="help-box">
      <summary>Quick guide</summary>
      <div>
        <p>
          <strong>Heatmap:</strong> rows are models, columns are brain regions, and color shows model alignment.
          Select a cell to explore image responses and clusters.
        </p>
        <p>
          <strong>Image clusters:</strong> hover to preview; click to enlarge. Compare FFA · EBA · PPA
          links the same image across regions.
        </p>
        <p>
          <strong>Brain &amp; voxels:</strong> map image predictions on cortex or group voxel profiles
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
