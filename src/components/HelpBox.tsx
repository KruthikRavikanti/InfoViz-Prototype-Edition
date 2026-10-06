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
          traces the same image across three regions.
        </p>
        <p>
          <strong>Compare selections:</strong> choose two cells to compare rankings, image overlap, and clusters.
          Use Enter to select a focused cell and Escape to close a comparison.
        </p>
      </div>
    </details>
  );
}
