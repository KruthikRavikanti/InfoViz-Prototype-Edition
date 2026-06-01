export function HelpBox() {
  return (
    <details className="help-box">
      <summary>How to use this dashboard</summary>
      <div>
        <p>
          The heatmap is your primary overview: columns are ROIs, rows are models, and color encodes the aggregate score.
          Use the controls bar to search, sort, and filter the model set.
        </p>
        <p>
          Click any cell (or focus it and press Enter) to open image-level evidence in the right panel. Use the top/bottom
          tier selector to control how many images are shown.
        </p>
        <p>
          Enable compare mode to select two cells and open a comparison overlay with image overlap and score difference analysis.
          Press Escape to close the comparison.
        </p>
      </div>
    </details>
  );
}
