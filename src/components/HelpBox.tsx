export function HelpBox() {
  return (
    <details className="help-box">
      <summary>How to use this tool</summary>
      <div>
        <p>
          The heatmap is your entry point: columns are ROIs, rows are models, and color encodes the aggregate
          alignment score. Use the controls bar to search, sort, or filter the model set.
        </p>
        <p>
          Click any cell (or focus it and press Enter) to open image-level evidence in the right panel. Adjust
          the top-k / bottom-k selector to show more or fewer images per tier.
        </p>
        <p>
          Enable <strong>Compare mode</strong> to select two cells, then review their image overlap and score
          differences in the comparison overlay. Press Escape to dismiss.
        </p>
      </div>
    </details>
  );
}
