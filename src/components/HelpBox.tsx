export function HelpBox() {
  return (
    <details className="help-box">
      <summary>How to use this prototype</summary>
      <div>
        <p>
          Start with the heatmap: columns are ROIs, rows are models, and color encodes the aggregate JSON score. Use
          search and sort controls to narrow the model set.
        </p>
        <p>
          Click a cell, or focus it and press Enter, to open image-level evidence in the details panel. Turn on compare
          mode in the control bar to select two cells and open a comparison pop-up with overlap and evidence details.
        </p>
        <p>
          Image-level evidence is currently available for only part of the aggregate score matrix. Cells without matching
          CSV evidence still appear in the heatmap, but compare mode only enables evidence-backed selections.
        </p>
      </div>
    </details>
  );
}
