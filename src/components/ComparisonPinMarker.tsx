/** Neutral pin outlines preserve the response-gap or cluster color at the center. */
export function ComparisonPinMarker({
  index, x, y, color, label, labelSide = 'right', labelDy = -12,
}: {
  index: number;
  x: number;
  y: number;
  color?: string;
  label?: string;
  labelSide?: 'left' | 'right';
  labelDy?: number;
}) {
  const diamond = index === 1;
  const path = `M ${x} ${y - 9} L ${x + 9} ${y} L ${x} ${y + 9} L ${x - 9} ${y} Z`;
  return (
    <g
      className="comparison-pin-marker"
      data-pin-index={index}
      data-pin-shape={diamond ? 'diamond' : 'circle'}
      aria-label={label ? `Pinned image ${label}` : undefined}
      role={label ? 'img' : undefined}
      style={{ pointerEvents: 'none' }}
    >
      {color && <circle cx={x} cy={y} r="3.8" fill={color} style={{ stroke: 'none', opacity: 1 }} />}
      {diamond ? (
        <>
          <path d={path} fill="none" stroke="white" strokeWidth="5" />
          <path d={path} fill="none" stroke="#242424" strokeWidth="1.8" />
        </>
      ) : (
        <>
          <circle cx={x} cy={y} r="8.5" fill="none" stroke="white" strokeWidth="5" style={{ stroke: 'white', strokeWidth: 5, opacity: 1 }} />
          <circle cx={x} cy={y} r="8.5" fill="none" stroke="#242424" strokeWidth="1.8" style={{ stroke: '#242424', strokeWidth: 1.8, opacity: 1 }} />
        </>
      )}
      {label && (
        <text
          x={x + (labelSide === 'left' ? -13 : 13)}
          y={y + labelDy}
          textAnchor={labelSide === 'left' ? 'end' : 'start'}
          fill="#242424"
          stroke="white"
          strokeWidth="3.5"
          paintOrder="stroke"
          fontFamily="inherit"
          fontWeight="700"
          fontSize="12"
        >{label}</text>
      )}
    </g>
  );
}
