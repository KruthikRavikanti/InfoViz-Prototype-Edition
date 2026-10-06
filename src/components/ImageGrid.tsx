import { useState, type CSSProperties } from 'react';
import type { EvidenceImage } from '../types/data';

type ImageGridProps = {
  title: string;
  images: EvidenceImage[];
  onOpenImage: (image: EvidenceImage) => void;
  compact?: boolean;
};

type ImageCardProps = {
  image: EvidenceImage;
  onOpenImage: (image: EvidenceImage) => void;
};

function formatValue(value: number): string {
  return value.toFixed(3);
}

function ImageCard({ image, onOpenImage }: ImageCardProps) {
  const [imageMissing, setImageMissing] = useState(false);

  return (
    <article className="evidence-image-card">
      <button
        className={`image-preview-button ${image.origin ? `image-preview-button--${image.origin}` : ''}`}
        type="button"
        onClick={() => onOpenImage(image)}
        style={image.markerColor ? { '--stimulus-color': image.markerColor } as CSSProperties : undefined}
      >
        {imageMissing ? (
          <span className="missing-image-fallback">Image unavailable</span>
        ) : (
          <img src={image.imageUrl} alt={image.imageName} loading="lazy" onError={() => setImageMissing(true)} />
        )}
        {image.origin === 'uploaded' && (
          <span className="image-origin-badge image-origin-badge--uploaded">
            Uploaded
          </span>
        )}
      </button>
      <div className="image-card-meta">
        <span>{image.imageName}</span>
        <strong>{image.valueLabel ?? formatValue(image.value)}</strong>
        {image.origin === 'uploaded' && image.sessionRank !== undefined && (
          <small>Reference {image.referencePercentile?.toFixed(1)}th pct · combined #{image.sessionRank}/{image.sessionCount}</small>
        )}
      </div>
    </article>
  );
}

export function ImageGrid({ title, images, onOpenImage, compact = false }: ImageGridProps) {
  return (
    <section className="evidence-grid-section" aria-label={title}>
      <div className="evidence-grid-heading">
        <h3>{title}</h3>
        <span>{images.length} images</span>
      </div>
      <div className={`evidence-image-grid${compact ? ' compact' : ''}`}>
        {images.map((image) => (
          <ImageCard key={`${title}-${image.stimulusId ?? image.imageName}-${image.rank}`} image={image} onOpenImage={onOpenImage} />
        ))}
      </div>
    </section>
  );
}
