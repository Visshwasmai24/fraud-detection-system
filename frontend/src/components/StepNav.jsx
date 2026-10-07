// Back / Next buttons shown at the bottom of each step page.
export default function StepNav({ onBack, onNext, backLabel, nextLabel }) {
  return (
    <div className="stepnav">
      <button onClick={onBack}>← {backLabel}</button>
      {onNext && <button className="primary" onClick={onNext}>{nextLabel} →</button>}
    </div>
  );
}
