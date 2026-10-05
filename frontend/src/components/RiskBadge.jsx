export default function RiskBadge({ level }) {
  return <span className={`badge ${level.toLowerCase()}`}>{level}</span>;
}
