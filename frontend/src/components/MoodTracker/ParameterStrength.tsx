export function ParameterStrength({
  label,
  value,
}: {
  label: string;
  value: number;
}) {
  return (
    <div className="parameter">
      <div>
        <span>{label}</span>
        <span>{value.toFixed(2)}</span>
      </div>
      <progress aria-label={label} value={value} max={1} />
    </div>
  );
}
