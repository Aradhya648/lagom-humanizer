export function StatsPanel({ showDetailed, refreshInterval }: { showDetailed: boolean; refreshInterval: number }) {
  return (
    <div className="rounded-lg border p-4">
      <p className="text-sm text-muted-foreground">Stats Panel (refresh: {refreshInterval}s)</p>
    </div>
  )
}
