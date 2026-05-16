import { StatsPanel } from "@/components/StatsPanel"

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <h1>Lagom Humanizer</h1>
      <StatsPanel showDetailed={true} refreshInterval={30} />
    </main>
  )
}
