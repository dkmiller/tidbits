import { getEntries } from '@/app/actions';
import MetricForm from '@/app/components/MetricForm';
import HistoryList from '@/app/components/HistoryList';

export const dynamic = 'force-dynamic';

export default async function Home() {
  const entries = await getEntries();

  return (
    <main className="min-h-screen p-4 md:p-8 relative overflow-hidden">
      {/* Dynamic Background */}
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden -z-10 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-slate-900 via-background to-background">
        <div className="absolute top-[-10%] right-[-10%] w-[500px] h-[500px] bg-primary/20 rounded-full blur-[100px] opacity-50 animate-pulse" />
        <div className="absolute bottom-[-10%] left-[-10%] w-[500px] h-[500px] bg-accent/10 rounded-full blur-[100px] opacity-30" />
      </div>

      <div className="max-w-2xl mx-auto space-y-10 pt-10">
        <header className="text-center space-y-2">
          <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-primary to-accent">
            Metrics Tracker
          </h1>
          <p className="text-muted-foreground">Log your progress, reach your goals.</p>
        </header>

        <section>
          <MetricForm />
        </section>

        <section>
          <h3 className="text-xl font-semibold mb-4 px-2 text-gray-300">
            Recent Activity
          </h3>
          <HistoryList entries={entries} />
        </section>
      </div>
    </main>
  );
}
