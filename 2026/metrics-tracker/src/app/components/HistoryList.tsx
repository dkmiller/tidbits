import { Clock, Tag } from 'lucide-react';

/* eslint-disable @typescript-eslint/no-explicit-any */
type Entry = {
    id: string;
    metric_name: string;
    value: number;
    dimensions: any;
    created_at: string; // ISO string
};
/* eslint-enable @typescript-eslint/no-explicit-any */

export default function HistoryList({ entries }: { entries: Entry[] }) {
    if (entries.length === 0) {
        return (
            <div className="text-center text-muted-foreground p-10 glass rounded-3xl">
                <p>No entries yet. Start tracking!</p>
            </div>
        );
    }

    return (
        <div className="grid gap-4 w-full max-w-lg mx-auto">
            {entries.map((entry) => (
                <div key={entry.id} className="glass p-5 rounded-2xl flex justify-between items-center group hover:bg-white/5 transition-colors">
                    <div>
                        <h3 className="font-semibold text-lg text-white">{entry.metric_name}</h3>
                        <div className="flex flex-wrap gap-2 mt-2">
                            {entry.dimensions && Object.entries(entry.dimensions).map(([k, v]) => (
                                <span key={k} className="text-xs bg-secondary/80 text-secondary-foreground px-2 py-1 rounded-md flex items-center gap-1">
                                    <Tag size={10} className="opacity-50" /> {k}: <span className="font-medium text-white">{String(v)}</span>
                                </span>
                            ))}
                        </div>
                        <div className="flex items-center gap-1 text-xs text-muted-foreground mt-3">
                            <Clock size={12} />
                            {new Date(entry.created_at).toLocaleString()}
                        </div>
                    </div>
                    <div className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-br from-white to-gray-400 font-mono">
                        {entry.value}
                    </div>
                </div>
            ))}
        </div>
    );
}
