'use server';

import sql from './lib/db';
import { revalidatePath } from 'next/cache';

/* eslint-disable @typescript-eslint/no-explicit-any */
export type Entry = {
    id: string;
    metric_name: string;
    value: number;
    dimensions: any;
    created_at: string;
};
/* eslint-enable @typescript-eslint/no-explicit-any */

export async function addEntry(formData: FormData) {
    const metricName = formData.get('metric_name') as string;
    const value = parseFloat(formData.get('value') as string);

    const dimensionsRaw = formData.get('dimensions') as string;
    let dimensions = {};
    if (dimensionsRaw) {
        try {
            dimensions = JSON.parse(dimensionsRaw);
        } catch (e) {
            console.error("Failed to parse dimensions", e);
        }
    }

    if (!metricName || isNaN(value)) {
        throw new Error('Invalid input');
    }

    await sql`
    INSERT INTO entries (metric_name, value, dimensions)
    VALUES (${metricName}, ${value}, ${sql.json(dimensions)})
  `;

    revalidatePath('/');
}

export async function getEntries(limit = 20): Promise<Entry[]> {
    const entries = await sql`
    SELECT * FROM entries
    ORDER BY created_at DESC
    LIMIT ${limit}
  `;

    // Cast and convert
    return entries.map(e => ({
        id: e.id,
        metric_name: e.metric_name,
        value: parseFloat(e.value), // Ensure number
        dimensions: e.dimensions,
        created_at: e.created_at.toISOString(),
    }));
}

export async function getSuggestions(query: string) {
    if (!query) return [];

    const metrics = await sql`
    SELECT DISTINCT metric_name
    FROM entries
    WHERE metric_name ILIKE ${'%' + query + '%'}
    LIMIT 5
  `;

    return metrics.map(m => m.metric_name);
}
