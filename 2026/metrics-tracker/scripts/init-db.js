import postgres from 'postgres';
import dotenv from 'dotenv';
import path from 'path';

// Load .env from project root
dotenv.config({ path: path.resolve(process.cwd(), '.env') });

const connectionString = process.env.SUPABASE_CONNECTION;

if (!connectionString) {
  console.error('Error: SUPABASE_CONNECTION not found in .env');
  process.exit(1);
}

const sql = postgres(connectionString);

async function init() {
  try {
    console.log('Creating entries table...');
    await sql`
      CREATE TABLE IF NOT EXISTS entries (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        metric_name TEXT NOT NULL,
        value NUMERIC NOT NULL,
        dimensions JSONB DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ DEFAULT NOW()
      );
    `;
    console.log('✅ entries table ready.');
    process.exit(0);
  } catch (err) {
    console.error('❌ Failed to init DB:', err);
    process.exit(1);
  }
}

init();
