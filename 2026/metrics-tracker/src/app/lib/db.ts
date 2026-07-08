import postgres from 'postgres';

const connectionString = process.env.SUPABASE_CONNECTION;

if (!connectionString) {
  throw new Error('Missing SUPABASE_CONNECTION environment variable');
}

const sql = postgres(connectionString);

export default sql;
