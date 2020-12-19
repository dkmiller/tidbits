// https://finnhub.io/docs/api

let api_key = "bt669r748v6oi7tnpb5g";

export function getStocks(symbol: string): Promise<string> {
    return fetch(`https://finnhub.io/api/v1/stock/profile2?symbol=${symbol}&token=${api_key}`)
    .then(r => r.json())
    .then(JSON.stringify)
}
