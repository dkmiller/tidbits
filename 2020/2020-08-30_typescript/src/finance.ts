// https://finnhub.io/docs/api

let api_key = "bt669r748v6oi7tnpb5g";

export function getStocks(symbol: string): Promise<string> {
    return fetch(`https://finnhub.io/api/v1/stock/profile2?symbol=${symbol}&token=${api_key}`)
        .then(r => r.json())
        .then(JSON.stringify)
}

export function displayStockInfo(doc: Document, id: string, symbol: string): Promise<void | string> {
    const element = doc.getElementById(id);
    const new_element = doc.createElement("p");
    new_element.innerText = `Loading ${symbol}...`;
    element.appendChild(new_element);

    return getStocks(symbol)
        .then(x => new_element.innerText = x);
}
