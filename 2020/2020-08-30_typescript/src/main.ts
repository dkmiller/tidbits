import { showGreeting } from "./greeter";
import { displayStockInfo, displayMultipleStockInfo } from "./finance";


showGreeting("greeting", "Type", "Script");

const elt = <HTMLInputElement>document.getElementById("symbols");
elt.value = "MSFT, GOOG, NVDA";

elt.addEventListener("submit", e => console.log(`log: ${e}`));

// let btn = document.getElementById("coolbutton");
// btn.addEventListener("click", (e:Event) => this.getTrainingName(4));

displayMultipleStockInfo(document, "finance", "TSLA, FB");

["MSFT", "GOOG", "NVDA"].forEach(
    symbol => displayStockInfo(document, "finance", symbol)
)
