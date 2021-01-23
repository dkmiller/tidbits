import { showGreeting } from "./greeter";
import { displayStockInfo } from "./finance";


showGreeting("greeting", "Type", "Script");


["MSFT", "GOOG", "NVDA"].forEach(
    symbol => displayStockInfo(document, "sample_p", symbol)
)

// const financeElt = document.getElementById("finance");
// getStocks("MSFT")
//     .then(x => financeElt.innerText = x );

// // getStocks("GOOG")
// //     .then(x => {
// //         const div = document.getElementById("sample_p");
// //         const otherElt = document.createElement("other");
// //         otherElt.innerText = x;
// //         div?.appendChild(otherElt);
// //     });

// displayStockInfo(document, "sample_p", "NVDA");
