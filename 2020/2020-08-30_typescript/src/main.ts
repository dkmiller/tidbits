import { greeter } from "./greeter";
import { getStocks } from "./finance";

function showGreeting(divName: string, firstName: string, lastName: string) {
    const elt = document.getElementById(divName);
    let innerText = greeter({ firstName: firstName, lastName: lastName });
    console.log(innerText);
    elt.innerText = innerText;
}

showGreeting("greeting", "Type", "Script");

const financeElt = document.getElementById("finance");
getStocks("MSFT")
    .then(x => financeElt.innerText = x );
