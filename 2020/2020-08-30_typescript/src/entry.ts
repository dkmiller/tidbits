export interface Configuration {
    readonly initialDivId: string;
    readonly initialSymbols: string;
}

export class Run {
    config: Configuration;

    constructor(config: Configuration) {
        this.config = config;
    }

    public run(): void {
        console.log(`Symbols: ${this.config.initialSymbols}`);

        const elt = this.initialElement();
        const new_element = document.createElement("p");
        new_element.innerText = `!! Loading ${this.config.initialSymbols}...`;
        elt.appendChild(new_element);
    }

    initialElement(): HTMLInputElement {
        return <HTMLInputElement>document.getElementById(this.config.initialDivId);
    }
}