package main

import (
	"flag"
	"fmt"

	"rsc.io/quote"
)

var (
	yamlFlag string
)

func main() {
	fmt.Println(quote.Go())
	flag.StringVar(&yamlFlag, "str", "example.yml", "Path to YAML file")
	fmt.Println(yamlFlag)
}
