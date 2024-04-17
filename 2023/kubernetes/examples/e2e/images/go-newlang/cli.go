package main

import "flag"
import "fmt"

func main() {
	fmt.Println("Hello, World!")
	var configFlag = flag.String("config", "config.yaml", "Path to configuration file")
	flag.Parse()
	fmt.Println(*configFlag)
}
