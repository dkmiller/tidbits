package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/gin-gonic/gin"
	"gopkg.in/yaml.v3"
)

type Config struct {
	Port   int    `yaml:"port"`
	Target string `yaml:"target"`
}

func main() {
	fmt.Println("Hello, World!")
	var configFlag = flag.String("config", "config.yaml", "Path to configuration file")
	flag.Parse()
	fmt.Println(*configFlag)
	f, err := os.ReadFile(*configFlag)
	if err != nil {
		log.Fatal(err)
	}

	var conf Config

	// Unmarshal our input YAML file into empty Car (var c)
	if err := yaml.Unmarshal(f, &conf); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("%+v\n", conf)

	r := gin.Default()
	r.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"ok": true,
		})
	})

	// TODO: Swagger page seems PITA.

	// http://localhost:1233/health
	var address = fmt.Sprintf("localhost:%d", conf.Port)
	r.Run(address)
}
