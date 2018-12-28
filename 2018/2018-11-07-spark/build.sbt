name := "HelloDaniel"

version := "0.0"

// Note: sbt run fails if this is 2.10.
scalaVersion := "2.12.0"

// The book "Scala Cookbook" gives a wrong (out of date?) recipe.
libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.0" % Test

// set the main class for 'sbt run'
mainClass in (Compile, run) := Some("HelloDaniel.Main")