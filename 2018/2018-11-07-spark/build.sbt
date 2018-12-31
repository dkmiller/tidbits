name := "HelloDaniel"

version := "0.0"

// Note: sbt run fails if this is 2.10.
scalaVersion := "2.12.0"

// The book "Scala Cookbook" gives a wrong (out of date?) recipe.
libraryDependencies ++= Seq(
    "com.typesafe" % "config" % "1.3.3",
    "org.scalatest" %% "scalatest" % "3.0.0" % Test
)

// set the main class for 'sbt run' and 'sbt package'.
mainClass in (Compile, run) := Some("HelloDaniel.Main")
mainClass in (Compile, packageBin) := Some("HelloDaniel.Main")