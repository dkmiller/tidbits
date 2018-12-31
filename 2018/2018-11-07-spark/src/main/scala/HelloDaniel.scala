package HelloDaniel

import com.typesafe.config.{Config, ConfigFactory}

/** Simple class for learning about Scala's features.
  */
object Main extends App {
    val p = Person("Daniel Miller")

    println(s"Hi from Scala (${p.name})!")

    val config = ConfigFactory.load("HelloDaniel.conf")
    val dummy = config.getString("dan.dummy")
    println(s"Value from config: '$dummy'.")
}
